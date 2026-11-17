# Unit tests for mot_eval.exe on synthetic cases with hand-computed metrics.
#
# Usage: c:\python\python310\python.exe tools\mot_eval\tests_mot_eval.py

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOT_EVAL = os.path.join(ROOT, "build", "Release", "mot_eval.exe")

INT_KEYS = ["frames", "objects", "preds", "matches", "fn", "fp", "suppressed", "idsw", "idtp"]
FLOAT_KEYS = ["mota", "motp", "idf1"]


def run(gt_rows, res_rows, extra_args=()):
    tmp = tempfile.mkdtemp(prefix="mot_test_")
    gt_path = os.path.join(tmp, "gt.txt")
    res_path = os.path.join(tmp, "res.txt")
    with open(gt_path, "w") as f:
        f.writelines(r + "\n" for r in gt_rows)
    with open(res_path, "w") as f:
        f.writelines(r + "\n" for r in res_rows)
    proc = subprocess.run([MOT_EVAL, "--gt=" + gt_path, "--res=" + res_path,
                           "--label=test", *extra_args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("mot_eval failed:\n" + proc.stdout + proc.stderr)
    out = proc.stdout
    got = {}
    m = re.search(r"IDTP/IDFP/IDFN\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", out)
    got["idtp"], got["idfp"], got["idfn"] = (int(x) for x in m.groups())
    for key, name in (("IDSW", "idsw"), ("GT objects", "objects"), ("predictions", "preds"),
                      ("matches", "matches"), ("FN (misses)", "fn"), ("FP", "fp"),
                      ("FP suppressed", "suppressed"), ("frames", "frames")):
        got[name] = int(re.search(re.escape(key) + r"\s+([-\d.]+)", out).group(1))
    for key, name in (("MOTA", "mota"), ("MOTP (1-IoU)", "motp"), ("IDF1", "idf1")):
        got[name] = float(re.search(re.escape(key) + r"\s+([-\d.]+)", out).group(1))
    return got


def check(name, got, **expected):
    failures = []
    for key, val in expected.items():
        actual = got[key]
        tol = 5e-4 if key in FLOAT_KEYS else 0
        if abs(actual - val) > tol:
            failures.append(f"    {key}: expected {val}, got {actual}")
    print(f"  {name}: {'FAIL' if failures else 'OK'}")
    for f in failures:
        print(f)
    return not failures


def main():
    if not os.path.isfile(MOT_EVAL):
        print(f"mot_eval.exe not found: {MOT_EVAL}")
        return 1
    ok = True

    # A. Perfect match: 1 track, 2 frames.
    gt = ["1,1,0,0,10,10,1,1,0,0", "2,1,0,0,10,10,1,1,0,0"]
    ok &= check("perfect", run(gt, gt), frames=2, objects=2, preds=2, matches=2,
                fn=0, fp=0, idsw=0, idtp=2, mota=1.0, motp=0.0, idf1=1.0)

    # B. FN: the second frame detection is missing.
    #    CLEAR: matches=1, FN=1, MOTA=1-1/2=0.5.
    #    ID: oc=2, hc=1, tps(1,1)=1 -> idfn=1, idfp=0, IDTP=1, IDF1=2/3.
    res = ["1,1,0,0,10,10,1,-1,-1,-1"]
    ok &= check("one_fn", run(gt, res), matches=1, fn=1, fp=0, idsw=0, idtp=1,
                mota=0.5, idf1=2.0 / 3.0)

    # C. FP: an extra far box under a new id in frame 2.
    #    CLEAR: FP=1, MOTA=0.5. ID: hc(id5)=1 unmatched -> idfp=1,
    #    IDF1=2*2/(2+3)=0.8.
    res = gt[:1] + ["2,1,0,0,10,10,1,-1,-1,-1", "2,5,500,500,10,10,1,-1,-1,-1"]
    ok &= check("one_fp", run(gt, res), matches=2, fn=0, fp=1, idtp=2, idfp=1,
                mota=0.5, idf1=0.8)

    # D. ID switch: gt is one continuous track, res splits it into id1 (f1-2)
    #    and id2 (f3-4). IDSW=1, MOTA=1-1/4=0.75, IDF1=2*2/(4+4)=0.5.
    gt = ["1,1,0,0,10,10,1,1,0,0", "2,1,0,0,10,10,1,1,0,0",
          "3,1,0,0,10,10,1,1,0,0", "4,1,0,0,10,10,1,1,0,0"]
    res = ["1,1,0,0,10,10,1,-1,-1,-1", "2,1,0,0,10,10,1,-1,-1,-1",
           "3,2,0,0,10,10,1,-1,-1,-1", "4,2,0,0,10,10,1,-1,-1,-1"]
    ok &= check("id_switch", run(gt, res), matches=4, fn=0, fp=0, idsw=1, idtp=2,
                idfp=2, idfn=2, mota=0.75, idf1=0.5)

    # E. Ignore-class suppression: gt has class 1 (evaluated) and class 8
    #    (ignore) boxes; a stray hyp inside the ignore box is not an FP.
    #    CLEAR: matches=1, FN=1, FP=0, suppressed=1, MOTA=0.5.
    #    ID: IDF1=2*1/(2+2)=0.5.
    gt = ["1,1,0,0,10,10,1,1,0,0", "2,1,0,0,10,10,1,1,0,0",
          "1,7,100,100,50,50,1,8,0,0", "2,7,100,100,50,50,1,8,0,0"]
    res = ["1,1,0,0,10,10,1,-1,-1,-1", "2,2,105,105,40,40,1,-1,-1,-1"]
    ok &= check("ignore_suppress",
                run(gt, res, ("--gt_classes=1", "--gt_ignore_classes=8")),
                objects=2, matches=1, fn=1, fp=0, suppressed=1, idtp=1, idfp=1,
                mota=0.5, idf1=0.5)

    # F. Dropped class: class 3 rows are neither evaluated nor ignored: not
    #    counted as GT, their hyps are NOT suppressed (plain FP).
    gt = ["1,1,0,0,10,10,1,1,0,0", "1,9,200,200,30,30,1,3,0,0"]
    res = ["1,1,0,0,10,10,1,-1,-1,-1", "1,5,205,205,25,25,1,-1,-1,-1"]
    ok &= check("dropped_class",
                run(gt, res, ("--gt_classes=1", "--gt_ignore_classes=8")),
                objects=1, preds=2, matches=1, fn=0, fp=1, suppressed=0,
                idtp=1, idfp=1, mota=0.0, idf1=2.0 / 3.0)

    # G. Clamping: a result box entirely outside the frame is dropped before
    #    counting.
    gt = ["1,1,0,0,10,10,1,1,0,0"]
    res = ["1,1,0,0,10,10,1,-1,-1,-1", "1,5,2000,2000,50,50,1,-1,-1,-1"]
    ok &= check("clamped_fp", run(gt, res, ("--width=1920", "--height=1080")),
                objects=1, preds=1, matches=1, fp=0, idtp=1, idf1=1.0)

    # H. Center-inside suppression: the hyp box is much smaller than the
    #    ignore region (plain IoU would be ~0.1), but its center is inside.
    gt = ["1,1,0,0,10,10,1,1,0,0", "1,7,100,100,300,300,1,8,0,0"]
    res = ["1,1,0,0,10,10,1,-1,-1,-1", "1,5,200,200,30,30,1,-1,-1,-1"]
    ok &= check("center_inside_suppress",
                run(gt, res, ("--gt_classes=1", "--gt_ignore_classes=8")),
                objects=1, preds=2, matches=1, fp=0, suppressed=1,
                idtp=1, idfp=1, mota=1.0)

    # I. min_area: result boxes below the threshold are dropped before
    #    counting (VisDrone practice: unannotated small objects).
    gt = ["1,1,0,0,100,100,1,1,0,0", "1,7,300,300,50,50,1,8,0,0"]
    res = ["1,1,0,0,100,100,1,-1,-1,-1", "1,5,305,305,10,10,1,-1,-1,-1"]
    ok &= check("min_area",
                run(gt, res, ("--gt_classes=1", "--gt_ignore_classes=8",
                              "--min_area=500")),
                objects=1, preds=1, matches=1, fp=0, suppressed=0,
                idtp=1, idfp=0, mota=1.0, idf1=1.0)

    print("ALL OK" if ok else "TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
