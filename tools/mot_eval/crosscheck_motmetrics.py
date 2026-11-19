# Cross-check of mot_eval.exe against py-motmetrics.
#
# Usage:
#   c:\python\python310\python.exe tools\mot_eval\crosscheck_motmetrics.py
#
# Cases:
#   1. GT vs GT (MOT20-05, truncated to --frames)  -> all metrics perfect
#   2. Perturbed GT vs GT (injected FN / FP / IDSW) -> metrics must match
#
# The tool is invoked as ./build/Release/mot_eval.exe with plain options
# (no class filtering: py-motmetrics evaluates every GT row).

import argparse
import os
import random
import re
import subprocess
import sys
import tempfile

import numpy as np
# motmetrics 1.4.0 uses np.asfarray, removed in NumPy 2.x.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
import motmetrics as mm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOT_EVAL = os.path.join(ROOT, "build", "Release", "mot_eval.exe")
GT = r"C:\work\dataset\motchallenge\MOT20Labels\train\MOT20-05\gt\gt.txt"

METRICS = ["mota", "motp", "idf1", "num_switches", "num_objects", "num_predictions"]


def py_metrics(gt_path, res_path):
    gt = mm.io.loadtxt(gt_path)
    ts = mm.io.loadtxt(res_path)
    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou")
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=METRICS, name="seq")
    s = summary.iloc[0]
    return {
        "mota": float(s["mota"]),
        "motp": float(s["motp"]),
        "idf1": float(s["idf1"]),
        "idsw": int(s["num_switches"]),
        "objects": int(s["num_objects"]),
        "preds": int(s["num_predictions"]),
    }


def my_metrics(gt_path, res_path):
    out = subprocess.run([MOT_EVAL, "--gt=" + gt_path, "--res=" + res_path, "--label=cross"],
                         capture_output=True, text=True, check=True).stdout
    vals = {}
    m = re.search(r"IDTP/IDFP/IDFN\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", out)
    vals["idtp"], vals["idfp"], vals["idfn"] = (int(x) for x in m.groups())
    for key, name in (("MOTA", "mota"), ("MOTP (1-IoU)", "motp"), ("IDF1", "idf1")):
        vals[name] = float(re.search(re.escape(key) + r"\s+([-\d.]+)", out).group(1))
    for key, name in (("IDSW", "idsw"), ("GT objects", "objects"), ("predictions", "preds")):
        vals[name] = int(re.search(re.escape(key) + r"\s+([-\d.]+)", out).group(1))
    return {
        "mota": vals["mota"],
        "motp": vals["motp"],
        "idf1": vals["idf1"],
        "idsw": vals["idsw"],
        "objects": vals["objects"],
        "preds": vals["preds"],
    }


def truncate_gt(src, dst, frames):
    kept = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if int(line.split(",")[0]) <= frames:
                fout.write(line)
                kept += 1
    return kept


def perturb(src, dst, frames, seed=7):
    """Perturbed results vs the original GT: drop 3% rows (FN), add shifted
    copies under NEW ids (FP), and change the id of 5% of the tracks from a
    random split point on (IDSW)."""
    rng = random.Random(seed)
    rows = []
    with open(src) as fin:
        for line in fin:
            if int(line.split(",")[0]) <= frames:
                rows.append(line.rstrip("\n").split(","))
    ids = sorted({r[1] for r in rows})
    victims = set(rng.sample(ids, max(1, len(ids) // 20)))  # 5% of tracks switch id
    split = {tid: rng.randint(20, max(21, frames - 5)) for tid in victims}
    new_id = 9000
    out = []
    for r in rows:
        f, tid, x, y, w, h, conf, cls = r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        if tid in victims and int(f) > split[tid]:
            tid = str(int(tid) + 5000)
        p = rng.random()
        if p < 0.03:  # FN: the detection is missing
            continue
        out.append([f, tid, x, y, w, h, conf, cls])
        if p > 0.985:  # FP: a shifted box under a fresh id
            dx, dy = rng.randint(-120, 120), rng.randint(-60, 60)
            new_id += 1
            out.append([f, str(new_id), str(int(x) + dx), str(int(y) + dy), w, h, conf, cls])
    with open(dst, "w") as fout:
        for r in out:
            fout.write(",".join(r) + "\n")
    return len(out)


def compare(name, gt_path, res_path):
    a = py_metrics(gt_path, res_path)
    b = my_metrics(gt_path, res_path)
    ok = True
    for key, tol in (("mota", 1e-3), ("motp", 1e-3), ("idf1", 1e-3), ("idsw", 0),
                     ("objects", 0), ("preds", 0)):
        diff = abs(a[key] - b[key])
        good = diff <= tol
        ok &= good
        print(f"  {name} {key}: py={a[key]:.6f} mot_eval={b[key]:.6f} diff={diff:.6f} {'OK' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    if not os.path.isfile(MOT_EVAL):
        print(f"mot_eval.exe not found: {MOT_EVAL}")
        return 1

    tmp = tempfile.mkdtemp(prefix="mot_cross_")
    gt_small = os.path.join(tmp, "gt_small.txt")

    ok = True

    n = truncate_gt(GT, gt_small, args.frames)
    print(f"[1] GT vs GT, first {args.frames} frames ({n} rows)")
    ok &= compare("self", gt_small, gt_small)

    gt_pert = os.path.join(tmp, "gt_pert.txt")
    perturb(GT, gt_pert, args.frames)
    print(f"[2] perturbed results vs original GT")
    ok &= compare("perturbed", gt_small, gt_pert)

    print("ALL OK" if ok else "MISMATCH FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
