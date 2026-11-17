#!/usr/bin/env python3
"""Visual inspection of MOT benchmark runs.

Feeds a tracker result file (MOTChallenge format, as written by
MultitargetTracker --mot_res) together with the matching image sequence and
ground truth, and shows every frame with:

  * blue   - ground truth boxes of the evaluated classes;
  * orange - GT that got no matching detection this frame (FN), drawn as
             corner ticks around the blue GT box;
  * green  - detections matched to GT (IoU >= threshold) with confidence;
  * red    - unmatched detections (FP) with confidence;
  * gray   - GT boxes of the ignore classes (they suppress FPs in mot_eval).

HUD: "<frame> / <total>" in the top right corner, sequence name and per-frame
counts (gt/fn/match/fp/det) in the top left, color legend at the bottom.

Keys: RIGHT - next frame, LEFT - previous frame, HOME - first frame,
END - last frame, SPACE - pause/resume auto-play, ESC - quit.

Usage:
  python tools\\mot_viz\\mot_viz.py --res C:\\work\\home\\mtracker\\results\\baseline1\\MOT20-05.txt
  python tools\\mot_viz\\mot_viz.py --res ...\\uav0000117_02622_v.txt --scale 1.0

Video and GT paths are discovered automatically from the result file name
(MOT20 train and VisDrone2019-MOT-val layout); --video_pattern / --gt override
them. Optionally --dump_frame N --dump_out out.png renders a single frame to a
file without opening a window (for scripted checks).
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2

# --------------------------------------------------------------------------
# Dataset layouts
# --------------------------------------------------------------------------

MOT20_NAMES = {
    1: "pedestrian", 2: "person_veh", 3: "car", 4: "bicycle",
    5: "motorbike", 6: "nonmot_veh", 7: "distractor", 8: "occluder",
    9: "occ_ground", 10: "occ_full", 11: "occ_partial", 12: "occ_behind",
}
VISDRONE_NAMES = {
    0: "ignore_region", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle", 8: "awning_tri",
    9: "bus", 10: "motor", 11: "others",
}

DATASETS = [
    {
        "name": "MOT20",
        "img_pattern": r"C:\work\dataset\motchallenge\MOT20\train\{seq}\img1\%06d.jpg",
        "gt_path": r"C:\work\dataset\motchallenge\MOT20Labels\train\{seq}\gt\gt.txt",
        "eval_classes": {1},
        "ignore_classes": {2, 7, 8, 9, 10, 11, 12},
        "min_area": 0,
        "names": MOT20_NAMES,
    },
    {
        "name": "VisDrone",
        "img_pattern": r"C:\work\dataset\visdrone\mot_2019\VisDrone2019-MOT-val\sequences\{seq}\%07d.jpg",
        "gt_path": r"C:\work\dataset\visdrone\mot_2019\VisDrone2019-MOT-val\annotations\{seq}.txt",
        "eval_classes": set(range(1, 11)),
        "ignore_classes": {0, 11},
        "min_area": 800,
        "names": VISDRONE_NAMES,
    },
]

# BGR colors
C_BLUE = (255, 0, 0)
C_ORANGE = (0, 140, 255)
C_GREEN = (0, 200, 0)
C_RED = (0, 0, 255)
C_GRAY = (160, 160, 160)
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def discover_dataset(seq):
    """Find a dataset config whose GT file and image dir exist for seq."""
    for cfg in DATASETS:
        gt = cfg["gt_path"].format(seq=seq)
        pattern = cfg["img_pattern"].format(seq=seq)
        if os.path.exists(gt) and os.path.exists(os.path.dirname(pattern)):
            return cfg, gt, pattern
    return None, None, None


def load_results(path):
    """MOTChallenge result rows: frame,id,x,y,w,h,conf,-1,-1,-1 (1-based)."""
    res = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            try:
                frame = int(float(p[0]))
                x, y = int(float(p[2])), int(float(p[3]))
                w, h = int(float(p[4])), int(float(p[5]))
                conf = float(p[6]) if len(p) > 6 else 1.0
                tid = int(float(p[1]))
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            res[frame].append({"id": tid, "box": (x, y, w, h), "conf": conf})
    return res


def load_gt(path, cfg):
    """GT rows: frame,id,x,y,w,h,conf,class,... (1-based, class = col 8)."""
    gt = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            try:
                frame = int(float(p[0]))
                x, y = int(float(p[2])), int(float(p[3]))
                w, h = int(float(p[4])), int(float(p[5]))
                cls = int(float(p[7]))
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            kind = ("eval" if cls in cfg["eval_classes"] else
                    "ignore" if cls in cfg["ignore_classes"] else "drop")
            if kind == "drop":  # never shown, never matched (mot_eval semantics)
                continue
            gt[frame].append({"cls": cls, "box": (x, y, w, h), "kind": kind})
    return gt


# --------------------------------------------------------------------------
# Matching / rendering
# --------------------------------------------------------------------------

def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def match_frame(gts, dets, iou_thr):
    """Greedy IoU matching (display aid, not the continuity-aware CLEAR MOT)."""
    pairs = []
    for gi, g in enumerate(gts):
        for di, d in enumerate(dets):
            v = iou(g["box"], d["box"])
            if v >= iou_thr:
                pairs.append((v, gi, di))
    pairs.sort(reverse=True)
    g_used, d_used = set(), set()
    for _, gi, di in pairs:
        if gi in g_used or di in d_used:
            continue
        g_used.add(gi)
        d_used.add(di)
    return g_used, d_used


def draw_box(img, box, color, thickness, label=None):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label:
        draw_label(img, x, y, label, color)


def draw_label(img, x, y, text, color):
    """Label above the box, filled black background for readability."""
    (tw, th), base = cv2.getTextSize(text, FONT, 0.42, 1)
    ty = y - base - 2
    if ty - th < 0:  # fold inside the frame
        ty = y + th + base + 2
    cv2.rectangle(img, (x, ty - th), (x + tw + 2, ty + base), C_BLACK, -1)
    cv2.putText(img, text, (x + 1, ty), FONT, 0.42, color, 1, cv2.LINE_AA)


def draw_fn_ticks(img, box, color, length=14, inset=4, thickness=2):
    """Orange corner ticks just outside the blue GT box: FN marker."""
    x, y, w, h = box
    x0, y0 = x - inset, y - inset
    x1, y1 = x + w + inset, y + h + inset
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness)


def put_text(img, text, org, color=C_WHITE, scale=0.5, shadow=True):
    if shadow:
        cv2.putText(img, text, (org[0] + 1, org[1] + 1), FONT, scale,
                    C_BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def render_frame(frame, frame_no, gts, dets, cfg, iou_thr, min_area,
                 show_labels=True):
    """Draw boxes on the full-res frame; returns the annotated frame."""
    # detections: min_area filter first, same as mot_eval
    dets = [d for d in dets
            if min_area <= 0 or d["box"][2] * d["box"][3] >= min_area]
    evals = [g for g in gts if g["kind"] == "eval"]
    ignores = [g for g in gts if g["kind"] == "ignore"]
    # only eval-class GT matches detections; ignore boxes never do (they are
    # FP suppressors in mot_eval, not match targets)
    g_used, d_used = match_frame(evals, dets, iou_thr)

    # GT: ignore classes in gray first (under everything else)
    for g in ignores:
        name = cfg["names"].get(g["cls"], str(g["cls"]))
        draw_box(frame, g["box"], C_GRAY, 1,
                 "ign " + name if show_labels else None)
    for gi, g in enumerate(evals):
        name = cfg["names"].get(g["cls"], str(g["cls"]))
        draw_box(frame, g["box"], C_BLUE, 2, name if show_labels else None)
        if gi not in g_used:  # FN marker
            draw_fn_ticks(frame, g["box"], C_ORANGE)

    # detections: green when matched, red when not
    for di, d in enumerate(dets):
        label = "%.2f" % d["conf"]
        if di in d_used:
            draw_box(frame, d["box"], C_GREEN, 2, label)
        else:
            draw_box(frame, d["box"], C_RED, 1, label)

    stats = (len(evals), len(evals) - len(g_used), len(d_used),
             len(dets) - len(d_used), len(dets))
    return frame, stats


def render_hud(disp, frame_no, total, seq, stats):
    """HUD on the (already scaled) display image: counter, name, legend."""
    h, w = disp.shape[:2]
    put_text(disp, "%d / %d" % (frame_no, total), (w - 150, 28), scale=0.55)
    gt_n, fn, mt, fp, det_n = stats
    put_text(disp, seq, (10, 28), scale=0.55)
    put_text(disp, "gt=%d fn=%d match=%d fp=%d det=%d" % (gt_n, fn, mt, fp, det_n),
             (10, 52), scale=0.45)

    # legend strip at the bottom
    legend = "GT blue | GT miss orange | det ok green | det FP red | ignore gray"
    (tw, th), base = cv2.getTextSize(legend, FONT, 0.45, 1)
    cv2.rectangle(disp, (0, h - th - base - 10), (tw + 10, h), C_BLACK, -1)
    cv2.putText(disp, legend, (5, h - 6), FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Video reading with reliable random access
# --------------------------------------------------------------------------

class VideoReader:
    def __init__(self, path):
        self.path = path.replace("\\", "/")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise SystemExit("cannot open video: %s" % path)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get(self, idx):
        """Frame at 0-based idx; reopens and re-seeks if the backend drifts."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = self.cap.read()
        if ok:
            return img
        # fallback: reopen and walk (image-sequence backends can refuse seeks)
        self.cap.release()
        self.cap = cv2.VideoCapture(self.path)
        for _ in range(idx):
            self.cap.grab()
        ok, img = self.cap.read()
        return img if ok else None

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True, help="tracker result file (.txt)")
    ap.add_argument("--gt", help="override ground truth file")
    ap.add_argument("--video_pattern", help="override image sequence pattern, e.g. ...img1/%%06d.jpg")
    ap.add_argument("--iou", type=float, default=0.5, help="match IoU threshold (default 0.5)")
    ap.add_argument("--min_area", type=int, default=-1,
                    help="drop result boxes below this area; -1 = dataset default")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="display scale; 0 = fit into 1600x950")
    ap.add_argument("--no_labels", action="store_true",
                    help="hide GT/ignore text labels (dense scenes); "
                         "detection confidences are still shown")
    ap.add_argument("--dump_frame", type=int, default=0,
                    help="render this 1-based frame to --dump_out and exit (no GUI)")
    ap.add_argument("--dump_out", default="mot_viz_dump.png",
                    help="output file for --dump_frame")
    args = ap.parse_args()

    seq = os.path.splitext(os.path.basename(args.res))[0]
    if args.gt and args.video_pattern:
        gt, pattern = args.gt, args.video_pattern
        cfg = DATASETS[1] if "visdrone" in gt.lower() else DATASETS[0]
    else:
        cfg, gt, pattern = discover_dataset(seq)
        if args.gt:
            gt = args.gt
        if args.video_pattern:
            pattern = args.video_pattern
        if cfg is None:
            raise SystemExit(
                "cannot find dataset layout for '%s'; pass --gt and --video_pattern" % seq)

    min_area = cfg["min_area"] if args.min_area < 0 else args.min_area
    print("seq=%s dataset=%s" % (seq, cfg["name"]))
    print("video=%s" % pattern)
    print("gt=%s" % gt)
    print("iou=%.2f min_area=%d eval=%s ignore=%s" %
          (args.iou, min_area, sorted(cfg["eval_classes"]),
           sorted(cfg["ignore_classes"])))

    results = load_results(args.res)
    gt_data = load_gt(gt, cfg)
    print("results: %d frames, %d rows | gt: %d frames" %
          (len(results), sum(len(v) for v in results.values()), len(gt_data)))

    reader = VideoReader(pattern)
    total = reader.total
    print("video frames: %d" % total)
    if total <= 0:
        raise SystemExit("empty video")

    if args.dump_frame:  # scripted check: render one frame, save, exit
        idx = max(0, min(total - 1, args.dump_frame - 1))
        frame = reader.get(idx)
        frame, stats = render_frame(frame, idx + 1, gt_data.get(idx + 1, []),
                                    results.get(idx + 1, []), cfg,
                                    args.iou, min_area, not args.no_labels)
        render_hud(frame, idx + 1, total, seq, stats)
        cv2.imwrite(args.dump_out, frame)
        print("dump: frame=%d stats(gt,fn,match,fp,det)=%s -> %s" %
              (idx + 1, stats, args.dump_out))
        reader.release()
        return

    # GUI loop
    first = reader.get(0)
    fh, fw = first.shape[:2]
    scale = args.scale if args.scale > 0 else min(1.0, 1600.0 / fw, 950.0 / fh)
    dw, dh = int(fw * scale), int(fh * scale)
    win = "MOT viz  (%s)" % seq
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    print("display %dx%d (scale %.2f) | keys: right/left/home/end/space/esc" %
          (dw, dh, scale))

    def show(idx):
        frame = reader.get(idx)
        if frame is None:
            return False
        frame, stats = render_frame(frame, idx + 1, gt_data.get(idx + 1, []),
                                    results.get(idx + 1, []), cfg,
                                    args.iou, min_area, not args.no_labels)
        disp = frame if scale == 1.0 else cv2.resize(frame, (dw, dh))
        render_hud(disp, idx + 1, total, seq, stats)
        cv2.imshow(win, disp)
        return True

    idx, playing = 0, False
    if not show(idx):
        raise SystemExit("cannot read frame 1")
    while True:
        key = cv2.waitKeyEx(66 if playing else 1)
        if key == -1:
            if playing:
                idx = min(idx + 1, total - 1)
                if not show(idx):
                    break
            continue
        k = key & 0xFF
        if key == 27 or k == 27:                          # ESC
            break
        elif key in (39, 0x210000, 65363):                # RIGHT
            idx = min(idx + 1, total - 1)
        elif key in (37, 0x1b0000, 65361):                # LEFT
            idx = max(idx - 1, 0)
        elif key in (36, 65360):                          # HOME
            idx = 0
        elif key in (35, 65367):                          # END
            idx = total - 1
        elif key == 0x20 or k == 0x20:                    # SPACE: play/pause
            playing = not playing
        elif k in (ord("q"), ord("Q")):
            break
        else:
            continue
        if not show(idx):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
