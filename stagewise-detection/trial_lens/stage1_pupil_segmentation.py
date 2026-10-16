#!/usr/bin/env python3
# =============================================================================
#  PUPIL SEGMENTATION FOR TRIAL-LENS CALIBRATION IMAGES  (beginner-friendly)
# =============================================================================
#
#  WHAT THIS SCRIPT DOES
#  ---------------------
#  For every image you give it, it automatically finds BOTH pupils and writes:
#  inside the output folder, in three separate sub-folders:
#     annotated/<name>_annotated.png   the image with the pupil outlines drawn on
#     masks/<name>_mask_left.png       black/white mask (white = pupil), LEFT  pupil
#     masks/<name>_mask_right.png      black/white mask (white = pupil), RIGHT pupil
#     csv/results.csv                  centre, area, diameter, ellipse fit ... of
#                                      every pupil in every image (one row per image)
#
# =============================================================================

import argparse
import csv
import glob
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import cv2          # OpenCV: image reading, thresholding, contours, drawing
import numpy as np  # NumPy : fast array maths


# =============================================================================
#  PARAMETERS  -  the ONLY other section you normally need to touch
# =============================================================================
PARAMS = dict(

    # ---- 0. Glint / reflection removal (done first, on the whole image) ------
    GLINT_DIAM=21,

    # ---- 1. Pre-processing ---------------------------------------------------
    PRE_BLUR_SIGMA=1.0,
    BLUR_SIGMA=2.0,

    # ---- 2. Brightness thresholds that are tried (0 = black, 255 = white) ----
    THRESH_MIN=8,
    THRESH_MAX=235,
    THRESH_RATIO=1.06,

    # ---- 3. Expected pupil size (diameter in pixels) -------------------------
    MIN_DIAM=25,
    MAX_DIAM=110,

    # ---- 4. Shape tests (pupil is round-ish but NOT a perfect circle) --------
    MIN_CIRCULARITY=0.80,
    MIN_SOLIDITY=0.92,
    MAX_ASPECT=1.4,

    # ---- 5. "Is it brighter than its surroundings?" test ---------------------
    MIN_CONTRAST=0.35,
    RING_WIDTH=10,
    MAX_INSIDE_VARIATION=0.25,

    # ---- 6. Choosing the pair of eyes ----------------------------------------
    MIN_EYE_SEPARATION=250,
    MAX_EYE_DY=150,

    # ---- 7. Final-outline clean-up -------------------------------------------
    CLEAN_RADIUS=3,

    # ---- 8. Exact boundary search (Step 5) -----------------------------------
    RADIAL_MIN_FRAC=0.65,
    RADIAL_MAX_FRAC=1.5,
    EDGE_WIDTH=1.5,
    EDGE_FLOOR=0.15,
    STIFFNESS=0.05,
    PRIOR_WEIGHT=1.0,
    GLINT_LEVEL=1.3,
    GLINT_DILATE=4,
    EDGE_OFFSET_PX=0.0,
    SMOOTH_SIGMA=3.0,
    REFINE_PASSES=3,
    MAX_GROWTH=2.4,
    REJECT_BORDER=True,
)


# =============================================================================
#  SMALL HELPER FUNCTIONS
# =============================================================================

def load_gray(path):
    """Read an image file and return it as an 8-bit grey-scale array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"could not read image: {path}")
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    return img


def ellipse_kernel(radius):
    """A round structuring element (a small disc) used for morphology."""
    d = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def fill_external_contours(binary):
    """Fill the inside of every outer outline."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(binary)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out


# =============================================================================
#  STEP 1-3 : FIND PUPIL-LIKE BLOBS AT MANY THRESHOLDS
# =============================================================================

def find_candidates(blur, P):
    H, W = blur.shape
    min_area = np.pi * (P["MIN_DIAM"] / 2) ** 2
    max_area = np.pi * (P["MAX_DIAM"] / 2) ** 2
    open_k = ellipse_kernel(P["CLEAN_RADIUS"])
    ring_k = ellipse_kernel(P["RING_WIDTH"])
    gap_k = ellipse_kernel(2)
    candidates = []

    levels, t = [], float(P["THRESH_MIN"])
    while t <= P["THRESH_MAX"]:
        levels.append(int(round(t)))
        t = max(t + 1, t * P["THRESH_RATIO"])

    for t in sorted(set(levels)):
        binary = (blur > t).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for i in range(1, n):
            x, y, w, h, area = stats[i]

            if area < min_area * 0.8 or area > max_area * 1.2:
                continue
            if max(w, h) / max(1, min(w, h)) > P["MAX_ASPECT"]:
                continue
            if area / float(w * h) < 0.55:
                continue
            if P["REJECT_BORDER"] and (x <= 0 or y <= 0 or x + w >= W or y + h >= H):
                continue

            pad = P["RING_WIDTH"] + 3
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            blob = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            blob = fill_external_contours(blob)

            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea)
            a = cv2.contourArea(c)
            per = cv2.arcLength(c, True)
            if per == 0:
                continue
            circularity = 4 * np.pi * a / per ** 2
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = a / hull_area if hull_area > 0 else 0
            if circularity < P["MIN_CIRCULARITY"] or solidity < P["MIN_SOLIDITY"]:
                continue

            crop = blur[y0:y1, x0:x1]
            inside = crop[blob > 0]
            inside_med = float(np.median(inside))
            ring = (cv2.dilate(blob, ring_k) > 0) & ~(cv2.dilate(blob, gap_k) > 0)
            if ring.sum() < 20:
                continue
            ring_med = float(np.median(crop[ring]))
            contrast = 1.0 - ring_med / max(inside_med, 1e-6)
            variation = float(inside.std() / max(inside.mean(), 1e-6))
            if contrast < P["MIN_CONTRAST"] or variation > P["MAX_INSIDE_VARIATION"]:
                continue

            M = cv2.moments(c)
            cx = M["m10"] / M["m00"] + x0
            cy = M["m01"] / M["m00"] + y0
            candidates.append(dict(
                thr=t, area=float(a), cx=cx, cy=cy,
                diam=2 * np.sqrt(a / np.pi),
                circ=float(circularity), solid=float(solidity),
                contrast=float(contrast),
                inside_med=inside_med, ring_med=ring_med,
                score=float(contrast * circularity * solidity),
            ))
    return candidates


def merge_duplicates(cands):
    cands = sorted(cands, key=lambda d: d["score"], reverse=True)
    kept = []
    for c in cands:
        too_close = any(np.hypot(c["cx"] - k["cx"], c["cy"] - k["cy"])
                        < 0.6 * max(c["diam"], k["diam"]) for k in kept)
        if not too_close:
            kept.append(c)
    return kept


# =============================================================================
#  STEP 4 : PICK THE BEST LEFT/RIGHT PAIR
# =============================================================================

def choose_pair(cands, P):
    top = cands[:8]
    best, best_score = None, -1
    for a, b in combinations(top, 2):
        if abs(a["cx"] - b["cx"]) < P["MIN_EYE_SEPARATION"]:
            continue
        if abs(a["cy"] - b["cy"]) > P["MAX_EYE_DY"]:
            continue
        s = a["score"] + b["score"]
        if s > best_score:
            best, best_score = (a, b), s
    if best is not None:
        a, b = sorted(best, key=lambda d: d["cx"])
        return a, b
    if top:
        only = top[0]
        return (only, None) if only["cx"] < P["_img_w"] / 2 else (None, only)
    return None, None


# =============================================================================
#  STEP 5-6 : ACCURATE BOUNDARY OF ONE PUPIL
# =============================================================================

def _outline_once(clean, base, cx, cy, r0, inside_med, P):
    H, W = clean.shape
    STEP = 0.5
    d = P["EDGE_WIDTH"]
    sh = int(round(d / STEP))
    N = 360
    theta = np.deg2rad(np.arange(N))

    r_lo, r_hi = P["RADIAL_MIN_FRAC"] * r0, P["RADIAL_MAX_FRAC"] * r0
    R = np.arange(r_lo - d, r_hi + d + STEP, STEP)
    mx = (cx + np.outer(np.cos(theta), R)).astype(np.float32)
    my = (cy + np.outer(np.sin(theta), R)).astype(np.float32)
    polar = cv2.remap(clean.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    c = P["EDGE_FLOOR"] * max(inside_med, 1.0)
    E = np.log((polar[:, :-2 * sh] + c) / (polar[:, 2 * sh:] + c))
    E = np.clip(E, 0, None)
    Rk = R[sh:-sh]

    glint = (base > P["GLINT_LEVEL"] * inside_med).astype(np.uint8)
    if P["GLINT_DILATE"] > 0:
        glint = cv2.dilate(glint, ellipse_kernel(P["GLINT_DILATE"]))
    gp = cv2.remap(glint, mx, my, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    bad = gp[:, :-2 * sh] | gp[:, sh:-sh] | gp[:, 2 * sh:]
    E[bad] = 0

    cost = -E + P["PRIOR_WEIGHT"] * ((Rk[None, :] - r0) / r0) ** 2
    K = cost.shape[1]
    cost3 = np.vstack([cost, cost, cost])
    S = 2
    jump_pen = {s: P["STIFFNESS"] * s * s for s in range(-S, S + 1)}
    acc = cost3.copy()
    back = np.zeros(cost3.shape, np.int8)
    for a in range(1, cost3.shape[0]):
        prev = acc[a - 1]
        best = np.full(K, np.inf)
        arg = np.zeros(K, np.int8)
        for s in range(-S, S + 1):
            shifted = np.full(K, np.inf)
            if s >= 0:
                shifted[:K - s] = prev[s:]
            else:
                shifted[-s:] = prev[:K + s]
            cand_cost = shifted + jump_pen[s]
            better = cand_cost < best
            best[better] = cand_cost[better]
            arg[better] = s
        acc[a] = cost3[a] + best
        back[a] = arg
    k = int(np.argmin(acc[-1]))
    path = np.zeros(cost3.shape[0], int)
    for a in range(cost3.shape[0] - 1, -1, -1):
        path[a] = k
        k = min(max(k + int(back[a, k]), 0), K - 1)
    radii = Rk[path[N:2 * N]]

    if P["SMOOTH_SIGMA"] > 0:
        w = int(4 * P["SMOOTH_SIGMA"]) + 1
        g = cv2.getGaussianKernel(2 * w + 1, P["SMOOTH_SIGMA"]).ravel()
        ext = np.concatenate([radii[-w:], radii, radii[:w]])
        radii = np.convolve(ext, g, mode="valid")
    radii = radii + P["EDGE_OFFSET_PX"]
    pts = np.stack([cx + radii * np.cos(theta), cy + radii * np.sin(theta)], 1)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(pts * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)

    return mask


def refine_mask(clean, base, cand, P):
    cx, cy, r0 = cand["cx"], cand["cy"], cand["diam"] / 2.0
    mask = None
    for _ in range(P["REFINE_PASSES"]):
        mask = _outline_once(clean, base, cx, cy, r0, cand["inside_med"], P)
        area = float(mask.sum())
        if area < 10:
            break
        M = cv2.moments(mask, binaryImage=True)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        r0 = np.sqrt(area / np.pi)

    ratio = mask.sum() / max(cand["area"], 1.0)
    if mask.sum() < 10 or ratio > P["MAX_GROWTH"] or ratio < 1.0 / P["MAX_GROWTH"]:
        mask = np.zeros_like(mask)
        cv2.circle(mask, (int(round(cand["cx"])), int(round(cand["cy"]))),
                   int(round(cand["diam"] / 2)), 1, -1)
        return mask, "fallback"
    return mask, "ok"


def describe(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    info = dict(cx=M["m10"] / M["m00"], cy=M["m01"] / M["m00"], area=area,
                eq_diam=2 * np.sqrt(area / np.pi), major=np.nan, minor=np.nan, angle=np.nan)
    if len(c) >= 5:
        (_, _), (d1, d2), ang = cv2.fitEllipse(c)
        info.update(major=max(d1, d2), minor=min(d1, d2), angle=ang)
    return info, c


# =============================================================================
#  PROCESS ONE IMAGE
# =============================================================================

COLORS = {"left": (0, 255, 0), "right": (0, 200, 255)}


def process_image(path, out_dir, P, debug=False):
    gray = load_gray(path)
    P = dict(P, _img_w=gray.shape[1])

    base = cv2.GaussianBlur(gray, (0, 0), P["PRE_BLUR_SIGMA"])
    clean = cv2.morphologyEx(base, cv2.MORPH_OPEN,
                             ellipse_kernel(P["GLINT_DIAM"] // 2))
    blur = cv2.GaussianBlur(clean, (0, 0), P["BLUR_SIGMA"])
    cands = merge_duplicates(find_candidates(blur, P))
    left_c, right_c = choose_pair(cands, P)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if debug:
        for c in cands:
            cv2.circle(overlay, (int(c["cx"]), int(c["cy"])), int(c["diam"] / 2),
                       (160, 160, 160), 1, cv2.LINE_AA)

    stem = Path(path).stem
    row = {"image": Path(path).name}
    for side, cand in (("left", left_c), ("right", right_c)):
        if cand is None:
            row.update({f"{side}_found": 0})
            continue
        mask, flag = refine_mask(clean, base, cand, P)
        info, contour = describe(mask)
        cv2.imwrite(str(out_dir / "masks" / f"{stem}_mask_{side}.png"), mask * 255)
        cv2.polylines(overlay, [contour], True, COLORS[side], 2, cv2.LINE_AA)
        cv2.drawMarker(overlay, (int(info["cx"]), int(info["cy"])), COLORS[side],
                       cv2.MARKER_CROSS, 10, 1)
        row.update({
            f"{side}_found": 1,
            f"{side}_cx": round(info["cx"], 2), f"{side}_cy": round(info["cy"], 2),
            f"{side}_area_px": round(info["area"], 1),
            f"{side}_eq_diam_px": round(info["eq_diam"], 2),
            f"{side}_ellipse_major_px": round(info["major"], 2),
            f"{side}_ellipse_minor_px": round(info["minor"], 2),
            f"{side}_ellipse_angle_deg": round(info["angle"], 1),
            f"{side}_outline": flag,
            f"{side}_score": round(cand["score"], 3),
        })
    cv2.imwrite(str(out_dir / "annotated" / f"{stem}_annotated.png"), overlay)
    ok = left_c and right_c and "fallback" not in (row.get("left_outline"), row.get("right_outline"))
    row["status"] = "OK" if ok else "CHECK: eye(s) missing or fallback outline"
    return row


# =============================================================================
#  COMMAND-LINE INTERFACE  (batch processing + CSV)
# =============================================================================

def collect_inputs(spec):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    p = Path(spec)
    if p.is_dir():
        files = [f for f in p.iterdir() if f.suffix.lower() in exts
                 and not any(s in f.stem for s in ("_annotated", "_overlay", "_mask_"))]
    elif p.is_file():
        files = [p]
    else:
        files = [Path(f) for f in glob.glob(spec)]
    return sorted(files)


def _safe_process(f, out_dir, P, debug):
    try:
        return process_image(f, out_dir, P, debug=debug)
    except Exception as e:
        return {"image": Path(f).name, "status": f"ERROR: {e}"}


def main():
    ap = argparse.ArgumentParser(description="Automatic pupil segmentation (bright-pupil IR images)")
    ap.add_argument("input", nargs="?", default="",
                    help="image file, folder, or glob pattern")
    ap.add_argument("-o", "--out", default="", help="output folder")
    ap.add_argument("--debug", action="store_true",
                    help="draw every candidate that was considered as a thin grey circle")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="number of images processed in parallel")
    for k, v in PARAMS.items():
        flag = "--" + k.lower().replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    args = ap.parse_args()
    P = {k: getattr(args, k.lower()) for k in PARAMS}

    input_paths_list = []
    base_folder = ""

    if args.input:
        input_paths_list = [args.input]
        out_dir = Path(args.out) if args.out else (Path(args.input) if Path(args.input).is_dir() else Path(args.input).parent)
    else:
        print()
        print("=" * 80)
        print("STAGE 1 - PUPIL SEGMENTATION (INTERACTIVE INPUT)")
        print("=" * 80)
        print()

        # Step 1: Enter base folder path (optional prefix)
        base_folder = input("Enter base folder path (press Enter to skip): ").strip().strip('"')

        # Step 2: Enter number of paths
        while True:
            try:
                num_paths_str = input("Enter the number of input paths: ").strip()
                num_paths = int(num_paths_str)
                if num_paths > 0:
                    break
                print("Please enter a number greater than 0.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

        # Step 3: Enter each subfolder or image path
        for i in range(1, num_paths + 1):
            path_input = input(f"Enter input path {i} (subfolder or full path): ").strip().strip('"')
            
            if base_folder and not os.path.isabs(path_input):
                full_path = os.path.join(base_folder, path_input)
            else:
                full_path = path_input
                
            input_paths_list.append(full_path)

        # Output folder prompt
        default_out = base_folder if base_folder else (input_paths_list[0] if len(input_paths_list) > 0 else "")
        out_input = input(f"Enter output folder path [default: {default_out or 'current directory'}]: ").strip().strip('"')
        if out_input:
            if base_folder and not os.path.isabs(out_input):
                out_dir = Path(os.path.join(base_folder, out_input))
            else:
                out_dir = Path(out_input)
        else:
            out_dir = Path(default_out) if default_out else Path(".")

    input_folders = [
        os.path.abspath(os.path.expanduser(path))
        for path in input_paths_list
    ]

    files = []
    for folder in input_folders:
        files.extend(collect_inputs(folder))
    files = sorted(list(set(files)))

    if not files:
        sys.exit("No images found for the specified input paths.")

    out_dir = Path(out_dir)
    for sub in ("masks", "annotated", "csv"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_safe_process, f, out_dir, P, args.debug) for f in files]
            for n, fu in enumerate(futures, 1):
                rows.append(fu.result())
                print(f"[{n}/{len(files)}] {rows[-1]['image']}: {rows[-1]['status']}")
    else:
        for n, f in enumerate(files, 1):
            rows.append(_safe_process(f, out_dir, P, args.debug))
            print(f"[{n}/{len(files)}] {f.name}: {rows[-1]['status']}")

    cols = ["image", "status"]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out_dir / "csv" / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    bad = sum(r["status"] != "OK" for r in rows)
    print(f"Done in {time.time() - t0:.1f}s.  {len(rows) - bad} OK, {bad} need a look "
          f"(see 'status' column in {out_dir / 'csv' / 'results.csv'}).")


if __name__ == "__main__":
    main()