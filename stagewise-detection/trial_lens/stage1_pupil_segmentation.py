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
#  NOTE ON "LEFT" / "RIGHT":  "left" means the pupil that is on the LEFT SIDE OF
#  THE PICTURE (it is the patient's right eye).  Swap the names in the CSV if
#  you prefer the anatomical convention.
#
#  HOW TO RUN  (type in a terminal)
#  --------------------------------
#     pip install opencv-python numpy               # one-time install
#
#     python pupil_segmentation.py  my_image.png                  # one image
#     python pupil_segmentation.py  my_folder/                    # a whole folder
#     python pupil_segmentation.py  "my_folder/*Sph_0.5*.png"     # a pattern
#     python pupil_segmentation.py  my_folder/  -o results/       # choose output folder
#     python pupil_segmentation.py  my_folder/  --debug           # also show rejected
#                                                                 # candidates (grey)
#     python pupil_segmentation.py  my_folder/  --min-diam 30 --max-diam 90
#                                   (any parameter in the PARAMETERS block below
#                                    can be overridden like this, without editing
#                                    the file;  see  python pupil_segmentation.py -h)
#
#  THE IDEA (read this once, it makes everything below easy)
#  ---------------------------------------------------------
#  These pictures are taken with infra-red light shining along the camera axis.
#  Light goes into the eye and bounces straight back out of the retina, so the
#  PUPIL APPEARS AS A BRIGHT, SMOOTH DISC (the "bright-pupil" effect), surrounded
#  by a darker iris.  Inside the disc there is a tiny, very bright dot: that is
#  the reflection from the front of the cornea (the "glint").
#
#  The problem: the metal trial-lens frame is also bright and shiny, it has
#  screws that sparkle, and it has glare streaks.  There can also be extra
#  glints on the rim of the pupil.  A plain "everything brighter than X" rule
#  would therefore pick up the frame and glints too.
#
#  So instead of using one brightness number, we do this:
#
#   STEP 0  "Erase" every small bright feature from the image (glints, sparkles,
#           thin reflection streaks).  The big smooth pupil survives, they don't.
#   STEP 1  Try MANY brightness thresholds (25, 29, 33, ... 235).
#           At each threshold, cut the image into bright "blobs".
#   STEP 2  Throw away every blob that does not LOOK LIKE A PUPIL:
#             - wrong size            (frame pieces are much bigger,
#                                      screws/glints are much smaller)
#             - not round enough      (frame pieces are elongated / irregular)
#             - has dents or holes    (frame pieces have slots and screw holes)
#             - not clearly brighter than its surroundings
#                                     (a pupil sits in a DARK ring; shiny metal
#                                      is surrounded by more shiny metal)
#             - too blotchy inside    (a pupil is smooth; metal is textured)
#             - touches the image border
#   STEP 3  Give each surviving blob a score ("how pupil-like is it?").
#           Merge duplicates (the same pupil appears at many thresholds).
#   STEP 4  Pick the best PAIR of blobs that sit side by side (one left, one
#           right, at about the same height).  This is why the script does not
#           need to be told where the pupils are: it works even when the eyes
#           move around inside the frame.
#   STEP 5  Find the exact edge.  Around each chosen pupil the image is
#           "unrolled" (360 directions x distance from the centre) and the
#           sharpest bright->dark transition is traced in every direction.
#           The edge strength is a brightness RATIO, so a pupil that is bright
#           on one side and dim on the other (uneven illumination) is still
#           outlined completely.  Glint pixels are ignored and bridged over
#           smoothly.  The search is repeated, re-centred, a few times.
#   STEP 6  Fill the outline, save the masks / overlay, write the CSV line.
#
# =============================================================================

import argparse
import csv
import glob
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import cv2          # OpenCV: image reading, thresholding, contours, drawing
import numpy as np  # NumPy : fast array maths


# =============================================================================
#  WHERE ARE YOUR IMAGES?   <-- type your folder paths here (between the quotes)
# =============================================================================
#  Option A (easiest): fill in the two lines below, save, and just press "Run"
#  in your editor (Spyder / VS Code / PyCharm / IDLE) or type:
#       python pupil_segmentation.py
#  Windows example :  INPUT_FOLDER = r"C:\Users\me\Desktop\calibration_images"
#  Linux/Mac example: INPUT_FOLDER = "/home/me/calibration_images"
#  (keep the  r  in front of Windows paths; it stops backslashes causing errors)
#  INPUT_FOLDER may also be a single image file or a pattern like "imgs/*.png".
#
#  Option B: leave them empty and give the folder on the command line instead:
#       python pupil_segmentation.py  "C:\path\to\images"  -o  "C:\path\to\results"
#  (anything typed on the command line overrides the two lines below)
INPUT_FOLDER = r" "                    # folder with your images
OUTPUT_FOLDER = r" "

# =============================================================================
#  PARAMETERS  -  the ONLY other section you normally need to touch
# =============================================================================
#  Each parameter has:  what it means / what happens if you raise or lower it /
#  the symptom that tells you it needs changing.
#  All sizes are in PIXELS of the image you feed in (the examples here are
#  1280 x 800 images where a pupil is about 60-65 px wide).
#  You can also override any of them from the command line, e.g.  --min-diam 30
# -----------------------------------------------------------------------------
PARAMS = dict(

    # ---- 0. Glint / reflection removal (done first, on the whole image) ------
    # GLINT_DIAM : diameter (px) of the "eraser".  Every bright feature SMALLER
    #   than this (the corneal glint, a second glint on the pupil rim, sparkling
    #   screws, thin eyelash reflections / glare streaks) is wiped out of the
    #   image BEFORE we look for the pupil, while the big smooth pupil disc is
    #   left untouched.  (Technically: a grey-scale "morphological opening".)
    #   Rule of thumb: bigger than your biggest glint, smaller than your SMALLEST
    #   pupil.  Here glints are <= 16 px and pupils >= 45 px, so 21 is safe.
    #   Raise (25-31) if a glint still bulges out of the outline;
    #   lower (13-17) if the pupil is tiny (e.g. < 35 px) or its outline looks
    #   too "rounded off" at the corners.
    GLINT_DIAM=21,

    # ---- 1. Pre-processing ---------------------------------------------------
    # PRE_BLUR_SIGMA : tiny blur applied before the eraser to suppress
    #   sensor noise.  Keep ~1.  (Raise to 1.5-2 for very noisy images.)
    PRE_BLUR_SIGMA=1.0,

    # BLUR_SIGMA : how much the image is blurred before looking for blobs.
    #   Blurring suppresses sensor noise and makes the tiny glint nearly vanish.
    #   Higher (3-4) -> smoother, more tolerant to noise, but outline may become
    #                   slightly rounder.
    #   Lower  (1)   -> follows the edge more tightly but is noisier.
    #   Change it if: the outline looks "wobbly/noisy" (raise it) or it cuts off
    #   real corners of the pupil (lower it).
    BLUR_SIGMA=2.0,

    # ---- 2. Brightness thresholds that are tried (0 = black, 255 = white) ----
    # The script tries thresholds from THRESH_MIN up to THRESH_MAX.  Each next
    # level is THRESH_RATIO times the previous one (e.g. 1.06 = 6 % brighter),
    # so dark images get fine steps and bright images get coarser ones.
    # You rarely need to change these.
    #   Pupils in VERY dark images (their brightness is below THRESH_MIN)
    #   -> lower THRESH_MIN (e.g. 5).
    #   Smaller THRESH_RATIO (1.03) = more careful but 2x slower;
    #   larger (1.10) = faster but may miss the ideal level.
    THRESH_MIN=8,
    THRESH_MAX=235,
    THRESH_RATIO=1.06,

    # ---- 3. Expected pupil size (diameter in pixels) -------------------------
    # Blobs outside this range are ignored.  This is the main weapon against the
    # frame: its bright pieces are far bigger than a pupil, and screws / glints
    # are far smaller.
    #   Pupil missed because it is bigger/smaller than expected -> widen range.
    #   Frame piece selected instead of the pupil               -> narrow range.
    #   If you change the camera distance / resolution, scale these.
    MIN_DIAM=25,
    MAX_DIAM=110,

    # ---- 4. Shape tests (pupil is round-ish but NOT a perfect circle) --------
    # MIN_CIRCULARITY : 1.0 = perfect circle, smaller = more irregular.
    #   0.80 accepts mildly oval / slightly squarish pupils.
    #   Lower it (0.70) if a real, oddly shaped pupil is rejected;
    #   raise it (0.88) if elongated frame bits are accepted.
    MIN_CIRCULARITY=0.80,

    # MIN_SOLIDITY : blob area / area of its "rubber-band" (convex hull).
    #   1.0 = no dents at all.  Pupils are convex; frame parts with slots are not.
    #   Lower it (0.85) only if the pupil is partly covered by an eyelid and gets
    #   rejected because of a notch.
    MIN_SOLIDITY=0.92,

    # MAX_ASPECT : longest side / shortest side of the blob's bounding box.
    #   1.4 means "at most 40 % longer than wide".  Raise (1.7) for strongly
    #   oval pupils (e.g. squinting child).
    MAX_ASPECT=1.4,

    # ---- 5. "Is it brighter than its surroundings?" test ---------------------
    # CONTRAST = 1 - (brightness of ring around blob) / (brightness inside blob)
    #   0 = same brightness inside and outside, 1 = ring is black.
    # MIN_CONTRAST : blobs with less contrast than this are rejected.
    #   Raise (0.5) if bright frame areas are being picked.
    #   Lower (0.25) if a dim pupil in a dark image is missed.
    MIN_CONTRAST=0.35,
    # RING_WIDTH : thickness (px) of the ring around the blob used above.
    RING_WIDTH=10,

    # MAX_INSIDE_VARIATION : a pupil is smooth; shiny textured metal is not.
    #   = (std-dev of brightness inside) / (mean brightness inside).
    #   Raise (0.35) if the real pupil has strong shading/vignetting
    #   (bright on one side, darker on the other) and gets rejected.
    MAX_INSIDE_VARIATION=0.25,

    # ---- 6. Choosing the pair of eyes ----------------------------------------
    # MIN_EYE_SEPARATION : the two pupils must be at least this far apart
    #   horizontally (px).  Stops the script from picking two blobs of one eye.
    # MAX_EYE_DY : the two pupils must be within this vertical distance (px).
    #   The trial frame can be tilted a little, so keep some slack.
    MIN_EYE_SEPARATION=250,
    MAX_EYE_DY=150,

    # ---- 7. Final-outline clean-up -------------------------------------------
    # CLEAN_RADIUS : radius (px) of the "opening" that deletes thin features
    #   (eyelash reflections, thin glare streaks touching the pupil).
    #   Anything thinner than 2*CLEAN_RADIUS+1 px is removed.
    #   Raise (4-5) if a thin tail/streak is still attached to the outline;
    #   lower (1-2) if the pupil is small and its edge is eroded.
    CLEAN_RADIUS=3,

    # ---- 8. Exact boundary search (Step 5) -----------------------------------
    # For each of 360 directions, starting from the pupil centre, the script
    # looks along that ray for the SHARPEST bright->dark drop (the pupil edge)
    # between RADIAL_MIN_FRAC and RADIAL_MAX_FRAC times the rough radius.
    # The 360 edge points are then linked into one smooth closed outline.
    # This works even when the pupil has a dim "glow" around it, because the
    # glow's soft fade is weaker than the pupil's real edge.
    #
    # RADIAL_MIN_FRAC / RADIAL_MAX_FRAC : search window as a fraction of the
    #   rough radius.  0.7-1.35 means "from 30 % inside to 35 % outside".
    #   Widen (0.6-1.5) if the outline is clipped at the window limit
    #   (strongly non-circular pupil); narrow it to be stricter against glare.
    RADIAL_MIN_FRAC=0.65,
    RADIAL_MAX_FRAC=1.5,

    # EDGE_WIDTH : px.  The edge strength is "brightness 1.5 px inside minus
    #   brightness 1.5 px outside".  Larger = more tolerant of soft/blurry
    #   edges (defocus), smaller = sharper but noisier.
    EDGE_WIDTH=1.5,

    # EDGE_FLOOR : stabiliser of the brightness ratio, as a fraction of the pupil
    #   brightness.  Prevents the extremely dark background (noise!) from
    #   producing huge ratios.  Raise (0.25) for very noisy / dark images,
    #   lower (0.05) for clean ones.
    EDGE_FLOOR=0.15,

    # STIFFNESS : how strongly the outline resists sudden jumps between
    #   neighbouring directions.  Higher (0.05-0.1) -> smoother, more "circular"
    #   outline, ignores small bumps;  lower (0.005) -> follows every wiggle.
    #   Raise it if the outline is jagged; lower it if a real corner/notch of
    #   the pupil is being cut.
    STIFFNESS=0.05,

    # PRIOR_WEIGHT : gentle pull towards the rough radius, so the outline
    #   cannot wander off to some far-away edge.  0 = off, 1 = default,
    #   larger = rounder/more conservative.
    PRIOR_WEIGHT=1.0,

    # GLINT_LEVEL / GLINT_DILATE : pixels brighter than GLINT_LEVEL x (normal
    #   pupil brightness) are treated as glints, grown by GLINT_DILATE px, and
    #   IGNORED when searching for the edge (the outline is bridged smoothly
    #   across them).  This stops a glint on the pupil rim from pulling the
    #   outline outwards.
    #   Outline still bulges near a glint -> lower GLINT_LEVEL (1.2) or raise
    #   GLINT_DILATE (6).  Parts of a brightly lit real pupil edge are
    #   ignored -> raise GLINT_LEVEL (1.5).
    GLINT_LEVEL=1.3,
    GLINT_DILATE=4,

    # EDGE_OFFSET_PX : shift the final outline outward (+) or inward (-) by this
    #   many pixels (sub-pixel values allowed).  Use it if your reference /
    #   hand-labelled masks are consistently slightly bigger or smaller.
    EDGE_OFFSET_PX=0.0,

    # SMOOTH_SIGMA : extra smoothing of the final radius profile, in DEGREES
    #   of angle.  0 = none.  Raise for a smoother curve.
    SMOOTH_SIGMA=3.0,

    # REFINE_PASSES : how many times the edge search is repeated, each time
    #   re-centred on the previous result.  The rough pupil from Step 3 can be
    #   too small / off-centre when one part of the pupil is much dimmer (uneven
    #   illumination); extra passes let the outline "grow" to the true edge.
    #   3 is plenty; use 1 for speed if your illumination is always even.
    REFINE_PASSES=3,

    # MAX_GROWTH : safety net on AREA.  If the final outline is more than this
    #   factor larger (or 1/factor smaller) than the rough pupil from Step 3,
    #   it is considered a failure and the rough circle is used instead
    #   (flagged "fallback" in the CSV -> look at those images by eye).
    #   Raise it if good outlines on strongly shaded pupils are flagged.
    MAX_GROWTH=2.4,

    # REJECT_BORDER : ignore blobs touching the image edge (partly visible).
    #   Set to False only if pupils can legitimately be cut off by the image edge.
    REJECT_BORDER=True,
)


# =============================================================================
#  SMALL HELPER FUNCTIONS
# =============================================================================

def load_gray(path):
    """Read an image file and return it as an 8-bit grey-scale array.

    Works for 8-bit or 16-bit files and for grey or colour (3/4 channel) files,
    so you do not have to worry about the format of the camera output.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)   # keep original bit depth
    if img is None:
        raise IOError(f"could not read image: {path}")
    if img.ndim == 3:                                    # colour -> grey
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    if img.dtype == np.uint16:                           # 16-bit -> 8-bit
        img = (img / 256).astype(np.uint8)
    return img


def ellipse_kernel(radius):
    """A round structuring element (a small disc) used for morphology."""
    d = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def fill_external_contours(binary):
    """Fill the inside of every outer outline.  This closes holes in a blob
    (for example the hole left if the glint was not bright enough to be
    included) so the pupil mask is a solid shape."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(binary)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out


# =============================================================================
#  STEP 1-3 : FIND PUPIL-LIKE BLOBS AT MANY THRESHOLDS
# =============================================================================

def find_candidates(blur, P):
    """Return a list of 'candidate' pupils (dictionaries).

    `blur` is the blurred grey image.  For every brightness threshold we
    split the image into connected bright blobs and keep only the blobs that
    pass all the pupil-likeness tests.
    """
    H, W = blur.shape
    min_area = np.pi * (P["MIN_DIAM"] / 2) ** 2          # area of smallest pupil
    max_area = np.pi * (P["MAX_DIAM"] / 2) ** 2          # area of largest pupil
    open_k = ellipse_kernel(P["CLEAN_RADIUS"])
    ring_k = ellipse_kernel(P["RING_WIDTH"])
    gap_k = ellipse_kernel(2)                            # small gap next to the edge
    candidates = []

    levels, t = [], float(P["THRESH_MIN"])
    while t <= P["THRESH_MAX"]:                          # geometric ladder of levels
        levels.append(int(round(t)))
        t = max(t + 1, t * P["THRESH_RATIO"])

    for t in sorted(set(levels)):
        # --- cut the image: 1 where brighter than t, 0 elsewhere ---
        binary = (blur > t).astype(np.uint8)
        # --- delete thin streaks / specks (opening = erode then dilate) ---
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
        # --- label every separate bright blob ---
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for i in range(1, n):                            # label 0 = background
            x, y, w, h, area = stats[i]

            # ---------- cheap tests first (size, bounding box) ----------
            if area < min_area * 0.8 or area > max_area * 1.2:
                continue
            if max(w, h) / max(1, min(w, h)) > P["MAX_ASPECT"]:
                continue
            if area / float(w * h) < 0.55:               # circle fills 78 % of its box
                continue
            if P["REJECT_BORDER"] and (x <= 0 or y <= 0 or x + w >= W or y + h >= H):
                continue

            # ---------- crop around the blob (with room for the ring) ----------
            pad = P["RING_WIDTH"] + 3
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            blob = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            blob = fill_external_contours(blob)          # fill glint hole etc.

            # ---------- shape tests ----------
            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea)
            a = cv2.contourArea(c)
            per = cv2.arcLength(c, True)
            if per == 0:
                continue
            circularity = 4 * np.pi * a / per ** 2       # 1.0 = perfect circle
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = a / hull_area if hull_area > 0 else 0
            if circularity < P["MIN_CIRCULARITY"] or solidity < P["MIN_SOLIDITY"]:
                continue

            # ---------- brightness tests (inside vs. ring around) ----------
            crop = blur[y0:y1, x0:x1]
            inside = crop[blob > 0]
            inside_med = float(np.median(inside))
            # ring = band of RING_WIDTH px just outside the blob
            ring = (cv2.dilate(blob, ring_k) > 0) & ~(cv2.dilate(blob, gap_k) > 0)
            if ring.sum() < 20:
                continue
            ring_med = float(np.median(crop[ring]))
            contrast = 1.0 - ring_med / max(inside_med, 1e-6)
            variation = float(inside.std() / max(inside.mean(), 1e-6))
            if contrast < P["MIN_CONTRAST"] or variation > P["MAX_INSIDE_VARIATION"]:
                continue

            # ---------- survived: remember it ----------
            M = cv2.moments(c)
            cx = M["m10"] / M["m00"] + x0
            cy = M["m01"] / M["m00"] + y0
            candidates.append(dict(
                thr=t, area=float(a), cx=cx, cy=cy,
                diam=2 * np.sqrt(a / np.pi),
                circ=float(circularity), solid=float(solidity),
                contrast=float(contrast),
                inside_med=inside_med, ring_med=ring_med,
                # SCORE: high contrast + round + no dents = very pupil-like
                score=float(contrast * circularity * solidity),
            ))
    return candidates


def merge_duplicates(cands):
    """The same pupil is found at many thresholds.  Keep only the best-scoring
    version of each (greedy 'non-maximum suppression')."""
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
    """Return (left, right).  Either may be None if it could not be found."""
    top = cands[:8]                                      # only the best few matter
    best, best_score = None, -1
    for a, b in combinations(top, 2):
        if abs(a["cx"] - b["cx"]) < P["MIN_EYE_SEPARATION"]:
            continue                                     # too close side-by-side
        if abs(a["cy"] - b["cy"]) > P["MAX_EYE_DY"]:
            continue                                     # not at the same height
        s = a["score"] + b["score"]
        if s > best_score:
            best, best_score = (a, b), s
    if best is not None:
        a, b = sorted(best, key=lambda d: d["cx"])       # smaller x = image left
        return a, b
    # fall back: only one eye found -> decide left/right by position in image
    if top:
        only = top[0]
        return (only, None) if only["cx"] < P["_img_w"] / 2 else (None, only)
    return None, None


# =============================================================================
#  STEP 5-6 : ACCURATE BOUNDARY OF ONE PUPIL
# =============================================================================

def _outline_once(clean, base, cx, cy, r0, inside_med, P):
    """One pass of the polar edge search around the centre (cx, cy).

    clean : glint-free grey image  (Step 0 result)
    base  : lightly blurred ORIGINAL image (still contains the glints; we use
            it only to know WHERE the glints are)
    cx, cy, r0 : rough centre and radius of the pupil
    inside_med : typical pupil brightness (used to recognise glints)

    Method ("polar dynamic programming"):
      1. unroll the image around the pupil centre into a table:
            rows = 360 directions, columns = distance from the centre
      2. in that table the pupil edge is a roughly horizontal line;
         edge strength = bright inside minus dark outside
      3. find the best continuous line through the table (dynamic programming)
         - strong edges are rewarded, glints are ignored,
           jumps between neighbouring directions are penalised
      4. turn that line back into an outline and fill it.
    Returns mask: full-size uint8, 1 = pupil.
    """
    H, W = clean.shape
    STEP = 0.5                                           # radial sampling step (px)
    d = P["EDGE_WIDTH"]
    sh = int(round(d / STEP))                            # gradient offset in samples
    N = 360
    theta = np.deg2rad(np.arange(N))

    # ---- 1. unroll into a polar table ----
    r_lo, r_hi = P["RADIAL_MIN_FRAC"] * r0, P["RADIAL_MAX_FRAC"] * r0
    R = np.arange(r_lo - d, r_hi + d + STEP, STEP)       # padded by the gradient width
    mx = (cx + np.outer(np.cos(theta), R)).astype(np.float32)
    my = (cy + np.outer(np.sin(theta), R)).astype(np.float32)
    polar = cv2.remap(clean.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ---- 2. edge strength = RATIO of brightness (d px inside) / (d px outside),
    #         measured in log scale.  Using a RATIO instead of a difference is
    #         the key trick for uneven illumination: the pupil is often bright
    #         on one side and dim on the other, but even the dim side is many
    #         times brighter than the dark iris outside it, whereas the step
    #         between the bright and dim parts INSIDE the pupil is only a
    #         small ratio.  So the true pupil boundary wins everywhere.
    c = P["EDGE_FLOOR"] * max(inside_med, 1.0)           # stops dark noise exploding the ratio
    E = np.log((polar[:, :-2 * sh] + c) / (polar[:, 2 * sh:] + c))
    E = np.clip(E, 0, None)                              # only bright->dark counts
    Rk = R[sh:-sh]                                       # radius each column refers to

    # ---- glints: where the ORIGINAL is far brighter than the pupil, edge
    #      information is unreliable -> set strength to 0 so the outline is
    #      simply bridged smoothly across them ----
    glint = (base > P["GLINT_LEVEL"] * inside_med).astype(np.uint8)
    if P["GLINT_DILATE"] > 0:
        glint = cv2.dilate(glint, ellipse_kernel(P["GLINT_DILATE"]))
    gp = cv2.remap(glint, mx, my, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    bad = gp[:, :-2 * sh] | gp[:, sh:-sh] | gp[:, 2 * sh:]
    E[bad] = 0

    # ---- 3. dynamic programming: best smooth line through the table ----
    # node cost = -edge strength + pull towards the rough radius
    cost = -E + P["PRIOR_WEIGHT"] * ((Rk[None, :] - r0) / r0) ** 2
    K = cost.shape[1]
    cost3 = np.vstack([cost, cost, cost])                # repeat 3x so the outline closes
    S = 2                                                # max jump (samples) per degree
    jump_pen = {s: P["STIFFNESS"] * s * s for s in range(-S, S + 1)}
    acc = cost3.copy()
    back = np.zeros(cost3.shape, np.int8)
    for a in range(1, cost3.shape[0]):
        prev = acc[a - 1]
        best = np.full(K, np.inf)
        arg = np.zeros(K, np.int8)
        for s in range(-S, S + 1):                       # try every allowed jump
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
    for a in range(cost3.shape[0] - 1, -1, -1):          # walk back
        path[a] = k
        k = min(max(k + int(back[a, k]), 0), K - 1)
    radii = Rk[path[N:2 * N]]                            # keep the middle copy

    # ---- 4. smooth, optional offset, rebuild outline, fill ----
    if P["SMOOTH_SIGMA"] > 0:
        w = int(4 * P["SMOOTH_SIGMA"]) + 1
        g = cv2.getGaussianKernel(2 * w + 1, P["SMOOTH_SIGMA"]).ravel()
        ext = np.concatenate([radii[-w:], radii, radii[:w]])
        radii = np.convolve(ext, g, mode="valid")
    radii = radii + P["EDGE_OFFSET_PX"]
    pts = np.stack([cx + radii * np.cos(theta), cy + radii * np.sin(theta)], 1)
    mask = np.zeros((H, W), np.uint8)
    # shift=4 -> sub-pixel accurate polygon filling (coordinates x16)
    cv2.fillPoly(mask, [np.round(pts * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)

    return mask


def refine_mask(clean, base, cand, P):
    """Step 5 driver: run the polar edge search REFINE_PASSES times.

    Pass 1 uses the rough centre / radius from Step 3 (which may be biased
    towards the brightest part of the pupil).  Each further pass re-centres on
    the outline found in the previous pass and searches again, which fixes
    pupils with strongly uneven brightness.  Returns (mask, "ok" | "fallback").
    """
    cx, cy, r0 = cand["cx"], cand["cy"], cand["diam"] / 2.0
    mask = None
    for _ in range(P["REFINE_PASSES"]):
        mask = _outline_once(clean, base, cx, cy, r0, cand["inside_med"], P)
        area = float(mask.sum())
        if area < 10:
            break
        M = cv2.moments(mask, binaryImage=True)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]  # new centre
        r0 = np.sqrt(area / np.pi)                         # new radius

    # safety net: absurd size compared with the rough Step-3 pupil -> fall back
    ratio = mask.sum() / max(cand["area"], 1.0)
    if mask.sum() < 10 or ratio > P["MAX_GROWTH"] or ratio < 1.0 / P["MAX_GROWTH"]:
        mask = np.zeros_like(mask)
        cv2.circle(mask, (int(round(cand["cx"])), int(round(cand["cy"]))),
                   int(round(cand["diam"] / 2)), 1, -1)
        return mask, "fallback"
    return mask, "ok"


def describe(mask):
    """Numbers describing a mask: centroid, area, diameter, fitted ellipse."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    info = dict(cx=M["m10"] / M["m00"], cy=M["m01"] / M["m00"], area=area,
                eq_diam=2 * np.sqrt(area / np.pi), major=np.nan, minor=np.nan, angle=np.nan)
    if len(c) >= 5:                                      # ellipse fit needs >= 5 points
        (_, _), (d1, d2), ang = cv2.fitEllipse(c)
        info.update(major=max(d1, d2), minor=min(d1, d2), angle=ang)
    return info, c


# =============================================================================
#  PROCESS ONE IMAGE
# =============================================================================

COLORS = {"left": (0, 255, 0), "right": (0, 200, 255)}   # BGR: green / orange


def process_image(path, out_dir, P, debug=False):
    gray = load_gray(path)
    P = dict(P, _img_w=gray.shape[1])

    # --- Step 0: erase glints / thin streaks (see GLINT_DIAM) ---
    base = cv2.GaussianBlur(gray, (0, 0), P["PRE_BLUR_SIGMA"])
    clean = cv2.morphologyEx(base, cv2.MORPH_OPEN,
                             ellipse_kernel(P["GLINT_DIAM"] // 2))
    blur = cv2.GaussianBlur(clean, (0, 0), P["BLUR_SIGMA"])
    cands = merge_duplicates(find_candidates(blur, P))   # Steps 1-3
    left_c, right_c = choose_pair(cands, P)              # Step 4

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if debug:                                            # grey circles = all candidates
        for c in cands:
            cv2.circle(overlay, (int(c["cx"]), int(c["cy"])), int(c["diam"] / 2),
                       (160, 160, 160), 1, cv2.LINE_AA)

    stem = Path(path).stem
    row = {"image": Path(path).name}
    for side, cand in (("left", left_c), ("right", right_c)):
        if cand is None:
            row.update({f"{side}_found": 0})
            continue
        mask, flag = refine_mask(clean, base, cand, P)   # Step 5
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
            f"{side}_outline": flag,                   # "fallback" = check manually
            f"{side}_score": round(cand["score"], 3),    # low score = double-check
        })
    cv2.imwrite(str(out_dir / "annotated" / f"{stem}_annotated.png"), overlay)
    ok = left_c and right_c and "fallback" not in (row.get("left_outline"), row.get("right_outline"))
    row["status"] = "OK" if ok else "CHECK: eye(s) missing or fallback outline"
    return row


# =============================================================================
#  COMMAND-LINE INTERFACE  (batch processing + CSV)
# =============================================================================

def collect_inputs(spec):
    """Turn 'file', 'folder' or 'glob pattern' into a sorted list of files."""
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
    """process_image, but a broken file only produces an ERROR row instead of
    stopping the whole batch."""
    try:
        return process_image(f, out_dir, P, debug=debug)
    except Exception as e:
        return {"image": Path(f).name, "status": f"ERROR: {e}"}


def main():
    ap = argparse.ArgumentParser(description="Automatic pupil segmentation (bright-pupil IR images)")
    ap.add_argument("input", nargs="?", default=INPUT_FOLDER,
                    help="image file, folder, or glob pattern (default: INPUT_FOLDER set at the top of the script)")
    ap.add_argument("-o", "--out", default=OUTPUT_FOLDER, help="output folder")
    ap.add_argument("--debug", action="store_true",
                    help="draw every candidate that was considered as a thin grey circle")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="number of images processed in parallel (use e.g. 4-8 for thousands of files)")
    # every parameter can be overridden:  --min-diam 30   (underscores -> dashes)
    for k, v in PARAMS.items():
        flag = "--" + k.lower().replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    args = ap.parse_args()
    P = {k: getattr(args, k.lower()) for k in PARAMS}

    if not args.input:
        sys.exit("No image folder given.  Either set INPUT_FOLDER at the top of "
                 "the script or run:  python pupil_segmentation.py  <folder>")
    files = collect_inputs(args.input)
    if not files:
        sys.exit("No images found for: " + args.input)
    out_dir = Path(args.out)
    for sub in ("masks", "annotated", "csv"):            # keep the different outputs apart
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    if args.jobs > 1:                                    # parallel: one image per CPU core
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_safe_process, f, out_dir, P, args.debug) for f in files]
            for n, fu in enumerate(futures, 1):
                rows.append(fu.result())
                print(f"[{n}/{len(files)}] {rows[-1]['image']}: {rows[-1]['status']}")
    else:                                                # simple loop
        for n, f in enumerate(files, 1):
            rows.append(_safe_process(f, out_dir, P, args.debug))
            print(f"[{n}/{len(files)}] {f.name}: {rows[-1]['status']}")

    # write the CSV (union of all column names, in a sensible order)
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