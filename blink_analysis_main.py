import os
import sys
import csv
import glob
import time
import cv2
import numpy as np
import math


# ============================================================
# ROOT FOLDER
# ============================================================

ROOT_HR_FOLDER = (
    
)



# ============================================================
# MULTIPLE INPUT PATHS
# ============================================================
# Add as many input folder paths as you need.
# Keep at least 2 paths in this list.
# Every listed folder is processed in one run and appears
# separately in the LEFT-EYE and RIGHT-EYE slope-vs-angle graphs.

INPUT_PATHS = [
]
# ============================================================
# OUTPUT FOLDERS
# ============================================================

RAW_CONTOUR_FOLDER = "raw_contour_results1"
PUPIL_CIRCLE_FOLDER = "pupil_circle_results1"
OLD_OUTPUT_FOLDER = "annotated_results"

MASK_FOLDER_NAME = "pupil_masks"

CSV_NAME = "pupil_annotations.csv"


# ============================================================
# INTENSITY PROFILE
# ============================================================

INTENSITY_PROFILE_FOLDER = "angle_intensity_profiles"
INTENSITY_PROFILE_CSV = "angle_intensity_values.csv"

INTENSITY_PROFILE_SAMPLES = 101
SAVE_INTENSITY_PROFILE_GRAPHS = True


# ============================================================
# SLOPE vs ANGLE GRAPH
# ============================================================

# Overall graph:
#     X-axis = common/profile angle (degrees)
#     Y-axis = fitted intensity-profile slope
#
# One point is stored for each image + eye.
SLOPE_ANGLE_GRAPH_FOLDER = "slope_vs_angle"
SLOPE_ANGLE_GRAPH_FILE = "slope_vs_angle.png"
SLOPE_ANGLE_CSV_FILE = "slope_vs_angle_values.csv"
SAVE_SLOPE_ANGLE_GRAPH = True

# ============================================================
# MULTI-INPUT SLOPE vs ANGLE GRAPH
# ============================================================

# All configured input folders are processed in one run.
# Final comparison contains: left-eye graph and right-eye graph.
# Each input is shown with a marker and label.
MULTI_INPUT_OUTPUT_FOLDER = "slope_vs_angle_3_inputs"
MULTI_LEFT_GRAPH_FILE = "left_eye_slope_vs_angle.png"
MULTI_RIGHT_GRAPH_FILE = "right_eye_slope_vs_angle.png"
MULTI_SLOPE_ANGLE_CSV_FILE = "slope_vs_angle_3_inputs.csv"


# ============================================================
# DISPLAY
# ============================================================

SHOW_OUTLIER_POINTS = False
DISPLAY_VISUALS = False


# ============================================================
# DRAWING COLORS
# ============================================================

CONTOUR_COLOR = (0, 255, 255)
OUTLIER_COLOR = (255, 0, 255)

CIRCLE_COLOR = (0, 0, 255)
CIRCLE_REVIEW_COLOR = (0, 165, 255)

ORIENTATION_ARROW_COLOR = (255, 255, 0)


# ============================================================
# GLINT PARAMETERS
# ============================================================

MIN_BRIGHT_VALUE = 35
MIN_LOCAL_CONTRAST = 10

LOCAL_MAX_KERNEL = 11
BACKGROUND_KERNEL = 61

PUPIL_RING_INNER = 14
PUPIL_RING_OUTER = 30
PUPIL_DARK_WEIGHT = 4.0

MIN_GLINT_CONTRAST = 12


# ============================================================
# EYE PAIR PARAMETERS
# ============================================================

MIN_EYE_SEPARATION_FRAC = 0.28
MAX_EYE_SEPARATION_FRAC = 0.72
MAX_EYE_Y_DIFFERENCE_FRAC = 0.22

MAX_CANDIDATES_PER_SIDE = 35


# ============================================================
# PUPIL PARAMETERS
# ============================================================

PUPIL_R_MIN = 6
PUPIL_R_MAX = 60

PUPIL_SEARCH = 24

PUPIL_BLUR_SIGMA = 2.0

PUPIL_REVIEW_R_BELOW = 10

PUPIL_MIN_BRIGHT_RATIO = 1.3

GLINT_ERASE_RADIUS = 9


# ============================================================
# BLOB PARAMETERS
# ============================================================

BLOB_THRESHOLD = 8.0
MAX_BLOB_CANDIDATES = 10

BORDER_MARGIN = 8

MIN_PUPIL_RATIO = 1.4
MIN_PUPIL_QUALITY = 0.5

MAX_EYE_RADIUS_RATIO = 2.0

USE_LEGACY_FALLBACK = True


# ============================================================
# MASK / CIRCLE
# ============================================================

SAVE_MASKS = True
DRAW_PUPIL_CIRCLE = True
FILL_PUPIL_ON_OVERLAY = False


# ============================================================
# ORIENTATION
# ============================================================

DRAW_ORIENTATION_ARROW = True

ORIENTATION_INNER_FRAC = 0.85

ORIENTATION_BLUR_SIGMA = 1.5

ORIENTATION_GLINT_EXCLUDE_FRAC = 0.3

ORIENTATION_MIN_STRENGTH = 0.0


# ============================================================
# NEW:
# GLINT IGNORE / CONTINUE PARAMETERS
# ============================================================

# Radius around the detected glint that should be ignored
# while traversing the intensity profile.
#
# Example:
# glint at sample 50
# ignore samples 48,49,50,51,52
# then continue with sample 53...
#
GLINT_PROFILE_IGNORE_RADIUS_PX = 7.0


# Additional brightness-spike protection.
#
# If a sampled point is much brighter than its local neighbours,
# it can also be considered a bright outlier.
#
GLINT_SPIKE_MIN_RISE = 20.0


# Number of neighbouring points used to check a brightness spike.
GLINT_SPIKE_NEIGHBOR_POINTS = 2


# ============================================================
# AVERAGE PIXEL PARAMETERS
# ============================================================

# Number of neighbouring valid profile pixels used to create
# one averaged intensity point.
AVERAGE_PROFILE_PIXELS = 5

# Fit a straight line to the averaged intensity points.
FIT_AVERAGED_PROFILE = True


# ============================================================
# ANNULUS MEDIAN
# ============================================================

def annulus_median(gray, cx, cy, r1, r2):

    h, w = gray.shape

    R = int(r2 + 2)

    x1 = max(0, int(cx - R))
    y1 = max(0, int(cy - R))

    x2 = min(w, int(cx + R + 1))
    y2 = min(h, int(cy + R + 1))

    crop = gray[y1:y2, x1:x2]

    yy, xx = np.ogrid[y1:y2, x1:x2]

    d2 = (xx - cx) ** 2 + (yy - cy) ** 2

    mask = (
        (d2 >= r1 * r1)
        &
        (d2 < r2 * r2)
    )

    values = crop[mask]

    if values.size == 0:
        return 255.0

    return float(np.median(values))


# ============================================================
# FIND GLINT CANDIDATES
# ============================================================

def find_glint_candidates(gray):

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    background = cv2.GaussianBlur(
        gray,
        (BACKGROUND_KERNEL, BACKGROUND_KERNEL),
        0
    ).astype(np.int16)

    diff = (
        blurred.astype(np.int16)
        -
        background
    )

    local_max = cv2.dilate(
        blurred,
        np.ones(
            (LOCAL_MAX_KERNEL, LOCAL_MAX_KERNEL),
            np.uint8
        )
    )

    candidate_mask = (
        (blurred >= local_max)
        &
        (diff >= MIN_LOCAL_CONTRAST)
        &
        (blurred >= MIN_BRIGHT_VALUE)
    )

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8),
        connectivity=8
    )

    h, w = gray.shape

    candidates = []

    for i in range(1, n):

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        if area > 80:
            continue

        cx, cy = centroids[i]

        cx = int(round(cx))
        cy = int(round(cy))

        if (
            cx < 3
            or cy < 3
            or cx >= w - 3
            or cy >= h - 3
        ):
            continue

        peak = int(
            blurred[cy, cx]
        )

        bg = int(
            background[cy, cx]
        )

        candidates.append(
            {
                "x": cx,
                "y": cy,
                "peak": peak,
                "background": bg,
                "area": area
            }
        )

    return candidates


# ============================================================
# PUPIL CONTEXT SCORE
# ============================================================

def pupil_context_score(gray, c):

    cx = c["x"]
    cy = c["y"]

    peak = c["peak"]

    near = annulus_median(
        gray,
        cx,
        cy,
        5,
        11
    )

    pupil_ring = annulus_median(
        gray,
        cx,
        cy,
        PUPIL_RING_INNER,
        PUPIL_RING_OUTER
    )

    outer_ring = annulus_median(
        gray,
        cx,
        cy,
        30,
        44
    )

    glint_contrast = peak - near

    pupil_darkness = 255.0 - pupil_ring

    score = (
        PUPIL_DARK_WEIGHT * pupil_darkness
        +
        0.55 * peak
        +
        0.70 * max(
            0.0,
            glint_contrast
        )
        +
        0.15 * max(
            0.0,
            outer_ring - pupil_ring
        )
    )

    return {
        "score": float(score),
        "near": float(near),
        "pupil_ring": float(pupil_ring),
        "outer_ring": float(outer_ring),
        "glint_contrast": float(glint_contrast)
    }


# ============================================================
# PREPARE CANDIDATES
# ============================================================

def prepare_candidates(gray, candidates):

    prepared = []

    for c in candidates:

        f = pupil_context_score(
            gray,
            c
        )

        if (
            f["glint_contrast"]
            <
            MIN_GLINT_CONTRAST
        ):
            continue

        item = dict(c)

        item.update(f)

        prepared.append(item)

    return prepared


# ============================================================
# CHOOSE EYE PAIR
# ============================================================

def choose_eye_pair(gray, candidates):

    h, w = gray.shape

    left = [
        c for c in candidates
        if c["x"] < w / 2
    ]

    right = [
        c for c in candidates
        if c["x"] >= w / 2
    ]

    if not left or not right:
        return None

    left.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    right.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    left = left[
        :MAX_CANDIDATES_PER_SIDE
    ]

    right = right[
        :MAX_CANDIDATES_PER_SIDE
    ]

    best = None

    for L in left:

        for R in right:

            dx = R["x"] - L["x"]

            dy = abs(
                R["y"] - L["y"]
            )

            sep_frac = dx / float(w)

            y_frac = dy / float(h)

            if (
                sep_frac
                <
                MIN_EYE_SEPARATION_FRAC
                or
                sep_frac
                >
                MAX_EYE_SEPARATION_FRAC
            ):
                continue

            if (
                y_frac
                >
                MAX_EYE_Y_DIFFERENCE_FRAC
            ):
                continue

            pair_score = (
                L["score"]
                +
                R["score"]
            )

            pair_score += (
                100.0 * sep_frac
            )

            pair_score -= (
                120.0
                *
                (
                    y_frac
                    /
                    MAX_EYE_Y_DIFFERENCE_FRAC
                ) ** 2
            )

            if (
                best is None
                or
                pair_score
                >
                best["pair_score"]
            ):

                best = {
                    "left": L,
                    "right": R,
                    "pair_score": pair_score
                }

    return best


# ============================================================
# VERIFY PUPIL CONTEXT
# ============================================================

def verify_pupil_context(gray, c):

    pupil_ring = c["pupil_ring"]

    darkness_ok = (
        pupil_ring <= 90
    )

    glint_ok = (
        c["glint_contrast"]
        >=
        MIN_GLINT_CONTRAST
    )

    return bool(
        darkness_ok
        and
        glint_ok
    )


# ============================================================
# SELECT FINAL GLINT
# ============================================================

def select_final_glint(
    gray,
    selected_pupil,
    all_candidates
):

    px = selected_pupil["x"]
    py = selected_pupil["y"]

    pupil_search_radius = 32

    inside = []

    for c in all_candidates:

        d = np.hypot(
            c["x"] - px,
            c["y"] - py
        )

        if d <= pupil_search_radius:

            inside.append(c)

    if not inside:
        return None

    inside.sort(
        key=lambda c: (
            c["score"],
            c["peak"]
        ),
        reverse=True
    )

    return inside[0]


# ============================================================
# CIRCLE FIT PARAMETERS
# ============================================================

_N_ANG = 64

_ANG = np.linspace(
    0,
    2 * np.pi,
    _N_ANG,
    endpoint=False
).astype(np.float32)

_COS = np.cos(_ANG)
_SIN = np.sin(_ANG)

_EDGE_DELTA = 2.5

_SIZE_BONUS = 0.002


# ============================================================
# SAMPLE IMAGE
# ============================================================

def _sample(gray, xs, ys):

    return cv2.remap(
        gray,
        np.ascontiguousarray(
            xs,
            dtype=np.float32
        ),
        np.ascontiguousarray(
            ys,
            dtype=np.float32
        ),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# ============================================================
# CONTRAST SCALE
# ============================================================

def _contrast_scale(gray, seed):

    sx, sy = seed

    h, w = gray.shape

    x1 = max(
        0,
        int(sx - 76)
    )

    x2 = min(
        w,
        int(sx + 77)
    )

    y1 = max(
        0,
        int(sy - 76)
    )

    y2 = min(
        h,
        int(sy + 77)
    )

    crop = gray[
        y1:y2,
        x1:x2
    ]

    yy, xx = np.ogrid[
        y1:y2,
        x1:x2
    ]

    d2 = (
        (xx - sx) ** 2
        +
        (yy - sy) ** 2
    )

    inner = crop[
        d2 <= 15 ** 2
    ]

    ring = crop[
        (d2 >= 45 ** 2)
        &
        (d2 <= 75 ** 2)
    ]

    hi = np.percentile(
        inner,
        85
    )

    if ring.size:

        lo = np.percentile(
            ring,
            25
        )

    else:

        lo = np.percentile(
            gray,
            10
        )

    return float(
        max(
            hi - lo,
            8.0
        )
    )


# ============================================================
# PERIMETER SCORE
# ============================================================

def _perimeter_score(
    gray,
    cx,
    cy,
    radii,
    tau
):

    rs = radii[:, None]

    outer = _sample(
        gray,
        cx
        +
        (rs + _EDGE_DELTA)
        *
        _COS[None, :],

        cy
        +
        (rs + _EDGE_DELTA)
        *
        _SIN[None, :]
    )

    inner = _sample(
        gray,
        cx
        +
        (rs - _EDGE_DELTA)
        *
        _COS[None, :],

        cy
        +
        (rs - _EDGE_DELTA)
        *
        _SIN[None, :]
    )

    d = np.clip(
        (inner - outer) / tau,
        -0.15,
        1.0
    )

    return (
        d.mean(axis=1)
        +
        _SIZE_BONUS * radii
    )


# ============================================================
# COARSE FIT
# ============================================================

def _coarse_fit(
    gray,
    seed,
    tau
):

    sx, sy = seed

    radii_all = np.arange(
        PUPIL_R_MIN,
        PUPIL_R_MAX + 1,
        2.0,
        dtype=np.float32
    )

    best_score = -1e9

    best = None

    for cy in np.arange(
        sy - PUPIL_SEARCH,
        sy + PUPIL_SEARCH + 1,
        3.0
    ):

        for cx in np.arange(
            sx - PUPIL_SEARCH,
            sx + PUPIL_SEARCH + 1,
            3.0
        ):

            d_seed = np.hypot(
                cx - sx,
                cy - sy
            )

            radii = radii_all[
                radii_all
                >=
                d_seed / 0.9
            ]

            if radii.size == 0:
                continue

            sc = _perimeter_score(
                gray,
                cx,
                cy,
                radii,
                tau
            )

            k = int(
                np.argmax(sc)
            )

            if (
                sc[k]
                >
                best_score
            ):

                best_score = float(
                    sc[k]
                )

                best = (
                    float(cx),
                    float(cy),
                    float(radii[k])
                )

    if best is None:
        return None

    sx_, sy_ = seed

    def score(p):

        if not (
            PUPIL_R_MIN
            <=
            p[2]
            <=
            PUPIL_R_MAX
        ):
            return -1e9

        if (
            np.hypot(
                p[0] - sx_,
                p[1] - sy_
            )
            >
            0.95 * p[2]
        ):
            return -1e9

        return float(
            _perimeter_score(
                gray,
                p[0],
                p[1],
                np.array(
                    [p[2]],
                    np.float32
                ),
                tau
            )[0]
        )

    p = np.array(
        best,
        dtype=np.float64
    )

    cur = score(p)

    for step in (
        2.0,
        1.0,
        0.5,
        0.25
    ):

        for _ in range(40):

            improved = False

            for dim in range(3):

                for sgn in (
                    1.0,
                    -1.0
                ):

                    q = p.copy()

                    q[dim] += (
                        sgn * step
                    )

                    s = score(q)

                    if (
                        s
                        >
                        cur + 1e-6
                    ):

                        p = q
                        cur = s

                        improved = True

            if not improved:
                break

    return p, cur


# ============================================================
# LEAST-SQUARE CIRCLE
# ============================================================

def _fit_circle_lsq(pts):

    A = np.c_[
        2 * pts[:, 0],
        2 * pts[:, 1],
        np.ones(len(pts))
    ]

    y = (
        pts ** 2
    ).sum(1)

    cx, cy, k = np.linalg.lstsq(
        A,
        y,
        rcond=None
    )[0]

    return np.array(
        [
            cx,
            cy,
            np.sqrt(
                max(
                    k + cx * cx + cy * cy,
                    1e-6
                )
            )
        ]
    )


# ============================================================
# REFINE HALF MAX
# ============================================================

def _refine_half_max(
    gray,
    c0,
    contrast,
    n_rays=180
):

    cx, cy, r = c0

    ts = np.arange(
        0.4 * r,
        1.8 * r,
        0.25
    )

    ang = np.linspace(
        0,
        2 * np.pi,
        n_rays,
        endpoint=False
    )

    xs = (
        cx
        +
        ts[None, :]
        *
        np.cos(ang)[:, None]
    )

    ys = (
        cy
        +
        ts[None, :]
        *
        np.sin(ang)[:, None]
    )

    prof = _sample(
        gray,
        xs,
        ys
    )

    in_sel = (
        (ts >= 0.4 * r)
        &
        (ts <= 0.8 * r)
    )

    out_sel = (
        (ts >= 1.3 * r)
        &
        (ts <= 1.7 * r)
    )

    j0 = int(
        np.searchsorted(
            ts,
            0.55 * r
        )
    )

    pts = []

    for k in range(n_rays):

        p = prof[k]

        inn = np.median(
            p[in_sel]
        )

        out = np.median(
            p[out_sel]
        )

        if (
            inn - out
            <
            max(
                4.0,
                0.12 * contrast
            )
        ):
            continue

        mid = 0.5 * (
            inn + out
        )

        below = p < mid

        hit = None

        for j in range(
            j0,
            len(ts) - 5
        ):

            if below[
                j:j + 5
            ].all():

                hit = j
                break

        if hit is None or hit == 0:
            continue

        t = (
            ts[hit - 1]
            +
            (
                p[hit - 1] - mid
            )
            /
            (
                p[hit - 1]
                -
                p[hit]
                +
                1e-9
            )
            *
            (
                ts[hit]
                -
                ts[hit - 1]
            )
        )

        pts.append(
            (
                cx
                +
                t * np.cos(ang[k]),

                cy
                +
                t * np.sin(ang[k])
            )
        )

    pts = np.array(pts)

    if len(pts) < 0.35 * n_rays:
        return None

    keep = np.ones(
        len(pts),
        bool
    )

    for _ in range(10):

        c = _fit_circle_lsq(
            pts[keep]
        )

        res = np.abs(
            np.hypot(
                pts[:, 0] - c[0],
                pts[:, 1] - c[1]
            )
            -
            c[2]
        )

        new_keep = (
            res
            <
            max(
                1.0,
                2.0 * np.median(
                    res[keep]
                )
            )
        )

        if (
            new_keep.sum()
            <
            0.3 * n_rays
            or
            (
                new_keep == keep
            ).all()
        ):
            break

        keep = new_keep

    c = _fit_circle_lsq(
        pts[keep]
    )

    return (
        c,
        float(
            keep.sum()
        ) / n_rays,
        pts,
        keep
    )


# ============================================================
# FIT PUPIL CIRCLE
# ============================================================

def fit_pupil_circle(
    gray_blur,
    seed
):

    contrast = _contrast_scale(
        gray_blur,
        seed
    )

    tau = 0.3 * contrast

    coarse = _coarse_fit(
        gray_blur,
        seed,
        tau
    )

    if coarse is None:
        return None

    c = coarse[0]

    method = "coarse"

    quality = float(
        np.clip(
            coarse[1],
            0.0,
            1.0
        )
    )

    contour_pts = None
    contour_keep = None

    for _ in range(4):

        out = _refine_half_max(
            gray_blur,
            c,
            contrast
        )

        if out is None:
            break

        new_c, frac, pts, keep = out

        shift = np.hypot(
            new_c[0] - c[0],
            new_c[1] - c[1]
        )

        if (
            shift > 0.5 * c[2]
            or
            not (
                0.7
                <
                new_c[2] / c[2]
                <
                1.5
            )
        ):
            break

        moved = (
            shift
            +
            abs(
                new_c[2] - c[2]
            )
        )

        c = new_c

        quality = frac

        method = "half_max"

        contour_pts = pts

        contour_keep = keep

        if moved < 0.15:
            break

    cx = float(c[0])
    cy = float(c[1])
    r = float(c[2])

    h, w = gray_blur.shape

    x1 = max(
        0,
        int(cx - 2 * r)
    )

    x2 = min(
        w,
        int(cx + 2 * r) + 1
    )

    y1 = max(
        0,
        int(cy - 2 * r)
    )

    y2 = min(
        h,
        int(cy + 2 * r) + 1
    )

    yy, xx = np.ogrid[
        y1:y2,
        x1:x2
    ]

    dist = np.hypot(
        xx - cx,
        yy - cy
    )

    crop = gray_blur[
        y1:y2,
        x1:x2
    ]

    inside = crop[
        dist <= 0.8 * r
    ]

    ring = crop[
        (dist >= 1.25 * r)
        &
        (dist <= 1.7 * r)
    ]

    ratio = (
        float(
            inside.mean()
            /
            max(
                ring.mean(),
                1.0
            )
        )
        if inside.size
        and ring.size
        else 0.0
    )

    needs_review = (
        method != "half_max"
        or
        quality < 0.6
        or
        r < PUPIL_REVIEW_R_BELOW
        or
        ratio < PUPIL_MIN_BRIGHT_RATIO
        or
        r <= PUPIL_R_MIN + 1.5
        or
        r >= PUPIL_R_MAX - 1.5
        or
        np.hypot(
            cx - seed[0],
            cy - seed[1]
        ) > r
        or
        cx - r < 0
        or
        cy - r < 0
        or
        cx + r > w
        or
        cy + r > h
    )

    return {
        "cx": cx,
        "cy": cy,
        "r": r,
        "quality": float(
            quality
        ),
        "ratio": ratio,
        "needs_review": int(
            bool(needs_review)
        ),
        "contour": contour_pts,
        "contour_keep": contour_keep
    }


# ============================================================
# DRAW CIRCLE
# ============================================================

_SH = 4


def _draw_circle(
    img,
    cx,
    cy,
    r,
    color,
    thickness
):

    cv2.circle(
        img,
        (
            int(
                round(
                    cx * (1 << _SH)
                )
            ),
            int(
                round(
                    cy * (1 << _SH)
                )
            )
        ),
        int(
            round(
                r * (1 << _SH)
            )
        ),
        color,
        thickness,
        cv2.LINE_AA
        if thickness > 0
        else cv2.LINE_8,
        _SH
    )


# ============================================================
# DRAW RAW CONTOUR
# ============================================================

def draw_raw_contour(
    img,
    p,
    thickness=1
):

    if p is None:
        return

    pts = p.get(
        "contour"
    )

    if pts is None or len(pts) < 3:
        return

    fixed = np.round(
        pts * (1 << _SH)
    ).astype(
        np.int32
    ).reshape(
        -1,
        1,
        2
    )

    cv2.polylines(
        img,
        [fixed],
        True,
        CONTOUR_COLOR,
        thickness,
        cv2.LINE_AA,
        _SH
    )


# ============================================================
# MAKE PUPIL MASK
# ============================================================

def make_pupil_mask(
    shape,
    pupils
):

    mask = np.zeros(
        shape[:2],
        np.uint8
    )

    for p in pupils:

        if p is not None:

            _draw_circle(
                mask,
                p["cx"],
                p["cy"],
                p["r"],
                255,
                -1
            )

    return mask


# ============================================================
# DRAW GLINT
# ============================================================

def draw_glint(
    image,
    g
):

    if g is None:
        return

    gx = int(
        g["x"]
    )

    gy = int(
        g["y"]
    )

    cv2.circle(
        image,
        (gx, gy),
        3,
        (0, 255, 0),
        -1,
        cv2.LINE_AA
    )


# ============================================================
# COMPUTE PUPIL ORIENTATION
# ============================================================

def compute_pupil_orientation(
    gray,
    p,
    glint=None
):

    if p is None:
        return None

    contour = p.get(
        "contour"
    )

    if (
        contour is None
        or
        len(contour) < 3
    ):
        return None

    cx = float(
        p["cx"]
    )

    cy = float(
        p["cy"]
    )

    r = float(
        p["r"]
    )

    h, w = gray.shape

    contour_pts = np.round(
        contour
    ).astype(
        np.int32
    ).reshape(
        -1,
        1,
        2
    )

    mask = np.zeros(
        (h, w),
        np.uint8
    )

    cv2.fillPoly(
        mask,
        [contour_pts],
        255
    )

    blurred = cv2.GaussianBlur(
        gray.astype(
            np.float32
        ),
        (0, 0),
        ORIENTATION_BLUR_SIGMA
    )

    # --------------------------------------------------------
    # IGNORE GLINT WHILE CALCULATING ORIENTATION
    # --------------------------------------------------------

    if glint is not None:

        ex_r = max(
            3.0,
            ORIENTATION_GLINT_EXCLUDE_FRAC * r
        )

        gx = float(
            glint["x"]
        )

        gy = float(
            glint["y"]
        )

        yy, xx = np.ogrid[
            :h,
            :w
        ]

        glint_mask = (
            (xx - gx) ** 2
            +
            (yy - gy) ** 2
            <=
            ex_r ** 2
        )

        mask[
            glint_mask
        ] = 0

    ys, xs = np.nonzero(
        mask > 0
    )

    if len(xs) < 20:
        return None

    values = blurred[
        ys,
        xs
    ].astype(
        np.float64
    )

    values = (
        values
        -
        values.min()
    )

    total = float(
        values.sum()
    )

    if total <= 1e-6:
        return None

    wx = float(
        np.sum(
            xs * values
        )
        /
        total
    )

    wy = float(
        np.sum(
            ys * values
        )
        /
        total
    )

    vx = wx - cx
    vy = wy - cy

    mag = float(
        np.hypot(
            vx,
            vy
        )
    )

    if mag < 1e-6:

        return {
            "angle_deg": None,
            "dx": 0.0,
            "dy": 0.0,
            "strength": 0.0,
            "ok": False,
            "dim_point": (
                cx,
                cy
            ),
            "bright_point": (
                cx,
                cy
            )
        }

    ux = vx / mag
    uy = vy / mag

    angle_deg = float(
        np.degrees(
            np.arctan2(
                uy,
                ux
            )
        )
    )

    strength = (
        mag
        /
        max(
            r,
            1e-6
        )
    )

    arrow_len = max(
        3.0,
        0.75 * r
    )

    return {
        "angle_deg": angle_deg,
        "dx": ux,
        "dy": uy,
        "strength": strength,
        "ok": (
            strength
            >=
            ORIENTATION_MIN_STRENGTH
        ),
        "dim_point": (
            cx - ux * arrow_len,
            cy - uy * arrow_len
        ),
        "bright_point": (
            cx + ux * arrow_len,
            cy + uy * arrow_len
        )
    }


# ============================================================
# NEW:
# CHECK WHETHER A PROFILE POINT IS THE GLINT
# ============================================================

def is_near_detected_glint(
    x,
    y,
    glint
):

    if glint is None:
        return False

    gx = float(
        glint["x"]
    )

    gy = float(
        glint["y"]
    )

    distance = np.hypot(
        x - gx,
        y - gy
    )

    return (
        distance
        <=
        GLINT_PROFILE_IGNORE_RADIUS_PX
    )


# ============================================================
# NEW:
# DETECT BRIGHT SPIKE IN PROFILE
# ============================================================

def is_bright_profile_spike(
    intensities,
    index
):

    n = len(
        intensities
    )

    k = GLINT_SPIKE_NEIGHBOR_POINTS

    if (
        index < k
        or
        index >= n - k
    ):
        return False

    current = float(
        intensities[index]
    )

    left_values = intensities[
        index - k:index
    ]

    right_values = intensities[
        index + 1:index + 1 + k
    ]

    neighbours = np.concatenate(
        [
            left_values,
            right_values
        ]
    )

    if neighbours.size == 0:
        return False

    neighbour_median = float(
        np.median(
            neighbours
        )
    )

    rise = (
        current
        -
        neighbour_median
    )

    return (
        rise
        >=
        GLINT_SPIKE_MIN_RISE
    )


# ============================================================
# AVERAGE VALID PROFILE PIXELS
# ============================================================

def average_valid_profile_pixels(
    distances,
    xs,
    ys,
    intensities,
    ignored,
    group_size=5
):
    """
    Average neighbouring valid pixels.

    Glint/spike samples are already marked in ``ignored``.
    Averaging is done only inside continuous valid regions,
    so a group can NEVER cross an ignored glint region.
    """

    distances = np.asarray(
        distances,
        dtype=np.float64
    ).reshape(-1)

    xs = np.asarray(
        xs,
        dtype=np.float64
    ).reshape(-1)

    ys = np.asarray(
        ys,
        dtype=np.float64
    ).reshape(-1)

    intensities = np.asarray(
        intensities,
        dtype=np.float64
    ).reshape(-1)

    ignored = np.asarray(
        ignored,
        dtype=bool
    ).reshape(-1)

    if group_size < 1:
        raise ValueError(
            "group_size must be >= 1"
        )

    n = len(intensities)

    if n == 0:
        return {
            "distances": np.array([], dtype=np.float64),
            "xs": np.array([], dtype=np.float64),
            "ys": np.array([], dtype=np.float64),
            "intensities": np.array([], dtype=np.float64)
        }

    valid_indices = np.flatnonzero(
        ~ignored
    )

    if len(valid_indices) == 0:
        return {
            "distances": np.array([], dtype=np.float64),
            "xs": np.array([], dtype=np.float64),
            "ys": np.array([], dtype=np.float64),
            "intensities": np.array([], dtype=np.float64)
        }

    # Split into continuous valid runs.
    # This prevents averaging across an ignored glint/spike.
    breaks = np.where(
        np.diff(valid_indices) != 1
    )[0] + 1

    runs = np.split(
        valid_indices,
        breaks
    )

    avg_distances = []
    avg_xs = []
    avg_ys = []
    avg_intensities = []

    for run in runs:

        if len(run) == 0:
            continue

        for start_index in range(
            0,
            len(run),
            group_size
        ):

            group = run[
                start_index:start_index + group_size
            ]

            if len(group) == 0:
                continue

            avg_distances.append(
                float(
                    np.mean(
                        distances[group]
                    )
                )
            )

            avg_xs.append(
                float(
                    np.mean(
                        xs[group]
                    )
                )
            )

            avg_ys.append(
                float(
                    np.mean(
                        ys[group]
                    )
                )
            )

            avg_intensities.append(
                float(
                    np.mean(
                        intensities[group]
                    )
                )
            )

    return {
        "distances": np.asarray(
            avg_distances,
            dtype=np.float64
        ),
        "xs": np.asarray(
            avg_xs,
            dtype=np.float64
        ),
        "ys": np.asarray(
            avg_ys,
            dtype=np.float64
        ),
        "intensities": np.asarray(
            avg_intensities,
            dtype=np.float64
        )
    }


# ============================================================
# UPDATED:
# EXTRACT INTENSITY PROFILE
#
# IMPORTANT:
# When the scan reaches the bright glint:
#
#     1. Detect glint
#     2. Ignore it
#     3. DO NOT STOP
#     4. Continue to the next point
#
# After glint/spike removal:
#
#     5. Average neighbouring valid pixels
#     6. Fit a straight line to the averaged points
#
# ============================================================

def extract_angle_intensity_profile(
    gray,
    pupil,
    orientation,
    glint=None
):

    if (
        gray is None
        or
        pupil is None
        or
        orientation is None
    ):
        return None

    if not orientation.get(
        "ok",
        False
    ):
        return None

    cx = float(
        pupil["cx"]
    )

    cy = float(
        pupil["cy"]
    )

    r = float(
        pupil["r"]
    )

    dx = float(
        orientation["dx"]
    )

    dy = float(
        orientation["dy"]
    )

    norm = math.hypot(
        dx,
        dy
    )

    if (
        norm < 1e-9
        or
        r <= 1
    ):
        return None

    dx /= norm
    dy /= norm

    half_len = (
        0.75 * r
    )

    distances = np.linspace(
        -half_len,
        half_len,
        INTENSITY_PROFILE_SAMPLES
    )

    xs = (
        cx
        +
        distances * dx
    )

    ys = (
        cy
        +
        distances * dy
    )

    h, w = gray.shape[:2]

    valid = (
        (xs >= 0)
        &
        (xs <= w - 1)
        &
        (ys >= 0)
        &
        (ys <= h - 1)
    )

    if np.count_nonzero(
        valid
    ) < 3:
        return None

    distances = distances[
        valid
    ]

    xs = xs[
        valid
    ]

    ys = ys[
        valid
    ]

    # --------------------------------------------------------
    # SAMPLE ORIGINAL IMAGE
    # --------------------------------------------------------

    intensities = np.asarray(
        cv2.remap(
            gray,
            xs.astype(
                np.float32
            ),
            ys.astype(
                np.float32
            ),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        ),
        dtype=np.float64
    ).reshape(-1)

    distances = np.asarray(
        distances,
        dtype=np.float64
    ).reshape(-1)

    xs = np.asarray(
        xs,
        dtype=np.float64
    ).reshape(-1)

    ys = np.asarray(
        ys,
        dtype=np.float64
    ).reshape(-1)

    # ========================================================
    # NEW GLINT IGNORE LOGIC
    # ========================================================

    ignored = np.zeros(
        len(intensities),
        dtype=bool
    )

    # --------------------------------------------------------
    # STEP 1:
    # Ignore points physically close to detected glint.
    # --------------------------------------------------------

    if glint is not None:

        for i in range(
            len(intensities)
        ):

            if is_near_detected_glint(
                xs[i],
                ys[i],
                glint
            ):

                ignored[i] = True

    # --------------------------------------------------------
    # STEP 2:
    # Also detect a sudden bright spike.
    #
    # This protects the algorithm if the exact glint centre
    # falls between two sampled pixels.
    # --------------------------------------------------------

    original_intensities = (
        intensities.copy()
    )

    for i in range(
        len(intensities)
    ):

        if ignored[i]:
            continue

        if is_bright_profile_spike(
            original_intensities,
            i
        ):

            # Only mark it as a glint-like spike if it is
            # reasonably bright.
            if (
                original_intensities[i]
                >=
                MIN_BRIGHT_VALUE
            ):

                ignored[i] = True

    # ========================================================
    # IMPORTANT:
    #
    # We DO NOT terminate at the ignored point.
    #
    # The remaining points after the glint are still processed.
    # ========================================================

    valid_profile = (
        ~ignored
    )

    if np.count_nonzero(
        valid_profile
    ) < 3:
        return None

    filtered_distances = (
        distances[
            valid_profile
        ]
    )

    filtered_xs = (
        xs[
            valid_profile
        ]
    )

    filtered_ys = (
        ys[
            valid_profile
        ]
    )

    filtered_intensities = (
        intensities[
            valid_profile
        ]
    )

    if len(
        filtered_intensities
    ) < 3:
        return None

    # ========================================================
    # AVERAGE NEIGHBOURING VALID PIXELS
    #
    # Averaging happens AFTER glint/spike removal.
    # A group can never cross an ignored glint region.
    # ========================================================

    averaged = average_valid_profile_pixels(
        distances,
        xs,
        ys,
        intensities,
        ignored,
        group_size=AVERAGE_PROFILE_PIXELS
    )

    avg_distances = averaged[
        "distances"
    ]

    avg_xs = averaged[
        "xs"
    ]

    avg_ys = averaged[
        "ys"
    ]

    avg_intensities = averaged[
        "intensities"
    ]

    if len(
        avg_intensities
    ) < 2:
        return None

    # ========================================================
    # FIT STRAIGHT LINE TO AVERAGED POINTS
    # ========================================================

    fit_slope = None
    fit_intercept = None
    fit_r2 = None

    if FIT_AVERAGED_PROFILE:

        fit_slope, fit_intercept = np.polyfit(
            avg_distances,
            avg_intensities,
            1
        )

        fitted_values = (
            fit_slope
            *
            avg_distances
            +
            fit_intercept
        )

        ss_res = np.sum(
            (
                avg_intensities
                -
                fitted_values
            ) ** 2
        )

        ss_tot = np.sum(
            (
                avg_intensities
                -
                np.mean(
                    avg_intensities
                )
            ) ** 2
        )

        if ss_tot > 1e-12:

            fit_r2 = (
                1.0
                -
                ss_res / ss_tot
            )

        else:

            fit_r2 = 1.0

    # ========================================================
    # DARK / BRIGHT POINTS
    #
    # These are calculated from the AVERAGED profile.
    # ========================================================

    dark_index = int(
        np.argmin(
            avg_intensities
        )
    )

    bright_index = int(
        np.argmax(
            avg_intensities
        )
    )

    return {
        "angle_deg": float(
            orientation["angle_deg"]
        ),

        # Averaged profile used by the graph and CSV.
        "distances": avg_distances,

        "xs": avg_xs,

        "ys": avg_ys,

        "intensities": avg_intensities,

        "dark_index": dark_index,

        "bright_index": bright_index,

        "dark_intensity": float(
            avg_intensities[
                dark_index
            ]
        ),

        "bright_intensity": float(
            avg_intensities[
                bright_index
            ]
        ),

        # Straight-line fit of averaged points.
        "fit_slope": (
            None
            if fit_slope is None
            else float(fit_slope)
        ),

        "fit_intercept": (
            None
            if fit_intercept is None
            else float(fit_intercept)
        ),

        "fit_r2": (
            None
            if fit_r2 is None
            else float(fit_r2)
        ),

        # Filtered individual values retained for debugging.
        "filtered_distances": filtered_distances,

        "filtered_xs": filtered_xs,

        "filtered_ys": filtered_ys,

        "filtered_intensities": filtered_intensities,

        # Original sampled values retained as before.
        "original_distances": distances,

        "original_xs": xs,

        "original_ys": ys,

        "original_intensities": (
            original_intensities
        ),

        "ignored_mask": ignored,

        "ignored_count": int(
            np.count_nonzero(
                ignored
            )
        )
    }


# ============================================================
# SAVE INTENSITY GRAPH
# ============================================================

def save_angle_intensity_graph(
    profile,
    output_path,
    image_name,
    eye
):

    import matplotlib

    matplotlib.use(
        "Agg"
    )

    import matplotlib.pyplot as plt

    x = np.asarray(
        profile["distances"],
        dtype=float
    ).reshape(-1)

    y = np.asarray(
        profile["intensities"],
        dtype=float
    ).reshape(-1)

    di = int(
        profile["dark_index"]
    )

    bi = int(
        profile["bright_index"]
    )

    if (
        x.size == 0
        or
        y.size == 0
        or
        x.size != y.size
    ):

        raise ValueError(
            f"Invalid averaged intensity profile for "
            f"{image_name} ({eye})"
        )

    slope = profile.get(
        "fit_slope"
    )

    intercept = profile.get(
        "fit_intercept"
    )

    r2 = profile.get(
        "fit_r2"
    )

    plt.figure(
        figsize=(8, 5)
    )

    # --------------------------------------------------------
    # AVERAGED PROFILE
    # --------------------------------------------------------

    plt.scatter(
        x,
        y,
        s=35,
        label=(
            f"Averaged intensity "
            f"({AVERAGE_PROFILE_PIXELS} pixels)"
        )
    )

    # --------------------------------------------------------
    # STRAIGHT-LINE FIT
    # --------------------------------------------------------

    if (
        slope is not None
        and
        intercept is not None
    ):

        sort_idx = np.argsort(
            x
        )

        x_sorted = x[
            sort_idx
        ]

        fitted_y = (
            slope
            *
            x_sorted
            +
            intercept
        )

        fit_label = (
            f"Linear fit: "
            f"y = {slope:.4f}x + {intercept:.4f}"
        )

        if r2 is not None:
            fit_label += (
                f"\nR² = {r2:.4f}"
            )

        plt.plot(
            x_sorted,
            fitted_y,
            linewidth=2.0,
            label=fit_label
        )

    # --------------------------------------------------------
    # DARK POINT
    # --------------------------------------------------------

    plt.scatter(
        [x[di]],
        [y[di]],
        s=55,
        label=(
            f"Dark = "
            f"{y[di]:.1f}"
        )
    )

    # --------------------------------------------------------
    # BRIGHT POINT
    # --------------------------------------------------------

    plt.scatter(
        [x[bi]],
        [y[bi]],
        s=55,
        label=(
            f"Bright = "
            f"{y[bi]:.1f}"
        )
    )

    plt.xlabel(
        "Distance along detected angle direction (pixels)\n"
        "Dark side → Bright side"
    )

    plt.ylabel(
        "Average grayscale intensity (0–255)"
    )

    title = (
        f"{eye.capitalize()} eye | "
        f"Angle = "
        f"{profile['angle_deg']:.1f}° | "
        f"Glint points ignored = "
        f"{profile['ignored_count']} | "
        f"Average = {AVERAGE_PROFILE_PIXELS} pixels"
    )

    if r2 is not None:
        title += (
            f" | R² = {r2:.4f}"
        )

    plt.title(
        title
    )

    plt.ylim(
        0,
        255
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160
    )

    plt.close()


# ============================================================
# SAVE SLOPE vs ANGLE GRAPH
# ============================================================

def save_slope_vs_angle_graph(
    slope_angle_rows,
    graph_path,
    csv_path
):
    """
    Create the HR-requested graph:

        X-axis = Angle (degrees)
        Y-axis = Slope

    slope_angle_rows contains one record per image + eye, so
    repeated intensity-profile samples are NOT plotted again.
    """

    if not slope_angle_rows:
        print(
            "No valid slope-vs-angle data available."
        )
        return

    import matplotlib

    matplotlib.use(
        "Agg"
    )

    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # Keep one point per image + eye.
    # This is an extra safety check in case duplicate rows
    # are ever added to slope_angle_rows.
    # --------------------------------------------------------

    unique = {}

    for row in slope_angle_rows:

        image_name = str(
            row["image"]
        )

        eye = str(
            row["eye"]
        ).lower()

        angle = row.get(
            "angle_deg"
        )

        slope = row.get(
            "slope"
        )

        r2 = row.get(
            "r2"
        )

        if (
            angle is None
            or slope is None
        ):
            continue

        try:
            angle = float(
                angle
            )

            slope = float(
                slope
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        if not (
            np.isfinite(
                angle
            )
            and
            np.isfinite(
                slope
            )
        ):
            continue

        key = (
            image_name,
            eye
        )

        unique[key] = {
            "image": image_name,
            "eye": eye,
            "angle_deg": angle,
            "slope": slope,
            "r2": (
                None
                if r2 is None
                else float(r2)
            )
        }

    rows = list(
        unique.values()
    )

    if not rows:
        print(
            "No valid slope-vs-angle points available."
        )
        return

    # --------------------------------------------------------
    # Save a compact CSV used by the graph.
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            csv_path
        ),
        exist_ok=True
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "eye",
                "angle_deg",
                "slope",
                "r2"
            ]
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    "image":
                        row["image"],

                    "eye":
                        row["eye"],

                    "angle_deg":
                        f"{row['angle_deg']:.6f}",

                    "slope":
                        f"{row['slope']:.6f}",

                    "r2":
                        ""
                        if row["r2"] is None
                        else f"{row['r2']:.6f}"
                }
            )

    # --------------------------------------------------------
    # Separate left and right eye.
    # --------------------------------------------------------

    left_rows = [
        row
        for row in rows
        if row["eye"] == "left"
    ]

    right_rows = [
        row
        for row in rows
        if row["eye"] == "right"
    ]

    # Sort by angle for cleaner visual ordering.
    left_rows.sort(
        key=lambda row:
        row["angle_deg"]
    )

    right_rows.sort(
        key=lambda row:
        row["angle_deg"]
    )

    plt.figure(
        figsize=(10, 6)
    )

    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    if left_rows:

        left_x = [
            row["angle_deg"]
            for row in left_rows
        ]

        left_y = [
            row["slope"]
            for row in left_rows
        ]

        plt.scatter(
            left_x,
            left_y,
            marker="o",
            s=45,
            label="Left Eye"
        )

    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    if right_rows:

        right_x = [
            row["angle_deg"]
            for row in right_rows
        ]

        right_y = [
            row["slope"]
            for row in right_rows
        ]

        plt.scatter(
            right_x,
            right_y,
            marker="x",
            s=55,
            label="Right Eye"
        )

    plt.xlabel(
        "Angle (degrees)"
    )

    plt.ylabel(
        "Slope"
    )

    plt.title(
        "Slope vs Angle"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    if (
        left_rows
        or
        right_rows
    ):
        plt.legend()

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(
            graph_path
        ),
        exist_ok=True
    )

    plt.savefig(
        graph_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Slope vs angle graph saved : "
        f"{graph_path}"
    )

    print(
        "Slope vs angle CSV saved   : "
        f"{csv_path}"
    )

    print(
        f"Slope vs angle data points : "
        f"{len(rows)}"
    )


# ============================================================
# SAVE 3-INPUT SLOPE vs ANGLE GRAPHS
# ============================================================

def save_multi_input_slope_vs_angle_graphs(
    all_rows,
    output_root
):
    """Create separate left-eye and right-eye slope-vs-angle graphs
    containing data from all configured input folders."""

    if not all_rows:
        print("No multi-input slope-vs-angle data available.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_root, exist_ok=True)

    # One point per image + eye + input.
    unique = {}

    for row in all_rows:
        input_id = str(row.get("input_id", ""))
        input_label = str(row.get("input_label", input_id))
        image_name = str(row.get("image", ""))
        eye = str(row.get("eye", "")).lower()
        angle = row.get("angle_deg")
        slope = row.get("slope")
        r2 = row.get("r2")

        if eye not in ("left", "right") or angle is None or slope is None:
            continue

        try:
            angle = float(angle)
            slope = float(slope)
        except (TypeError, ValueError):
            continue

        if not (np.isfinite(angle) and np.isfinite(slope)):
            continue

        try:
            r2_value = None if r2 is None else float(r2)
        except (TypeError, ValueError):
            r2_value = None

        key = (input_id, image_name, eye)

        unique[key] = {
            "input_id": input_id,
            "input_label": input_label,
            "image": image_name,
            "eye": eye,
            "angle_deg": angle,
            "slope": slope,
            "r2": r2_value
        }

    rows = list(unique.values())

    if not rows:
        print("No valid multi-input slope-vs-angle points available.")
        return

    # Save one combined CSV for all inputs.
    combined_csv = os.path.join(
        output_root,
        MULTI_SLOPE_ANGLE_CSV_FILE
    )

    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_id", "input_label", "image", "eye",
                "angle_deg", "slope", "r2"
            ]
        )
        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda x: (x["input_id"], x["eye"], x["image"])
        ):
            writer.writerow({
                "input_id": row["input_id"],
                "input_label": row["input_label"],
                "image": row["image"],
                "eye": row["eye"],
                "angle_deg": f"{row['angle_deg']:.6f}",
                "slope": f"{row['slope']:.6f}",
                "r2": "" if row["r2"] is None else f"{row['r2']:.6f}"
            })

    input_ids = []
    input_labels = {}
    for row in rows:
        if row["input_id"] not in input_ids:
            input_ids.append(row["input_id"])
            input_labels[row["input_id"]] = row["input_label"]

    # Cycle marker styles if there are more inputs than marker types.
    marker_list = [
        "o", "s", "^", "D", "v", "P", "X", "*",
        "<", ">", "p", "h", "8", "d", "H", "+"
    ]

    def create_eye_graph(eye_name, output_file):
        eye_rows = [row for row in rows if row["eye"] == eye_name]

        plt.figure(figsize=(10, 6))
        plotted = 0

        for idx, input_id in enumerate(input_ids):
            input_rows = [
                row for row in eye_rows
                if row["input_id"] == input_id
            ]

            if not input_rows:
                continue

            input_rows.sort(key=lambda row: row["angle_deg"])

            x = [row["angle_deg"] for row in input_rows]
            y = [row["slope"] for row in input_rows]

            plt.scatter(
                x,
                y,
                marker=marker_list[idx % len(marker_list)],
                s=45,
                label=(
                    f"Input {idx + 1}: "
                    f"{input_labels.get(input_id, input_id)}"
                )
            )

            plotted += len(input_rows)

        plt.xlabel("Angle (degrees)")
        plt.ylabel("Slope")
        plt.title(
            f"{eye_name.capitalize()} Eye - Slope vs Angle ({len(input_ids)} Inputs)"
        )
        plt.grid(True, alpha=0.3)

        if plotted:
            plt.legend()

        plt.tight_layout()

        output_path = os.path.join(output_root, output_file)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(
            f"{eye_name.capitalize()} eye graph saved : {output_path}"
        )
        print(
            f"{eye_name.capitalize()} eye points      : {plotted}"
        )

    create_eye_graph("left", MULTI_LEFT_GRAPH_FILE)
    create_eye_graph("right", MULTI_RIGHT_GRAPH_FILE)

    print(
        f"Combined CSV saved                  : {combined_csv}"
    )


# ============================================================
# CIRCULAR MEAN
# ============================================================

def circular_mean_degrees(
    angles
):

    valid = [
        float(a)
        for a in angles
        if (
            a is not None
            and
            np.isfinite(a)
        )
    ]

    if not valid:
        return None

    radians = np.radians(
        valid
    )

    s = float(
        np.mean(
            np.sin(radians)
        )
    )

    c = float(
        np.mean(
            np.cos(radians)
        )
    )

    if (
        abs(s) < 1e-12
        and
        abs(c) < 1e-12
    ):
        return valid[0]

    angle = float(
        np.degrees(
            np.arctan2(
                s,
                c
            )
        )
    )

    if angle > 180.0:

        angle -= 360.0

    elif angle <= -180.0:

        angle += 360.0

    return angle


# ============================================================
# COMMON ORIENTATION ANGLE
# ============================================================

def apply_common_orientation_angle(
    orientation,
    pupils
):

    raw_angles = []

    for side in (
        "left",
        "right"
    ):

        o = orientation.get(
            side
        )

        if (
            o is not None
            and
            o.get(
                "angle_deg"
            ) is not None
        ):

            raw_angles.append(
                o["angle_deg"]
            )

    common_angle = (
        circular_mean_degrees(
            raw_angles
        )
    )

    if common_angle is None:

        return (
            orientation,
            None
        )

    common_rad = math.radians(
        common_angle
    )

    ux = math.cos(
        common_rad
    )

    uy = math.sin(
        common_rad
    )

    updated = {}

    for side in (
        "left",
        "right"
    ):

        o = orientation.get(
            side
        )

        p = pupils.get(
            side
        )

        if (
            o is None
            or
            p is None
            or
            o.get(
                "angle_deg"
            ) is None
        ):

            updated[side] = o

            continue

        cx = float(
            p["cx"]
        )

        cy = float(
            p["cy"]
        )

        r = float(
            p["r"]
        )

        arrow_len = max(
            3.0,
            0.75 * r
        )

        updated[side] = dict(
            o
        )

        updated[side][
            "angle_deg"
        ] = common_angle

        updated[side][
            "dx"
        ] = ux

        updated[side][
            "dy"
        ] = uy

        updated[side][
            "dim_point"
        ] = (
            cx - ux * arrow_len,
            cy - uy * arrow_len
        )

        updated[side][
            "bright_point"
        ] = (
            cx + ux * arrow_len,
            cy + uy * arrow_len
        )

    return (
        updated,
        common_angle
    )


# ============================================================
# DRAW ORIENTATION ARROW
# ============================================================

def draw_orientation_arrow(
    img,
    o
):

    if (
        o is None
        or
        not o["ok"]
    ):
        return

    p1 = (
        int(
            round(
                o["dim_point"][0]
            )
        ),
        int(
            round(
                o["dim_point"][1]
            )
        )
    )

    p2 = (
        int(
            round(
                o["bright_point"][0]
            )
        ),
        int(
            round(
                o["bright_point"][1]
            )
        )
    )

    cv2.arrowedLine(
        img,
        p1,
        p2,
        ORIENTATION_ARROW_COLOR,
        2,
        cv2.LINE_AA,
        tipLength=0.3
    )


# ============================================================
# FIND BRIGHT BLOBS
# ============================================================

def find_bright_blobs(
    gray,
    r_min=None,
    r_max=None,
    n_keep=40
):

    r_min = (
        PUPIL_R_MIN + 2
        if r_min is None
        else r_min
    )

    r_max = (
        PUPIL_R_MAX
        if r_max is None
        else r_max
    )

    g = cv2.medianBlur(
        gray,
        9
    )

    small = cv2.resize(
        g,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA
    ).astype(
        np.float32
    )

    radii = np.geomspace(
        r_min / 2.0,
        r_max / 2.0,
        10
    )

    stack = []

    for r in radii:

        sg = (
            r
            /
            np.sqrt(2.0)
        )

        b = cv2.GaussianBlur(
            small,
            (0, 0),
            sg
        )

        stack.append(
            -sg
            *
            sg
            *
            cv2.Laplacian(
                b,
                cv2.CV_32F,
                ksize=3
            )
        )

    stack = np.array(
        stack
    )

    cands = []

    for k, r in enumerate(
        radii
    ):

        R = stack[k]

        k1 = (
            int(2 * r)
            |
            1
        )

        lm = cv2.dilate(
            R,
            np.ones(
                (k1, k1),
                np.uint8
            )
        )

        m = (
            (R >= lm)
            &
            (R > BLOB_THRESHOLD)
        )

        if k > 0:

            m &= (
                R
                >=
                stack[k - 1]
            )

        if k < len(
            radii
        ) - 1:

            m &= (
                R
                >=
                stack[k + 1]
            )

        ys, xs = np.nonzero(
            m
        )

        for y, x in zip(
            ys,
            xs
        ):

            if (
                x * 2
                <
                BORDER_MARGIN
                or
                y * 2
                <
                BORDER_MARGIN
                or
                x * 2
                >
                gray.shape[1]
                -
                BORDER_MARGIN
                or
                y * 2
                >
                gray.shape[0]
                -
                BORDER_MARGIN
            ):
                continue

            cands.append(
                (
                    float(
                        R[y, x]
                    ),
                    x * 2.0,
                    y * 2.0,
                    r * 2.0
                )
            )

    cands.sort(
        reverse=True
    )

    out = []

    for c in cands:

        if all(
            np.hypot(
                c[1] - o[1],
                c[2] - o[2]
            )
            >
            0.6
            *
            max(
                c[3],
                o[3]
            )
            for o in out
        ):

            out.append(c)

        if len(out) >= n_keep:
            break

    return out


# ============================================================
# QUICK RATIO
# ============================================================

def _quick_ratio(
    gray_blur,
    x,
    y,
    r
):

    h, w = gray_blur.shape

    x1 = max(
        0,
        int(x - 2 * r)
    )

    x2 = min(
        w,
        int(x + 2 * r) + 1
    )

    y1 = max(
        0,
        int(y - 2 * r)
    )

    y2 = min(
        h,
        int(y + 2 * r) + 1
    )

    yy, xx = np.ogrid[
        y1:y2,
        x1:x2
    ]

    d = np.hypot(
        xx - x,
        yy - y
    )

    crop = gray_blur[
        y1:y2,
        x1:x2
    ]

    ins = crop[
        d <= 0.8 * r
    ]

    ring = crop[
        (d >= 1.3 * r)
        &
        (d <= 1.8 * r)
    ]

    if (
        ins.size == 0
        or
        ring.size == 0
    ):
        return 0.0

    return float(
        ins.mean()
        /
        max(
            ring.mean(),
            1.0
        )
    )


# ============================================================
# FIND GLINT INSIDE PUPIL
# ============================================================

def find_glint_in_pupil(
    gray,
    p
):

    cx = p["cx"]
    cy = p["cy"]
    r = p["r"]

    h, w = gray.shape

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.0
    ).astype(
        np.float32
    )

    x1 = max(
        0,
        int(cx - r)
    )

    x2 = min(
        w,
        int(cx + r) + 1
    )

    y1 = max(
        0,
        int(cy - r)
    )

    y2 = min(
        h,
        int(cy + r) + 1
    )

    crop = blur[
        y1:y2,
        x1:x2
    ]

    yy, xx = np.ogrid[
        y1:y2,
        x1:x2
    ]

    inside = (
        np.hypot(
            xx - cx,
            yy - cy
        )
        <=
        0.9 * r
    )

    vals = crop[
        inside
    ]

    if vals.size == 0:
        return None

    med = float(
        np.median(
            vals
        )
    )

    peak = float(
        vals.max()
    )

    if (
        peak - med
        <
        MIN_GLINT_CONTRAST
    ):
        return None

    thr = (
        med
        +
        0.6
        *
        (
            peak - med
        )
    )

    mask = (
        (crop >= thr)
        &
        inside
    ).astype(
        np.uint8
    )

    n, lab, stats, cents = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if n < 2:
        return None

    masked = np.where(
        inside,
        crop,
        -1.0
    )

    py, px = np.unravel_index(
        int(
            np.argmax(
                masked
            )
        ),
        masked.shape
    )

    k = lab[
        py,
        px
    ]

    gx, gy = cents[k]

    return {
        "x": int(
            round(
                gx + x1
            )
        ),
        "y": int(
            round(
                gy + y1
            )
        ),
        "peak": int(
            peak
        ),
        "contrast": (
            peak - med
        )
    }


# ============================================================
# DETECT BRIGHT PUPILS
# ============================================================

def detect_bright_pupils(
    gray
):

    h, w = gray.shape

    gm = cv2.medianBlur(
        gray,
        9
    )

    gray_blur = cv2.GaussianBlur(
        gm.astype(
            np.float32
        ),
        (0, 0),
        PUPIL_BLUR_SIGMA
    )

    fits = []

    for resp, x, y, r_est in find_bright_blobs(
        gray
    ):

        if (
            len(fits)
            >=
            MAX_BLOB_CANDIDATES
        ):
            break

        if (
            _quick_ratio(
                gray_blur,
                x,
                y,
                r_est
            )
            <
            1.25
        ):
            continue

        p = fit_pupil_circle(
            gray_blur,
            (
                x,
                y
            )
        )

        if p is None:
            continue

        if (
            p["ratio"]
            <
            MIN_PUPIL_RATIO
            or
            p["quality"]
            <
            MIN_PUPIL_QUALITY
        ):
            continue

        if any(
            np.hypot(
                p["cx"] - q["cx"],
                p["cy"] - q["cy"]
            )
            <
            0.5
            *
            max(
                p["r"],
                q["r"]
            )
            for q in fits
        ):
            continue

        p["glint"] = find_glint_in_pupil(
            gray,
            p
        )

        p["score"] = (
            min(
                p["ratio"],
                6.0
            )
            *
            p["quality"]
            *
            (
                1.0
                +
                0.3
                *
                (
                    p["glint"]
                    is not None
                )
            )
        )

        fits.append(p)

    left = [
        p for p in fits
        if p["cx"] < w / 2
    ]

    right = [
        p for p in fits
        if p["cx"] >= w / 2
    ]

    pupils = {
        "left": None,
        "right": None
    }

    glints = {
        "left": None,
        "right": None
    }

    best = None

    for L in left:

        for R in right:

            sep = (
                R["cx"] - L["cx"]
            ) / float(w)

            ydf = (
                abs(
                    R["cy"] - L["cy"]
                )
                /
                float(h)
            )

            rr = (
                max(
                    L["r"],
                    R["r"]
                )
                /
                max(
                    min(
                        L["r"],
                        R["r"]
                    ),
                    1e-6
                )
            )

            if not (
                MIN_EYE_SEPARATION_FRAC
                <=
                sep
                <=
                MAX_EYE_SEPARATION_FRAC
            ):
                continue

            if (
                ydf
                >
                MAX_EYE_Y_DIFFERENCE_FRAC
                or
                rr
                >
                MAX_EYE_RADIUS_RATIO
            ):
                continue

            sc = (
                L["score"]
                +
                R["score"]
                -
                3.0
                *
                (
                    ydf
                    /
                    MAX_EYE_Y_DIFFERENCE_FRAC
                ) ** 2
            )

            if (
                best is None
                or
                sc > best[0]
            ):

                best = (
                    sc,
                    L,
                    R
                )

    if best is not None:

        chosen = {
            "left": best[1],
            "right": best[2]
        }

    else:

        chosen = {}

        for side, lst in (
            (
                "left",
                left
            ),
            (
                "right",
                right
            )
        ):

            if lst:

                q = max(
                    lst,
                    key=lambda p:
                    p["score"]
                )

                q[
                    "needs_review"
                ] = 1

                chosen[
                    side
                ] = q

    for side, p in chosen.items():

        glints[
            side
        ] = p.get(
            "glint"
        )

        pupils[
            side
        ] = {
            k: v
            for k, v in p.items()
            if k not in (
                "glint",
                "score"
            )
        }

    return (
        pupils,
        glints
    )


# ============================================================
# LEGACY GLINT DETECTION
# ============================================================

def detect_both_eye_glints(
    image_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        return (
            None,
            {},
            0.0
        )

    start = time.perf_counter()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    raw_candidates = (
        find_glint_candidates(
            gray
        )
    )

    candidates = (
        prepare_candidates(
            gray,
            raw_candidates
        )
    )

    pair = choose_eye_pair(
        gray,
        candidates
    )

    if pair is None:

        elapsed = (
            time.perf_counter()
            -
            start
        ) * 1000.0

        return (
            image,
            {
                "left": None,
                "right": None
            },
            elapsed
        )

    results = {
        "left": None,
        "right": None
    }

    left_pupil = pair[
        "left"
    ]

    right_pupil = pair[
        "right"
    ]

    if verify_pupil_context(
        gray,
        left_pupil
    ):

        results[
            "left"
        ] = select_final_glint(
            gray,
            left_pupil,
            candidates
        )

    if verify_pupil_context(
        gray,
        right_pupil
    ):

        results[
            "right"
        ] = select_final_glint(
            gray,
            right_pupil,
            candidates
        )

    elapsed = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return (
        image,
        results,
        elapsed
    )


# ============================================================
# DETECT PUPIL CIRCLES
# ============================================================

def detect_pupil_circles(
    image,
    results
):

    gray8 = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    erase = np.zeros_like(
        gray8
    )

    for side in (
        "left",
        "right"
    ):

        g = results.get(
            side
        )

        if g is not None:

            cv2.circle(
                erase,
                (
                    int(
                        g["x"]
                    ),
                    int(
                        g["y"]
                    )
                ),
                GLINT_ERASE_RADIUS,
                255,
                -1
            )

    if erase.any():

        gray8 = cv2.inpaint(
            gray8,
            erase,
            5,
            cv2.INPAINT_TELEA
        )

    gray_blur = cv2.GaussianBlur(
        gray8.astype(
            np.float32
        ),
        (0, 0),
        PUPIL_BLUR_SIGMA
    )

    pupils = {
        "left": None,
        "right": None
    }

    for side in (
        "left",
        "right"
    ):

        g = results.get(
            side
        )

        if g is not None:

            pupils[
                side
            ] = fit_pupil_circle(
                gray_blur,
                (
                    float(
                        g["x"]
                    ),
                    float(
                        g["y"]
                    )
                )
            )

            if pupils[
                side
            ] is not None:

                pupils[
                    side
                ]["needs_review"] = 1

    return pupils


# ============================================================
# DETECT EYES
# ============================================================

def detect_eyes(
    image_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        return (
            None,
            {},
            {},
            0.0,
            ""
        )

    t0 = time.perf_counter()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    pupils, glints = (
        detect_bright_pupils(
            gray
        )
    )

    method = (
        "bright-pupil"
    )

    if (
        USE_LEGACY_FALLBACK
        and
        pupils["left"] is None
        and
        pupils["right"] is None
    ):

        _, glints, _ = (
            detect_both_eye_glints(
                image_path
            )
        )

        pupils = (
            detect_pupil_circles(
                image,
                glints
            )
        )

        method = (
            "legacy-fallback"
        )

    return (
        image,
        glints,
        pupils,
        (
            time.perf_counter()
            -
            t0
        ) * 1000.0,
        method
    )


# ============================================================
# FORMAT PUPIL
# ============================================================

def _fmt_pupil(
    p
):

    if p is None:
        return "NOT FOUND"

    flag = (
        " *REVIEW"
        if p["needs_review"]
        else ""
    )

    return (
        f"({p['cx']:.0f},"
        f"{p['cy']:.0f}) "
        f"r={p['r']:.1f}"
        f"{flag}"
    )


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_and_save_folder(
    root_folder_path,
    display_visuals=False
):

    raw_contour_root = os.path.join(
        root_folder_path,
        RAW_CONTOUR_FOLDER
    )

    circle_root = os.path.join(
        root_folder_path,
        PUPIL_CIRCLE_FOLDER
    )

    mask_root = os.path.join(
        circle_root,
        MASK_FOLDER_NAME
    )

    intensity_root = os.path.join(
        root_folder_path,
        INTENSITY_PROFILE_FOLDER
    )

    os.makedirs(
        raw_contour_root,
        exist_ok=True
    )

    os.makedirs(
        circle_root,
        exist_ok=True
    )

    os.makedirs(
        intensity_root,
        exist_ok=True
    )

    extensions = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
        "*.BMP"
    )

    image_files = []

    for ext in extensions:

        pattern = os.path.join(
            root_folder_path,
            "**",
            ext
        )

        image_files.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    image_files = sorted(
        set(image_files)
    )

    # --------------------------------------------------------
    # NEVER PROCESS OUTPUT IMAGES
    # --------------------------------------------------------

    skip_names = {
        RAW_CONTOUR_FOLDER.lower(),
        PUPIL_CIRCLE_FOLDER.lower(),
        OLD_OUTPUT_FOLDER.lower()
    }

    def _is_output(path):

        parts = os.path.normpath(
            os.path.relpath(
                path,
                root_folder_path
            )
        ).lower().split(
            os.sep
        )

        return any(
            part in skip_names
            for part in parts[:-1]
        )

    image_files = [
        f
        for f in image_files
        if not _is_output(f)
    ]

    if not image_files:

        print(
            f"No valid images found in: "
            f"{root_folder_path}"
        )

        return []

    print()

    print(
        "PUPIL-FIRST DETECTION + "
        "RAW PUPIL CONTOUR + "
        "PUPIL CIRCLE + "
        "COMMON EYE ORIENTATION"
    )

    print(
        "No ML / no YOLO / no Haar cascade"
    )

    print(
        "Bright glint is ignored and "
        "profile continues to next point."
    )

    print(
        f"Neighbour averaging         : "
        f"{AVERAGE_PROFILE_PIXELS} valid pixels/group"
    )

    print(
        "Straight-line fit           : "
        "averaged points"
    )

    print(
        f"Images found: "
        f"{len(image_files)}"
    )

    print()

    print(
        f"{'Image':<40} | "
        f"{'ms':>6} | "
        f"{'Method':<15} | "
        f"{'Left pupil':<28} | "
        f"{'Right pupil'}"
    )

    print(
        "-" * 125
    )

    total_time_ms = 0.0

    both_found = 0

    n_done = 0

    n_review = 0

    total_ignored_glint_points = 0

    csv_rows = []

    intensity_csv_rows = []

    # One record per image + eye for the HR-requested
    # slope-vs-angle graph.
    slope_angle_rows = []

    # ========================================================
    # PROCESS EVERY IMAGE
    # ========================================================

    for image_path in image_files:

        (
            image,
            results,
            pupils,
            detect_ms,
            method
        ) = detect_eyes(
            image_path
        )

        if image is None:
            continue

        n_done += 1

        total_time_ms += (
            detect_ms
        )

        left = results.get(
            "left"
        )

        right = results.get(
            "right"
        )

        if (
            pupils["left"]
            is not None
            and
            pupils["right"]
            is not None
        ):

            both_found += 1

        # ====================================================
        # ORIENTATION
        # ====================================================

        gray_full = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        orientation = {
            "left": None,
            "right": None
        }

        for side in (
            "left",
            "right"
        ):

            if pupils[side] is not None:

                orientation[
                    side
                ] = compute_pupil_orientation(
                    gray_full,
                    pupils[side],
                    results.get(
                        side
                    )
                )

        # ====================================================
        # COMMON ORIENTATION
        # ====================================================

        (
            orientation,
            common_orientation_deg
        ) = apply_common_orientation_angle(
            orientation,
            pupils
        )

        # ====================================================
        # INTENSITY PROFILE
        #
        # NEW BEHAVIOUR:
        #
        #     dark point
        #          ↓
        #       points
        #          ↓
        #     BRIGHT GLINT
        #          ↓
        #       IGNORE
        #          ↓
        #     NEXT POINT
        #          ↓
        #       continue
        # ====================================================

        intensity_profiles = {
            "left": None,
            "right": None
        }

        for side in (
            "left",
            "right"
        ):

            if (
                pupils[side]
                is not None
                and
                orientation[side]
                is not None
            ):

                intensity_profiles[
                    side
                ] = extract_angle_intensity_profile(
                    gray_full,
                    pupils[side],
                    orientation[side],
                    results.get(
                        side
                    )
                )

        relative_path = os.path.relpath(
            image_path,
            root_folder_path
        )

        # ====================================================
        # SAVE INTENSITY CSV
        # ====================================================

        for side in (
            "left",
            "right"
        ):

            profile = (
                intensity_profiles[
                    side
                ]
            )

            if profile is None:
                continue

            total_ignored_glint_points += (
                profile[
                    "ignored_count"
                ]
            )

            # ------------------------------------------------
            # HR REQUEST:
            # Store ONE point for this image + eye:
            #
            #     X = angle
            #     Y = fitted slope
            #
            # The slope was already calculated in
            # extract_angle_intensity_profile().
            # ------------------------------------------------

            profile_slope = profile.get(
                "fit_slope"
            )

            if (
                profile_slope is not None
                and
                profile.get(
                    "angle_deg"
                ) is not None
            ):

                slope_angle_rows.append(
                    {
                        "image":
                            relative_path,

                        "eye":
                            side,

                        "angle_deg":
                            float(
                                profile[
                                    "angle_deg"
                                ]
                            ),

                        "slope":
                            float(
                                profile_slope
                            ),

                        "r2":
                            profile.get(
                                "fit_r2"
                            )
                    }
                )

            for (
                d,
                x,
                y,
                intensity
            ) in zip(
                profile["distances"],
                profile["xs"],
                profile["ys"],
                profile["intensities"]
            ):

                intensity_csv_rows.append(
                    {
                        "image":
                            relative_path,

                        "eye":
                            side,

                        "angle_deg":
                            f"{profile['angle_deg']:.2f}",

                        "distance_px":
                            f"{float(np.asarray(d).item()):.2f}",

                        "x":
                            f"{float(np.asarray(x).item()):.2f}",

                        "y":
                            f"{float(np.asarray(y).item()):.2f}",

                        "intensity":
                            f"{float(np.asarray(intensity).item()):.2f}",

                        # Straight-line slope of the averaged
                        # intensity profile for this eye/image.
                        "line_slope":
                            (
                                ""
                                if profile.get("fit_slope") is None
                                else f"{profile['fit_slope']:.6f}"
                            )
                    }
                )

        # ====================================================
        # STEP 1:
        # RAW CONTOUR
        # ====================================================

        contour_img = image.copy()

        for p in pupils.values():

            draw_raw_contour(
                contour_img,
                p
            )

        for side in (
            "left",
            "right"
        ):

            if (
                orientation[side]
                is not None
            ):

                draw_orientation_arrow(
                    contour_img,
                    orientation[side]
                )

        if left is not None:

            draw_glint(
                contour_img,
                left
            )

        if right is not None:

            draw_glint(
                contour_img,
                right
            )

        contour_path = os.path.join(
            raw_contour_root,
            relative_path
        )

        os.makedirs(
            os.path.dirname(
                contour_path
            ),
            exist_ok=True
        )

        cv2.imwrite(
            contour_path,
            contour_img
        )

        # ====================================================
        # STEP 2:
        # PUPIL CIRCLE
        # ====================================================

        circle_img = image.copy()

        if DRAW_PUPIL_CIRCLE:

            if FILL_PUPIL_ON_OVERLAY:

                tint = (
                    circle_img.copy()
                )

                for p in pupils.values():

                    if p is not None:

                        _draw_circle(
                            tint,
                            p["cx"],
                            p["cy"],
                            p["r"],
                            (0, 0, 255),
                            -1
                        )

                circle_img = (
                    cv2.addWeighted(
                        tint,
                        0.25,
                        circle_img,
                        0.75,
                        0
                    )
                )

            for p in pupils.values():

                if p is not None:

                    color = (
                        CIRCLE_REVIEW_COLOR
                        if p["needs_review"]
                        else CIRCLE_COLOR
                    )

                    _draw_circle(
                        circle_img,
                        p["cx"],
                        p["cy"],
                        p["r"],
                        color,
                        1
                    )

        if DRAW_ORIENTATION_ARROW:

            for side in (
                "left",
                "right"
            ):

                draw_orientation_arrow(
                    circle_img,
                    orientation[side]
                )

        if left is not None:

            draw_glint(
                circle_img,
                left
            )

        if right is not None:

            draw_glint(
                circle_img,
                right
            )

        circle_path = os.path.join(
            circle_root,
            relative_path
        )

        os.makedirs(
            os.path.dirname(
                circle_path
            ),
            exist_ok=True
        )

        cv2.imwrite(
            circle_path,
            circle_img
        )

        # ====================================================
        # SAVE INTENSITY GRAPH
        # ====================================================

        if SAVE_INTENSITY_PROFILE_GRAPHS:

            graph_stem = os.path.splitext(
                relative_path
            )[0]

            for side in (
                "left",
                "right"
            ):

                profile = (
                    intensity_profiles[
                        side
                    ]
                )

                if profile is not None:

                    graph_path = os.path.join(
                        intensity_root,
                        graph_stem
                        +
                        f"_{side}_intensity.png"
                    )

                    os.makedirs(
                        os.path.dirname(
                            graph_path
                        ),
                        exist_ok=True
                    )

                    save_angle_intensity_graph(
                        profile,
                        graph_path,
                        relative_path,
                        side
                    )

        # ====================================================
        # PUPIL MASK
        # ====================================================

        if SAVE_MASKS:

            stem = os.path.splitext(
                relative_path
            )[0]

            mask_path = os.path.join(
                mask_root,
                stem
                +
                "_mask.png"
            )

            os.makedirs(
                os.path.dirname(
                    mask_path
                ),
                exist_ok=True
            )

            cv2.imwrite(
                mask_path,
                make_pupil_mask(
                    image.shape,
                    [
                        pupils["left"],
                        pupils["right"]
                    ]
                )
            )

        # ====================================================
        # CSV ROW
        # ====================================================

        row = {
            "image": relative_path,
            "method": method
        }

        for side in (
            "left",
            "right"
        ):

            g = results.get(
                side
            )

            p = pupils[
                side
            ]

            row[
                f"{side}_glint_x"
            ] = (
                ""
                if g is None
                else g["x"]
            )

            row[
                f"{side}_glint_y"
            ] = (
                ""
                if g is None
                else g["y"]
            )

            row[
                f"{side}_pupil_cx"
            ] = (
                ""
                if p is None
                else f"{p['cx']:.2f}"
            )

            row[
                f"{side}_pupil_cy"
            ] = (
                ""
                if p is None
                else f"{p['cy']:.2f}"
            )

            row[
                f"{side}_pupil_r"
            ] = (
                ""
                if p is None
                else f"{p['r']:.2f}"
            )

            row[
                f"{side}_quality"
            ] = (
                ""
                if p is None
                else f"{p['quality']:.2f}"
            )

            row[
                f"{side}_contour_points"
            ] = (
                ""
                if (
                    p is None
                    or
                    p.get(
                        "contour"
                    ) is None
                )
                else len(
                    p["contour"]
                )
            )

            o = orientation[
                side
            ]

            row[
                f"{side}_orientation_deg"
            ] = (
                ""
                if (
                    o is None
                    or
                    o.get(
                        "angle_deg"
                    ) is None
                )
                else f"{o['angle_deg']:.1f}"
            )

            row[
                f"{side}_orientation_strength"
            ] = (
                ""
                if o is None
                else f"{o['strength']:.3f}"
            )

            row[
                f"{side}_orientation_ok"
            ] = (
                ""
                if o is None
                else int(
                    o["ok"]
                )
            )

            row[
                f"{side}_needs_review"
            ] = (
                1
                if (
                    p is None
                    or
                    p["needs_review"]
                )
                else 0
            )

            profile = (
                intensity_profiles[
                    side
                ]
            )

            row[
                f"{side}_profile_angle_deg"
            ] = (
                ""
                if profile is None
                else f"{profile['angle_deg']:.2f}"
            )

            row[
                f"{side}_dark_intensity"
            ] = (
                ""
                if profile is None
                else f"{profile['dark_intensity']:.2f}"
            )

            row[
                f"{side}_bright_intensity"
            ] = (
                ""
                if profile is None
                else f"{profile['bright_intensity']:.2f}"
            )

            row[
                f"{side}_profile_samples"
            ] = (
                ""
                if profile is None
                else len(
                    profile[
                        "intensities"
                    ]
                )
            )

            # ------------------------------------------------
            # NEW CSV INFORMATION
            # ------------------------------------------------

            row[
                f"{side}_glint_points_ignored"
            ] = (
                ""
                if profile is None
                else profile[
                    "ignored_count"
                ]
            )

            row[
                f"{side}_average_group_pixels"
            ] = (
                ""
                if profile is None
                else AVERAGE_PROFILE_PIXELS
            )

            row[
                f"{side}_line_slope"
            ] = (
                ""
                if profile is None
                or profile.get("fit_slope") is None
                else f"{profile['fit_slope']:.6f}"
            )

            row[
                f"{side}_line_intercept"
            ] = (
                ""
                if profile is None
                or profile.get("fit_intercept") is None
                else f"{profile['fit_intercept']:.6f}"
            )

            row[
                f"{side}_line_r2"
            ] = (
                ""
                if profile is None
                or profile.get("fit_r2") is None
                else f"{profile['fit_r2']:.6f}"
            )

        # ====================================================
        # COMMON ANGLE
        # ====================================================

        row[
            "common_orientation_deg"
        ] = (
            ""
            if common_orientation_deg is None
            else f"{common_orientation_deg:.1f}"
        )

        csv_rows.append(
            row
        )

        if any(
            row[
                f"{s}_needs_review"
            ]
            for s in (
                "left",
                "right"
            )
        ):

            n_review += 1

        print(
            f"{relative_path[-40:]:<40} | "
            f"{detect_ms:>6.0f} | "
            f"{method:<15} | "
            f"{_fmt_pupil(pupils['left']):<28} | "
            f"{_fmt_pupil(pupils['right'])}"
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        if display_visuals:

            cv2.imshow(
                "Raw contour (left) | "
                "Pupil circle (right)",
                np.hstack(
                    [
                        contour_img,
                        circle_img
                    ]
                )
            )

            key = cv2.waitKey(
                0
            )

            if key == 27:

                display_visuals = False

                cv2.destroyAllWindows()

    # ========================================================
    # CLOSE WINDOWS
    # ========================================================

    if display_visuals:

        cv2.destroyAllWindows()

    # ========================================================
    # SAVE INTENSITY CSV
    # ========================================================

    if intensity_csv_rows:

        intensity_csv_path = os.path.join(
            intensity_root,
            INTENSITY_PROFILE_CSV
        )

        with open(
            intensity_csv_path,
            "w",
            newline=""
        ) as f:

            fields = [
                "image",
                "eye",
                "angle_deg",
                "distance_px",
                "x",
                "y",
                "intensity",
                "line_slope"
            ]

            iw = csv.DictWriter(
                f,
                fieldnames=fields
            )

            iw.writeheader()

            iw.writerows(
                intensity_csv_rows
            )

    # ========================================================
    # SAVE SLOPE vs ANGLE GRAPH
    # ========================================================

    if (
        SAVE_SLOPE_ANGLE_GRAPH
        and
        slope_angle_rows
    ):

        slope_angle_root = os.path.join(
            root_folder_path,
            SLOPE_ANGLE_GRAPH_FOLDER
        )

        os.makedirs(
            slope_angle_root,
            exist_ok=True
        )

        slope_angle_graph_path = os.path.join(
            slope_angle_root,
            SLOPE_ANGLE_GRAPH_FILE
        )

        slope_angle_csv_path = os.path.join(
            slope_angle_root,
            SLOPE_ANGLE_CSV_FILE
        )

        save_slope_vs_angle_graph(
            slope_angle_rows,
            slope_angle_graph_path,
            slope_angle_csv_path
        )

    # ========================================================
    # SAVE MAIN CSV
    # ========================================================

    if csv_rows:

        csv_path = os.path.join(
            circle_root,
            CSV_NAME
        )

        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=list(
                    csv_rows[0].keys()
                )
            )

            w.writeheader()

            w.writerows(
                csv_rows
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    n = max(
        n_done,
        1
    )

    print(
        "-" * 125
    )

    print(
        f"Both pupils found          : "
        f"{both_found}/{n_done}"
    )

    print(
        f"Images flagged for review  : "
        f"{n_review}/{n_done}"
    )

    print(
        f"Average processing time    : "
        f"{total_time_ms / n:.0f} ms/image"
    )

    print(
        f"Total glint points ignored : "
        f"{total_ignored_glint_points}"
    )

    print()

    print(
        "Glint behaviour             : "
        "IGNORE → CONTINUE → NEXT POINT"
    )

    print(
        f"Profile averaging           : "
        f"{AVERAGE_PROFILE_PIXELS} valid pixels/group"
    )

    print(
        "Line fitting                : "
        "linear fit on averaged points"
    )

    print()

    print(
        f"Raw contour images          : "
        f"{raw_contour_root}"
    )

    print(
        f"Pupil circle images         : "
        f"{circle_root}"
    )

    print(
        f"Pupil masks                 : "
        f"{mask_root}"
    )

    print(
        f"CSV                         : "
        f"{os.path.join(circle_root, CSV_NAME)}"
    )

    print(
        f"Angle intensity graphs      : "
        f"{intensity_root}"
    )

    print(
        f"Angle intensity CSV         : "
        f"{os.path.join(intensity_root, INTENSITY_PROFILE_CSV)}"
    )

    print(
        f"Slope vs angle graph        : "
        f"{os.path.join(root_folder_path, SLOPE_ANGLE_GRAPH_FOLDER, SLOPE_ANGLE_GRAPH_FILE)}"
    )

    print(
        f"Slope vs angle CSV          : "
        f"{os.path.join(root_folder_path, SLOPE_ANGLE_GRAPH_FOLDER, SLOPE_ANGLE_CSV_FILE)}"
    )

    # Return the per-image/per-eye slope-angle records so the
    # three-input comparison can be plotted after all inputs run.
    return slope_angle_rows


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("MULTI-INPUT SLOPE vs ANGLE PROCESSING")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # USE ALL PATHS ENTERED ABOVE
    # --------------------------------------------------------
    # No command-line input and no runtime path prompt.
    # Edit INPUT_PATHS at the top of this file when you want
    # to process a different set of folders.

    if len(INPUT_PATHS) < 2:
        print(
            "ERROR: INPUT_PATHS must contain at least 2 folder paths."
        )
        sys.exit(1)

    input_folders = [
        os.path.abspath(os.path.expanduser(path.strip().strip('"')))
        for path in INPUT_PATHS
    ]

    # Validate all input paths before processing.
    for index, folder in enumerate(input_folders, start=1):

        if not os.path.isdir(folder):
            print()
            print(f"ERROR: Input path {index} does not exist:")
            print(folder)
            print()
            print("Edit INPUT_PATHS near the top of the code.")
            sys.exit(1)

    print("Input folders:")

    for i, folder in enumerate(input_folders, start=1):
        print(
            f"Input {i}: {folder}"
        )

    # Collect slope-angle records from all inputs.
    all_slope_angle_rows = []

    for i, folder in enumerate(input_folders, start=1):

        folder_label = os.path.basename(
            os.path.normpath(folder)
        )

        if not folder_label:
            folder_label = f"Input {i}"

        print()
        print("=" * 80)
        print(
            f"PROCESSING INPUT {i}: {folder_label}"
        )
        print("=" * 80)

        rows = process_and_save_folder(
            folder,
            display_visuals=DISPLAY_VISUALS
        )

        for row in rows:

            new_row = dict(row)
            new_row["input_id"] = str(i)
            new_row["input_label"] = folder_label
            all_slope_angle_rows.append(new_row)

    # --------------------------------------------------------
    # CREATE TWO SEPARATE COMPARISON GRAPHS:
    #   1. Left eye  -> all configured inputs
    #   2. Right eye -> all configured inputs
    # --------------------------------------------------------

    script_folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    multi_output_root = os.path.join(
        script_folder,
        MULTI_INPUT_OUTPUT_FOLDER
    )

    print()
    print("=" * 80)
    print("CREATING MULTI-INPUT COMPARISON GRAPHS")
    print("=" * 80)

    save_multi_input_slope_vs_angle_graphs(
        all_slope_angle_rows,
        multi_output_root
    )

    print()
    print("Completed.")
    print(
        f"Comparison output folder: {multi_output_root}"
    )
