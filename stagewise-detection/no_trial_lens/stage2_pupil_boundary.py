import os
import csv
import glob
import cv2
import numpy as np
import math


# ============================================================
# STAGE 2
# RAW PUPIL CONTOUR
#
# INPUT:
#     stage1_glint/
#
# OUTPUT:
#     stage2_raw_contour/
#         images/
#         pupil_raw_contour_points.csv
#
# IMPORTANT:
#     This file has ZERO dependency on stage3_angle.py
#
#     Uses the SAME pupil fitting logic from the working code.
#
# ============================================================


# ============================================================
# ROOT FOLDER
# ============================================================

ROOT_HR_FOLDER = (
       )


# ============================================================
# STAGE 1 / STAGE 2 FOLDER NAMES
# ============================================================

STAGE1_FOLDER_NAME = "raw_contour_results1"

STAGE2_FOLDER_NAME = "stage2_raw_contour"

STAGE2_IMAGE_FOLDER = "images"

STAGE2_CSV_NAME = "pupil_raw_contour_points.csv"


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)


# ============================================================
# STAGE 1 GREEN GLINT PARAMETERS
#
# Stage 1 output contains green glint markers.
# ============================================================

GREEN_LOWER = np.array(
    [35, 60, 60],
    dtype=np.uint8
)

GREEN_UPPER = np.array(
    [95, 255, 255],
    dtype=np.uint8
)

MIN_GREEN_AREA = 2

MAX_GREEN_AREA = 300


# ============================================================
# PUPIL PARAMETERS
#
# THESE ARE FROM YOUR WORKING CODE
# ============================================================

PUPIL_R_MIN = 6

PUPIL_R_MAX = 60

PUPIL_SEARCH = 24

PUPIL_BLUR_SIGMA = 2.0

PUPIL_REVIEW_R_BELOW = 10

PUPIL_MIN_BRIGHT_RATIO = 1.3


# ============================================================
# PUPIL FIT PARAMETERS
#
# THESE ARE FROM YOUR WORKING CODE
# ============================================================

_N_ANG = 64

_ANG = np.linspace(
    0,
    2 * np.pi,
    _N_ANG,
    endpoint=False
).astype(
    np.float32
)

_COS = np.cos(_ANG)

_SIN = np.sin(_ANG)

_EDGE_DELTA = 2.5

_SIZE_BONUS = 0.002


# ============================================================
# RAW CONTOUR DRAWING
# ============================================================

PUPIL_RAW_CONTOUR_THICKNESS = 1

CONTOUR_COLOR = (
    0,
    255,
    255
)

GLINT_COLOR = (
    0,
    255,
    0
)

GLINT_RADIUS = 3


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_VISUALS = False


# ============================================================
# IMAGE CHECK
# ============================================================

def is_image_file(path):

    if not os.path.isfile(path):
        return False

    return (
        os.path.splitext(path)[1].lower()
        in IMAGE_EXTENSIONS
    )


# ============================================================
# FIND STAGE 1 FOLDERS RECURSIVELY
# ============================================================

def find_stage1_folders(root_folder):

    folders = []

    for current_root, dirs, files in os.walk(
        root_folder
    ):

        # Do not enter generated stage folders.

        dirs[:] = [
            d for d in dirs
            if d.lower() not in (
                "stage2_raw_contour",
                "stage3_angle",
                "stage4_intensity_profile",
                "stage5_average_fit",
                "stage6_slope_detection"
            )
        ]

        if (
            os.path.basename(
                current_root
            ).lower()
            ==
            STAGE1_FOLDER_NAME.lower()
        ):

            folders.append(
                current_root
            )

    return sorted(
        set(folders)
    )


# ============================================================
# FIND IMAGES INSIDE STAGE 1 FOLDER
# ============================================================

def find_images(folder):

    image_files = []

    for current_root, dirs, files in os.walk(
        folder
    ):

        for filename in files:

            path = os.path.join(
                current_root,
                filename
            )

            if is_image_file(path):

                image_files.append(path)

    return sorted(
        set(image_files)
    )


# ============================================================
# DETECT GREEN MARKERS FROM STAGE 1
# ============================================================

def detect_stage1_green_glints(
    image
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        GREEN_LOWER,
        GREEN_UPPER
    )

    # Small cleanup.

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    glints = []

    for i in range(
        1,
        n
    ):

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA
            ]
        )

        if area < MIN_GREEN_AREA:
            continue

        if area > MAX_GREEN_AREA:
            continue

        cx = float(
            centroids[i][0]
        )

        cy = float(
            centroids[i][1]
        )

        glints.append(
            {
                "x": cx,
                "y": cy,
                "area": area
            }
        )

    return glints


# ============================================================
# SELECT LEFT / RIGHT GLINT
# ============================================================

def choose_left_right_glints(
    glints,
    image_width
):

    left_candidates = []

    right_candidates = []

    center_x = image_width / 2.0

    for g in glints:

        if g["x"] < center_x:

            left_candidates.append(g)

        else:

            right_candidates.append(g)

    left = None

    right = None

    # --------------------------------------------------------
    # If several green markers exist on one side,
    # choose the one closest to the expected eye center.
    # --------------------------------------------------------

    if left_candidates:

        left = min(
            left_candidates,
            key=lambda g:
            abs(
                g["x"]
                -
                image_width * 0.25
            )
        )

    if right_candidates:

        right = min(
            right_candidates,
            key=lambda g:
            abs(
                g["x"]
                -
                image_width * 0.75
            )
        )

    return {
        "left": left,
        "right": right
    }


# ============================================================
# SAMPLE IMAGE
#
# SAME AS WORKING CODE
# ============================================================

def _sample(
    gray,
    xs,
    ys
):

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
#
# SAME AS WORKING CODE
# ============================================================

def _contrast_scale(
    gray,
    seed
):

    sx, sy = seed

    h, w = gray.shape

    x1 = max(
        0,
        int(
            sx - 76
        )
    )

    x2 = min(
        w,
        int(
            sx + 77
        )
    )

    y1 = max(
        0,
        int(
            sy - 76
        )
    )

    y2 = min(
        h,
        int(
            sy + 77
        )
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

    if inner.size == 0:

        return 8.0

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
#
# SAME AS WORKING CODE
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
        (
            inner
            -
            outer
        )
        /
        tau,

        -0.15,
        1.0
    )

    return (
        d.mean(axis=1)
        +
        _SIZE_BONUS * radii
    )


# ============================================================
# COARSE PUPIL FIT
#
# SAME AS WORKING CODE
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

            scores = _perimeter_score(
                gray,
                cx,
                cy,
                radii,
                tau
            )

            k = int(
                np.argmax(
                    scores
                )
            )

            if scores[k] > best_score:

                best_score = float(
                    scores[k]
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

    current_score = score(p)

    # --------------------------------------------------------
    # Local refinement
    # --------------------------------------------------------

    for step in (
        2.0,
        1.0,
        0.5,
        0.25
    ):

        for _ in range(40):

            improved = False

            for dim in range(3):

                for sign in (
                    1.0,
                    -1.0
                ):

                    q = p.copy()

                    q[dim] += (
                        sign
                        *
                        step
                    )

                    s = score(q)

                    if (
                        s
                        >
                        current_score
                        +
                        1e-6
                    ):

                        p = q

                        current_score = s

                        improved = True

            if not improved:

                break

    return (
        p,
        current_score
    )


# ============================================================
# LEAST-SQUARE CIRCLE
#
# SAME AS WORKING CODE
# ============================================================

def _fit_circle_lsq(
    pts
):

    A = np.c_[
        2 * pts[:, 0],
        2 * pts[:, 1],
        np.ones(
            len(pts)
        )
    ]

    y = (
        pts ** 2
    ).sum(
        axis=1
    )

    cx, cy, k = (
        np.linalg.lstsq(
            A,
            y,
            rcond=None
        )[0]
    )

    radius = np.sqrt(
        max(
            k
            +
            cx * cx
            +
            cy * cy,
            1e-6
        )
    )

    return np.array(
        [
            cx,
            cy,
            radius
        ]
    )


# ============================================================
# REFINE HALF MAX
#
# THIS IS THE IMPORTANT WORKING METHOD.
#
# It does NOT use np.gradient().
# Therefore the previous gradient-size error cannot occur here.
#
# SAME AS YOUR WORKING CODE.
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

    angles = np.linspace(
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
        np.cos(
            angles
        )[:, None]
    )

    ys = (
        cy
        +
        ts[None, :]
        *
        np.sin(
            angles
        )[:, None]
    )

    profiles = _sample(
        gray,
        xs,
        ys
    )

    inside_selection = (
        (ts >= 0.4 * r)
        &
        (ts <= 0.8 * r)
    )

    outside_selection = (
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

    points = []

    # ========================================================
    # CHECK EVERY ANGULAR RAY
    # ========================================================

    for k in range(
        n_rays
    ):

        profile = profiles[k]

        inner_median = np.median(
            profile[
                inside_selection
            ]
        )

        outer_median = np.median(
            profile[
                outside_selection
            ]
        )

        # ----------------------------------------------------
        # Need dark pupil / brighter outer region transition
        # ----------------------------------------------------

        if (
            inner_median
            -
            outer_median
            <
            max(
                4.0,
                0.12 * contrast
            )
        ):

            continue

        middle_value = (
            0.5
            *
            (
                inner_median
                +
                outer_median
            )
        )

        below = (
            profile
            <
            middle_value
        )

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

        if (
            hit is None
            or
            hit == 0
        ):

            continue

        # ----------------------------------------------------
        # Interpolate boundary position
        # ----------------------------------------------------

        denominator = (
            profile[hit - 1]
            -
            profile[hit]
            +
            1e-9
        )

        t = (
            ts[hit - 1]
            +
            (
                profile[hit - 1]
                -
                middle_value
            )
            /
            denominator
            *
            (
                ts[hit]
                -
                ts[hit - 1]
            )
        )

        px = (
            cx
            +
            t
            *
            np.cos(
                angles[k]
            )
        )

        py = (
            cy
            +
            t
            *
            np.sin(
                angles[k]
            )
        )

        points.append(
            (
                px,
                py
            )
        )

    # ========================================================
    # NEED ENOUGH RAW CONTOUR POINTS
    # ========================================================

    points = np.array(
        points
    )

    if len(points) < (
        0.35
        *
        n_rays
    ):

        return None

    # ========================================================
    # OUTLIER REJECTION
    # ========================================================

    keep = np.ones(
        len(points),
        dtype=bool
    )

    for _ in range(10):

        if np.count_nonzero(
            keep
        ) < 3:

            return None

        circle = _fit_circle_lsq(
            points[
                keep
            ]
        )

        residuals = np.abs(
            np.hypot(
                points[:, 0]
                -
                circle[0],

                points[:, 1]
                -
                circle[1]
            )
            -
            circle[2]
        )

        median_residual = np.median(
            residuals[
                keep
            ]
        )

        new_keep = (
            residuals
            <
            max(
                1.0,
                2.0
                *
                median_residual
            )
        )

        if (
            new_keep.sum()
            <
            0.3
            *
            n_rays
        ):

            break

        if (
            new_keep
            ==
            keep
        ).all():

            break

        keep = new_keep

    if np.count_nonzero(
        keep
    ) < 3:

        return None

    circle = _fit_circle_lsq(
        points[
            keep
        ]
    )

    return (
        circle,
        float(
            keep.sum()
        )
        /
        n_rays,
        points,
        keep
    )


# ============================================================
# FIT PUPIL CIRCLE
#
# THIS IS THE SAME MAIN PUPIL-FIT LOGIC
# FROM YOUR WORKING CODE.
# ============================================================

def fit_pupil_circle(
    gray_blur,
    seed
):

    contrast = _contrast_scale(
        gray_blur,
        seed
    )

    tau = (
        0.3
        *
        contrast
    )

    # ========================================================
    # COARSE FIT
    # ========================================================

    coarse = _coarse_fit(
        gray_blur,
        seed,
        tau
    )

    if coarse is None:

        return None

    circle = coarse[0]

    method = "coarse"

    quality = float(
        np.clip(
            coarse[1],
            0.0,
            1.0
        )
    )

    contour_points = None

    contour_keep = None

    # ========================================================
    # HALF-MAX REFINEMENT
    # ========================================================

    for _ in range(4):

        refined = _refine_half_max(
            gray_blur,
            circle,
            contrast
        )

        if refined is None:

            break

        new_circle = refined[0]

        fraction = refined[1]

        points = refined[2]

        keep = refined[3]

        shift = np.hypot(
            new_circle[0]
            -
            circle[0],

            new_circle[1]
            -
            circle[1]
        )

        radius_ratio = (
            new_circle[2]
            /
            max(
                circle[2],
                1e-6
            )
        )

        if (
            shift
            >
            0.5 * circle[2]
            or
            not (
                0.7
                <
                radius_ratio
                <
                1.5
            )
        ):

            break

        moved = (
            shift
            +
            abs(
                new_circle[2]
                -
                circle[2]
            )
        )

        circle = new_circle

        quality = fraction

        method = "half_max"

        contour_points = points

        contour_keep = keep

        if moved < 0.15:

            break

    # ========================================================
    # FINAL CIRCLE VALUES
    # ========================================================

    cx = float(
        circle[0]
    )

    cy = float(
        circle[1]
    )

    r = float(
        circle[2]
    )

    h, w = gray_blur.shape

    # ========================================================
    # PUPIL / OUTSIDE RATIO
    # ========================================================

    x1 = max(
        0,
        int(
            cx - 2 * r
        )
    )

    x2 = min(
        w,
        int(
            cx + 2 * r
        )
        + 1
    )

    y1 = max(
        0,
        int(
            cy - 2 * r
        )
    )

    y2 = min(
        h,
        int(
            cy + 2 * r
        )
        + 1
    )

    yy, xx = np.ogrid[
        y1:y2,
        x1:x2
    ]

    distance = np.hypot(
        xx - cx,
        yy - cy
    )

    crop = gray_blur[
        y1:y2,
        x1:x2
    ]

    inside = crop[
        distance
        <=
        0.8 * r
    ]

    ring = crop[
        (distance >= 1.25 * r)
        &
        (distance <= 1.7 * r)
    ]

    if (
        inside.size
        and
        ring.size
    ):

        ratio = float(
            inside.mean()
            /
            max(
                ring.mean(),
                1.0
            )
        )

    else:

        ratio = 0.0

    # ========================================================
    # REVIEW FLAG
    # ========================================================

    needs_review = (

        method != "half_max"

        or

        quality < 0.6

        or

        r < PUPIL_REVIEW_R_BELOW

        or

        ratio
        <
        PUPIL_MIN_BRIGHT_RATIO

        or

        r
        <=
        PUPIL_R_MIN + 1.5

        or

        r
        >=
        PUPIL_R_MAX - 1.5

        or

        np.hypot(
            cx - seed[0],
            cy - seed[1]
        )
        >
        r

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
        "cx":
            cx,

        "cy":
            cy,

        "r":
            r,

        "quality":
            float(
                quality
            ),

        "ratio":
            ratio,

        "needs_review":
            int(
                bool(
                    needs_review
                )
            ),

        "contour":
            contour_points,

        "contour_keep":
            contour_keep
    }


# ============================================================
# DRAW RAW CONTOUR
#
# IMPORTANT:
# Draw the RAW contour points directly.
# Do NOT draw the fitted circle.
# ============================================================

def draw_raw_contour(
    image,
    pupil,
    thickness=1
):

    if pupil is None:

        return

    points = pupil.get(
        "contour"
    )

    if (
        points is None
        or
        len(points) < 3
    ):

        return

    points = np.asarray(
        points,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Keep all raw contour points in angular order.
    # --------------------------------------------------------

    cx = float(
        pupil["cx"]
    )

    cy = float(
        pupil["cy"]
    )

    angles = np.arctan2(
        points[:, 1] - cy,
        points[:, 0] - cx
    )

    order = np.argsort(
        angles
    )

    points = points[
        order
    ]

    fixed = np.round(
        points
    ).astype(
        np.int32
    ).reshape(
        -1,
        1,
        2
    )

    if len(fixed) < 3:

        return

    cv2.polylines(
        image,
        [
            fixed
        ],
        True,
        CONTOUR_COLOR,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW GLINT
# ============================================================

def draw_glint(
    image,
    glint
):

    if glint is None:

        return

    x = int(
        round(
            glint["x"]
        )
    )

    y = int(
        round(
            glint["y"]
        )
    )

    cv2.circle(
        image,
        (
            x,
            y
        ),
        GLINT_RADIUS,
        GLINT_COLOR,
        -1,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one_image(
    image_path,
    output_image_path,
    contour_rows
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "    ERROR: Cannot read image."
        )

        return 0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # GET GREEN GLINTS FROM STAGE 1 OUTPUT
    # ========================================================

    glints = detect_stage1_green_glints(
        image
    )

    h, w = gray.shape

    selected_glints = (
        choose_left_right_glints(
            glints,
            w
        )
    )

    # ========================================================
    # REMOVE GLINT FROM PUPIL-FITTING IMAGE
    #
    # This follows the logic of the working code:
    # the glint should not disturb pupil-boundary fitting.
    # ========================================================

    gray_for_pupil = gray.copy()

    erase = np.zeros_like(
        gray_for_pupil
    )

    for side in (
        "left",
        "right"
    ):

        glint = selected_glints[
            side
        ]

        if glint is None:

            continue

        cv2.circle(
            erase,
            (
                int(
                    round(
                        glint["x"]
                    )
                ),
                int(
                    round(
                        glint["y"]
                    )
                )
            ),
            9,
            255,
            -1
        )

    if erase.any():

        gray_for_pupil = cv2.inpaint(
            gray_for_pupil,
            erase,
            5,
            cv2.INPAINT_TELEA
        )

    # ========================================================
    # BLUR
    # ========================================================

    gray_blur = cv2.GaussianBlur(
        gray_for_pupil.astype(
            np.float32
        ),
        (0, 0),
        PUPIL_BLUR_SIGMA
    )

    # ========================================================
    # PUPIL DETECTION
    # ========================================================

    pupils = {
        "left": None,
        "right": None
    }

    for side in (
        "left",
        "right"
    ):

        glint = selected_glints[
            side
        ]

        if glint is None:

            continue

        seed = (
            float(
                glint["x"]
            ),
            float(
                glint["y"]
            )
        )

        pupil = fit_pupil_circle(
            gray_blur,
            seed
        )

        pupils[
            side
        ] = pupil

    # ========================================================
    # CREATE OUTPUT
    #
    # Original image
    # + raw pupil contour
    # + Stage 1 glint marker
    #
    # NO circle
    # NO angle
    # NO arrow
    # NO intensity
    # NO slope
    # ========================================================

    output = image.copy()

    detected_count = 0

    image_name = os.path.basename(
        image_path
    )

    for side in (
        "left",
        "right"
    ):

        pupil = pupils[
            side
        ]

        glint = selected_glints[
            side
        ]

        # ----------------------------------------------------
        # Raw contour
        # ----------------------------------------------------

        if pupil is not None:

            detected_count += 1

            draw_raw_contour(
                output,
                pupil,
                PUPIL_RAW_CONTOUR_THICKNESS
            )

        # ----------------------------------------------------
        # Glint
        # ----------------------------------------------------

        if glint is not None:

            draw_glint(
                output,
                glint
            )

        # ----------------------------------------------------
        # Save contour CSV
        # ----------------------------------------------------

        if (
            pupil is not None
            and
            pupil.get(
                "contour"
            ) is not None
        ):

            points = np.asarray(
                pupil["contour"],
                dtype=np.float64
            )

            keep = pupil.get(
                "contour_keep"
            )

            for index, point in enumerate(
                points
            ):

                if keep is None:

                    inlier = ""

                else:

                    inlier = int(
                        bool(
                            keep[index]
                        )
                    )

                contour_rows.append(
                    {
                        "image":
                            image_name,

                        "eye":
                            side,

                        "point_index":
                            index,

                        "x":
                            f"{float(point[0]):.2f}",

                        "y":
                            f"{float(point[1]):.2f}",

                        "inlier":
                            inlier
                    }
                )

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    if detected_count > 0:

        os.makedirs(
            os.path.dirname(
                output_image_path
            ),
            exist_ok=True
        )

        cv2.imwrite(
            output_image_path,
            output
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    if DISPLAY_VISUALS:

        cv2.imshow(
            "Stage 2 - Raw Pupil Contour",
            output
        )

        key = cv2.waitKey(
            1
        )

        if key == 27:

            cv2.destroyAllWindows()

            return -1

    return detected_count


# ============================================================
# PROCESS ONE STAGE 1 FOLDER
# ============================================================

def process_stage1_folder(
    stage1_folder
):

    print()
    print(
        "=" * 80
    )

    print(
        "STAGE 1 INPUT:"
    )

    print(
        stage1_folder
    )

    print(
        "=" * 80
    )

    # ========================================================
    # FIND IMAGES
    # ========================================================

    image_files = find_images(
        stage1_folder
    )

    if not image_files:

        print(
            "No images found."
        )

        return (
            0,
            0
        )

    print(
        f"Images found: {len(image_files)}"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    dataset_parent = os.path.dirname(
        stage1_folder
    )

    stage2_root = os.path.join(
        dataset_parent,
        STAGE2_FOLDER_NAME
    )

    stage2_images_root = os.path.join(
        stage2_root,
        STAGE2_IMAGE_FOLDER
    )

    csv_path = os.path.join(
        stage2_root,
        STAGE2_CSV_NAME
    )

    os.makedirs(
        stage2_images_root,
        exist_ok=True
    )

    # ========================================================
    # CSV DATA
    # ========================================================

    contour_rows = []

    total_detected = 0

    processed = 0

    # ========================================================
    # PROCESS ALL IMAGES
    # ========================================================

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        filename = os.path.basename(
            image_path
        )

        print(
            f"[{index}/{len(image_files)}] "
            f"{filename}"
        )

        # ----------------------------------------------------
        # Preserve nested folder structure.
        # ----------------------------------------------------

        relative_path = os.path.relpath(
            image_path,
            stage1_folder
        )

        output_path = os.path.join(
            stage2_images_root,
            relative_path
        )

        result = process_one_image(
            image_path,
            output_path,
            contour_rows
        )

        if result == -1:

            cv2.destroyAllWindows()

            return (
                processed,
                total_detected
            )

        processed += 1

        total_detected += result

        if result == 0:

            print(
                "    -> Pupil not detected"
            )

        elif result == 1:

            print(
                "    -> 1 pupil detected"
            )

        else:

            print(
                f"    -> {result} pupils detected"
            )

    # ========================================================
    # SAVE CONTOUR CSV
    # ========================================================

    os.makedirs(
        stage2_root,
        exist_ok=True
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "eye",
                "point_index",
                "x",
                "y",
                "inlier"
            ]
        )

        writer.writeheader()

        writer.writerows(
            contour_rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "Stage 2 output:"
    )

    print(
        stage2_root
    )

    print()

    print(
        "Raw contour images:"
    )

    print(
        stage2_images_root
    )

    print()

    print(
        "Contour CSV:"
    )

    print(
        csv_path
    )

    print()

    print(
        f"Contour points saved: "
        f"{len(contour_rows)}"
    )

    return (
        processed,
        total_detected
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "STAGE 2 - RAW PUPIL CONTOUR"
    )

    print(
        "=" * 80
    )

    print(
        "Input  : stage1_glint"
    )

    print(
        "Output : stage2_raw_contour"
    )

    print(
        "Method : same pupil fitting algorithm "
        "as working code"
    )

    print(
        "ML     : NO"
    )

    print(
        "YOLO   : NO"
    )

    print(
        "Haar   : NO"
    )

    print(
        "Stage3 : NOT REQUIRED"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # ROOT CHECK
    # ========================================================

    if not os.path.isdir(
        ROOT_HR_FOLDER
    ):

        print()
        print(
            "[ERROR] ROOT_HR_FOLDER does not exist:"
        )

        print(
            ROOT_HR_FOLDER
        )

        return

    # ========================================================
    # FIND ALL STAGE 1 FOLDERS
    # ========================================================

    stage1_folders = find_stage1_folders(
        ROOT_HR_FOLDER
    )

    if not stage1_folders:

        print()
        print(
            "[ERROR] No stage1_glint folder found."
        )

        print()
        print(
            "Expected structure:"
        )

        print(
            "pupil_image"
        )

        print(
            "    main_folder"
        )

        print(
            "        stage1_glint"
        )

        print(
            "            image1.png"
        )

        print(
            "            image2.png"
        )

        return

    print()

    print(
        f"Stage 1 folders found: "
        f"{len(stage1_folders)}"
    )

    total_images = 0

    total_pupils = 0

    # ========================================================
    # PROCESS ALL STAGE 1 FOLDERS
    # ========================================================

    for stage1_folder in stage1_folders:

        processed, detected = (
            process_stage1_folder(
                stage1_folder
            )
        )

        total_images += processed

        total_pupils += detected

    # ========================================================
    # CLOSE WINDOWS
    # ========================================================

    if DISPLAY_VISUALS:

        cv2.destroyAllWindows()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "STAGE 2 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"Stage 1 folders processed : "
        f"{len(stage1_folders)}"
    )

    print(
        f"Images processed          : "
        f"{total_images}"
    )

    print(
        f"Pupil detections          : "
        f"{total_pupils}"
    )

    print()

    print(
        "Stage 2 output contains:"
    )

    print(
        "    Raw pupil contour"
    )

    print(
        "    Stage 1 glint marker"
    )

    print(
        "    pupil_raw_contour_points.csv"
    )

    print()

    print(
        "Stage 2 does NOT contain:"
    )

    print(
        "    Pupil circle"
    )

    print(
        "    Angle"
    )

    print(
        "    Intensity profile"
    )

    print(
        "    5-point averaging"
    )

    print(
        "    Straight-line fit"
    )

    print(
        "    Slope"
    )

    print()

    print(
        "Stage 3 can now use the Stage 2 CSV "
        "for angle calculation."
    )

    print(
        "=" * 80
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()