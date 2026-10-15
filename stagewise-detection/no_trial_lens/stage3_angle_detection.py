import os
import csv
import cv2
import math
import numpy as np


# ============================================================
# STAGE 3
# PUPIL ORIENTATION ANGLE
# ============================================================
#
# INPUT:
#     Stage 2 raw pupil contour CSV
#
#     stage2_raw_contour/
#         pupil_raw_contour_points.csv
#
# OUTPUT:
#     stage3_angle/
#         images/
#             original image + pupil contour + angle arrow
#
#         pupil_angles.csv
#
#
# IMPORTANT:
#
# This Stage 3 keeps the ORIGINAL WORKING ANGLE CALCULATION.
#
# It does NOT:
#     - run YOLO
#     - run ML
#     - run Haar
#     - detect pupil again
#     - calculate intensity profile
#     - calculate slope
#
# Stage 2 provides the pupil contour.
#
# Stage 3 only calculates pupil orientation.
#
# ============================================================


# ============================================================
# ROOT FOLDER
# ============================================================

ROOT_HR_FOLDER = (
   )


# ============================================================
# STAGE FOLDER NAMES
# ============================================================

STAGE2_FOLDER_NAME = "stage2_raw_contour"
STAGE3_FOLDER_NAME = "stage3_angle"

STAGE2_CSV_NAME = "pupil_raw_contour_points.csv"
STAGE3_CSV_NAME = "pupil_angles.csv"


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
# OUTPUT SETTINGS
# ============================================================

SAVE_STAGE3_IMAGES = True
SAVE_STAGE3_CSV = True


# ============================================================
# VISUAL SETTINGS
# ============================================================

DRAW_PUPIL_CONTOUR = True
DRAW_CENTER = True
DRAW_ORIENTATION_ARROW = True
DRAW_ANGLE_TEXT = True


# ============================================================
# COLORS
# ============================================================

# Yellow pupil contour
CONTOUR_COLOR = (0, 255, 255)

# Green pupil center
CENTER_COLOR = (0, 255, 0)

# Cyan orientation arrow
ORIENTATION_ARROW_COLOR = (255, 255, 0)

# White text
TEXT_COLOR = (255, 255, 255)


# ============================================================
# DRAWING SETTINGS
# ============================================================

PUPIL_CONTOUR_THICKNESS = 1

CENTER_RADIUS = 3

ORIENTATION_ARROW_THICKNESS = 2

ANGLE_TEXT_SCALE = 0.45

ANGLE_TEXT_THICKNESS = 1


# ============================================================
# ORIGINAL WORKING ORIENTATION SETTINGS
# ============================================================
#
# These values are kept from the working code.
#
# ============================================================

ORIENTATION_BLUR_SIGMA = 1.5

ORIENTATION_MIN_STRENGTH = 0.0

ORIENTATION_ARROW_LENGTH_FRAC = 0.75


# ============================================================
# FIND STAGE 2 CSV FILES RECURSIVELY
# ============================================================

def find_stage2_csv_files(root_folder):

    found = []

    for current_root, dirs, files in os.walk(root_folder):

        # Never search inside Stage 3 output
        dirs[:] = [
            d for d in dirs
            if d.lower() != STAGE3_FOLDER_NAME.lower()
        ]

        # We only want folders named stage2_raw_contour
        if os.path.basename(
            current_root
        ).lower() != STAGE2_FOLDER_NAME.lower():

            continue

        for file_name in files:

            if file_name.lower() == STAGE2_CSV_NAME.lower():

                found.append(
                    os.path.join(
                        current_root,
                        file_name
                    )
                )

    return sorted(found)


# ============================================================
# FIND ALL ORIGINAL IMAGES RECURSIVELY
# ============================================================

def find_all_images(root_folder):

    image_files = []

    for current_root, dirs, files in os.walk(root_folder):

        # Never search generated output folders
        dirs[:] = [
            d for d in dirs
            if d.lower() not in (
                STAGE2_FOLDER_NAME.lower(),
                STAGE3_FOLDER_NAME.lower()
            )
        ]

        for file_name in files:

            if file_name.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                image_files.append(
                    os.path.join(
                        current_root,
                        file_name
                    )
                )

    return sorted(image_files)


# ============================================================
# NORMALIZE IMAGE NAME
# ============================================================

def normalize_image_name(name):

    return os.path.basename(
        str(name)
    ).strip().lower()


# ============================================================
# LOAD STAGE 2 RAW CONTOURS
# ============================================================

def load_stage2_contours(csv_path):

    contours = {}

    try:

        with open(
            csv_path,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                image_name = row.get(
                    "image",
                    ""
                ).strip()

                eye = row.get(
                    "eye",
                    ""
                ).strip().upper()

                if not image_name:
                    continue

                # --------------------------------------------
                # Normalize eye name
                # --------------------------------------------

                if eye in ("LEFT", "LE"):

                    eye_key = "LE"

                elif eye in ("RIGHT", "RE"):

                    eye_key = "RE"

                else:

                    continue

                try:

                    x = float(
                        row["x"]
                    )

                    y = float(
                        row["y"]
                    )

                except (
                    ValueError,
                    TypeError,
                    KeyError
                ):

                    continue

                key = (
                    normalize_image_name(
                        image_name
                    ),
                    eye_key
                )

                if key not in contours:

                    contours[key] = []

                contours[key].append(
                    (
                        float(x),
                        float(y)
                    )
                )

    except Exception as e:

        print()
        print(
            "[ERROR] Cannot read Stage 2 CSV:"
        )

        print(csv_path)

        print(e)

        return {}

    return contours


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

def build_image_index(image_files):

    image_index = {}

    for image_path in image_files:

        name = normalize_image_name(
            os.path.basename(
                image_path
            )
        )

        if name not in image_index:

            image_index[name] = []

        image_index[name].append(
            image_path
        )

    return image_index


# ============================================================
# FIND IMAGE FOR DATASET
# ============================================================

def find_matching_image(
    image_name,
    image_index,
    dataset_folder
):

    key = normalize_image_name(
        image_name
    )

    candidates = image_index.get(
        key,
        []
    )

    if not candidates:

        return None

    dataset_folder = os.path.abspath(
        dataset_folder
    )

    # --------------------------------------------------------
    # Prefer image inside the same dataset folder
    # --------------------------------------------------------

    local_candidates = []

    for path in candidates:

        path_abs = os.path.abspath(
            path
        )

        try:

            common = os.path.commonpath(
                [
                    path_abs,
                    dataset_folder
                ]
            )

        except ValueError:

            continue

        if common == dataset_folder:

            local_candidates.append(
                path
            )

    # --------------------------------------------------------
    # Only one local image
    # --------------------------------------------------------

    if len(local_candidates) == 1:

        return local_candidates[0]

    # --------------------------------------------------------
    # Multiple local images
    # --------------------------------------------------------

    if len(local_candidates) > 1:

        # Prefer the closest one
        local_candidates.sort(
            key=lambda p: len(
                os.path.relpath(
                    p,
                    dataset_folder
                ).split(os.sep)
            )
        )

        return local_candidates[0]

    # --------------------------------------------------------
    # Only one global match
    # --------------------------------------------------------

    if len(candidates) == 1:

        return candidates[0]

    # --------------------------------------------------------
    # Duplicate filename
    # --------------------------------------------------------

    print()
    print(
        "[WARNING] Multiple images have the same name:"
    )

    print(image_name)

    print(
        "Using:"
    )

    print(candidates[0])

    return candidates[0]


# ============================================================
# BUILD MASK FROM RAW CONTOUR
# ============================================================

def build_contour_mask(
    gray,
    contour
):

    mask = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    if contour is None:

        return mask

    if len(contour) < 3:

        return mask

    pts = np.round(
        np.asarray(
            contour,
            dtype=np.float32
        )
    ).astype(
        np.int32
    )

    pts = pts.reshape(
        (-1, 1, 2)
    )

    cv2.fillPoly(
        mask,
        [pts],
        255
    )

    return mask


# ============================================================
# CALCULATE PUPIL CENTER
# ============================================================

def calculate_pupil_center(
    contour
):

    if contour is None:

        return None

    if len(contour) < 3:

        return None

    pts = np.asarray(
        contour,
        dtype=np.float32
    )

    pts = pts.reshape(
        (-1, 1, 2)
    )

    moments = cv2.moments(
        pts
    )

    if abs(
        moments["m00"]
    ) > 1e-8:

        cx = (
            moments["m10"]
            /
            moments["m00"]
        )

        cy = (
            moments["m01"]
            /
            moments["m00"]
        )

    else:

        cx = float(
            np.mean(
                pts[:, 0, 0]
            )
        )

        cy = float(
            np.mean(
                pts[:, 0, 1]
            )
        )

    return (
        float(cx),
        float(cy)
    )


# ============================================================
# CALCULATE PUPIL RADIUS
# ============================================================

def calculate_pupil_radius(
    contour,
    center
):

    if contour is None:

        return 0.0

    if len(contour) == 0:

        return 0.0

    cx, cy = center

    pts = np.asarray(
        contour,
        dtype=np.float32
    )

    dx = (
        pts[:, 0]
        - cx
    )

    dy = (
        pts[:, 1]
        - cy
    )

    distances = np.hypot(
        dx,
        dy
    )

    if len(distances) == 0:

        return 0.0

    # Same robust idea as the working pupil geometry:
    # median contour distance.
    return float(
        np.median(
            distances
        )
    )


# ============================================================
# CONVERT STAGE 2 CONTOUR INTO PUPIL DICTIONARY
# ============================================================

def make_pupil_from_contour(
    gray,
    contour
):

    if contour is None:

        return None

    if len(contour) < 3:

        return None

    center = calculate_pupil_center(
        contour
    )

    if center is None:

        return None

    radius = calculate_pupil_radius(
        contour,
        center
    )

    if radius <= 1:

        return None

    return {
        "cx": center[0],
        "cy": center[1],
        "r": radius,
        "contour": np.asarray(
            contour,
            dtype=np.float32
        )
    }


# ============================================================
# ORIGINAL WORKING ANGLE CALCULATION
# ============================================================
#
# IMPORTANT:
#
# This is based on the working code supplied by you.
#
# The procedure is:
#
#     1. Create pupil mask from raw contour
#     2. Gaussian blur
#     3. Take pixels inside pupil
#     4. Convert intensity into weights
#     5. Find intensity-weighted bright point
#     6. Calculate vector from pupil center
#     7. Calculate angle using atan2
#
# ============================================================

def compute_pupil_orientation(
    gray,
    pupil
):

    if pupil is None:

        return None

    contour = pupil.get(
        "contour"
    )

    if contour is None:

        return None

    if len(contour) < 3:

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

    if r <= 1:

        return None

    # --------------------------------------------------------
    # Build mask from Stage 2 raw contour
    # --------------------------------------------------------

    mask = build_contour_mask(
        gray,
        contour
    )

    # --------------------------------------------------------
    # Same Gaussian smoothing used by working code
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray.astype(
            np.float32
        ),
        (0, 0),
        ORIENTATION_BLUR_SIGMA
    )

    # --------------------------------------------------------
    # Get pixels inside pupil
    # --------------------------------------------------------

    ys, xs = np.nonzero(
        mask > 0
    )

    if len(xs) < 20:

        return None

    # --------------------------------------------------------
    # Intensity values
    # --------------------------------------------------------

    values = blurred[
        ys,
        xs
    ].astype(
        np.float64
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Same intensity weighting from working code.
    #
    # Lowest intensity becomes zero.
    # Brighter pixels receive higher weight.
    # --------------------------------------------------------

    values = (
        values
        -
        values.min()
    )

    total = float(
        values.sum()
    )

    if total <= 1e-6:

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

    # --------------------------------------------------------
    # Intensity-weighted bright point
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Vector:
    #
    # pupil center -> weighted bright point
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normalize direction
    # --------------------------------------------------------

    ux = vx / mag

    uy = vy / mag

    # --------------------------------------------------------
    # SAME ANGLE CALCULATION
    # --------------------------------------------------------

    angle_deg = float(
        np.degrees(
            np.arctan2(
                uy,
                ux
            )
        )
    )

    # --------------------------------------------------------
    # Same strength concept
    # --------------------------------------------------------

    strength = (
        mag
        /
        max(
            r,
            1e-6
        )
    )

    # --------------------------------------------------------
    # Arrow length
    # --------------------------------------------------------

    arrow_len = max(
        3.0,
        ORIENTATION_ARROW_LENGTH_FRAC * r
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
            and np.isfinite(a)
        )
    ]

    if not valid:

        return None

    radians = np.radians(
        valid
    )

    s = float(
        np.mean(
            np.sin(
                radians
            )
        )
    )

    c = float(
        np.mean(
            np.cos(
                radians
            )
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

    if angle > 180:

        angle -= 360

    elif angle <= -180:

        angle += 360

    return angle


# ============================================================
# COMMON ORIENTATION
# ============================================================
#
# This is also retained from the working code.
#
# Left + Right raw angles
#          ↓
# Circular mean
#          ↓
# Same common direction for both eyes
#
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
            ORIENTATION_ARROW_LENGTH_FRAC * r
        )

        updated[side] = dict(
            o
        )

        updated[side][
            "angle_deg"
        ] = common_angle

        updated[side]["dx"] = ux

        updated[side]["dy"] = uy

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
# DRAW RAW PUPIL CONTOUR
# ============================================================

def draw_raw_contour(
    image,
    pupil
):

    if pupil is None:

        return

    points = pupil.get(
        "contour"
    )

    if points is None:

        return

    if len(points) < 3:

        return

    pts = np.round(
        points
    ).astype(
        np.int32
    )

    pts = pts.reshape(
        (-1, 1, 2)
    )

    cv2.polylines(
        image,
        [pts],
        True,
        CONTOUR_COLOR,
        PUPIL_CONTOUR_THICKNESS,
        cv2.LINE_AA
    )


# ============================================================
# DRAW PUPIL CENTER
# ============================================================

def draw_center(
    image,
    pupil
):

    if pupil is None:

        return

    cx = int(
        round(
            pupil["cx"]
        )
    )

    cy = int(
        round(
            pupil["cy"]
        )
    )

    cv2.circle(
        image,
        (
            cx,
            cy
        ),
        CENTER_RADIUS,
        CENTER_COLOR,
        -1,
        cv2.LINE_AA
    )


# ============================================================
# DRAW ORIENTATION ARROW
# ============================================================
#
# The line passes through the green center.
#
# This produces the type of visualization requested:
#
#       yellow pupil boundary
#              |
#              |
#          <---●---->
#              |
#           green center
#
# ============================================================

def draw_orientation_arrow(
    image,
    orientation
):

    if orientation is None:

        return

    if not orientation.get(
        "ok",
        False
    ):

        return

    p1 = (
        int(
            round(
                orientation[
                    "dim_point"
                ][0]
            )
        ),
        int(
            round(
                orientation[
                    "dim_point"
                ][1]
            )
        )
    )

    p2 = (
        int(
            round(
                orientation[
                    "bright_point"
                ][0]
            )
        ),
        int(
            round(
                orientation[
                    "bright_point"
                ][1]
            )
        )
    )

    # --------------------------------------------------------
    # Full orientation line with arrow
    # --------------------------------------------------------

    cv2.arrowedLine(
        image,
        p1,
        p2,
        ORIENTATION_ARROW_COLOR,
        ORIENTATION_ARROW_THICKNESS,
        cv2.LINE_AA,
        tipLength=0.30
    )


# ============================================================
# DRAW ANGLE TEXT
# ============================================================

def draw_angle_text(
    image,
    pupil,
    orientation
):

    if pupil is None:

        return

    if orientation is None:

        return

    angle = orientation.get(
        "angle_deg"
    )

    if angle is None:

        return

    cx = int(
        round(
            pupil["cx"]
        )
    )

    cy = int(
        round(
            pupil["cy"]
        )
    )

    r = int(
        round(
            pupil["r"]
        )
    )

    text = (
        f"{angle:.1f} deg"
    )

    tx = max(
        2,
        cx - r
    )

    ty = max(
        15,
        cy - r - 5
    )

    cv2.putText(
        image,
        text,
        (
            tx,
            ty
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        ANGLE_TEXT_SCALE,
        TEXT_COLOR,
        ANGLE_TEXT_THICKNESS,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS ONE STAGE 2 CSV
# ============================================================

def process_stage2_csv(
    stage2_csv_path,
    all_image_files,
    all_csv_rows
):

    print()
    print("=" * 80)

    print(
        "STAGE 2 INPUT:"
    )

    print(
        stage2_csv_path
    )

    # --------------------------------------------------------
    # Load contours
    # --------------------------------------------------------

    contours = load_stage2_contours(
        stage2_csv_path
    )

    if not contours:

        print(
            "[WARNING] No contour points found."
        )

        return

    # --------------------------------------------------------
    # Dataset folder
    #
    # Example:
    #
    # sachin/
    #     101-images/
    #         ...
    #     stage2_raw_contour/
    #         pupil_raw_contour_points.csv
    #
    # Dataset folder = sachin
    # --------------------------------------------------------

    stage2_folder = os.path.dirname(
        stage2_csv_path
    )

    dataset_folder = os.path.dirname(
        stage2_folder
    )

    # --------------------------------------------------------
    # Stage 3 output
    # --------------------------------------------------------

    stage3_folder = os.path.join(
        dataset_folder,
        STAGE3_FOLDER_NAME
    )

    stage3_images_folder = os.path.join(
        stage3_folder,
        "images"
    )

    if SAVE_STAGE3_IMAGES:

        os.makedirs(
            stage3_images_folder,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Image index
    # --------------------------------------------------------

    image_index = build_image_index(
        all_image_files
    )

    # --------------------------------------------------------
    # Images represented in CSV
    # --------------------------------------------------------

    image_names = sorted(
        set(
            key[0]
            for key in contours.keys()
        )
    )

    processed = 0

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for image_name in image_names:

        # ----------------------------------------------------
        # Find original image
        # ----------------------------------------------------

        image_path = find_matching_image(
            image_name,
            image_index,
            dataset_folder
        )

        if image_path is None:

            print(
                f"[WARNING] Image not found: "
                f"{image_name}"
            )

            continue

        # ----------------------------------------------------
        # Read original image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                f"[WARNING] Cannot read image:"
            )

            print(
                image_path
            )

            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        output = image.copy()

        # ----------------------------------------------------
        # LEFT CONTOUR
        # ----------------------------------------------------

        left_points = contours.get(
            (
                image_name,
                "LE"
            )
        )

        # ----------------------------------------------------
        # RIGHT CONTOUR
        # ----------------------------------------------------

        right_points = contours.get(
            (
                image_name,
                "RE"
            )
        )

        # ----------------------------------------------------
        # Convert raw contours to pupil dictionaries
        # ----------------------------------------------------

        left_pupil = make_pupil_from_contour(
            gray,
            left_points
        )

        right_pupil = make_pupil_from_contour(
            gray,
            right_points
        )

        pupils = {
            "left": left_pupil,
            "right": right_pupil
        }

        # ----------------------------------------------------
        # CALCULATE ORIENTATION
        # ----------------------------------------------------

        orientation = {
            "left": None,
            "right": None
        }

        if left_pupil is not None:

            orientation["left"] = (
                compute_pupil_orientation(
                    gray,
                    left_pupil
                )
            )

        if right_pupil is not None:

            orientation["right"] = (
                compute_pupil_orientation(
                    gray,
                    right_pupil
                )
            )

        # ----------------------------------------------------
        # COMMON ANGLE
        # ----------------------------------------------------

        orientation, common_angle = (
            apply_common_orientation_angle(
                orientation,
                pupils
            )
        )

        # ====================================================
        # DRAW LEFT EYE
        # ====================================================

        if left_pupil is not None:

            if DRAW_PUPIL_CONTOUR:

                draw_raw_contour(
                    output,
                    left_pupil
                )

            if DRAW_CENTER:

                draw_center(
                    output,
                    left_pupil
                )

            if DRAW_ORIENTATION_ARROW:

                draw_orientation_arrow(
                    output,
                    orientation["left"]
                )

            if DRAW_ANGLE_TEXT:

                draw_angle_text(
                    output,
                    left_pupil,
                    orientation["left"]
                )

        # ====================================================
        # DRAW RIGHT EYE
        # ====================================================

        if right_pupil is not None:

            if DRAW_PUPIL_CONTOUR:

                draw_raw_contour(
                    output,
                    right_pupil
                )

            if DRAW_CENTER:

                draw_center(
                    output,
                    right_pupil
                )

            if DRAW_ORIENTATION_ARROW:

                draw_orientation_arrow(
                    output,
                    orientation["right"]
                )

            if DRAW_ANGLE_TEXT:

                draw_angle_text(
                    output,
                    right_pupil,
                    orientation["right"]
                )

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        if SAVE_STAGE3_IMAGES:

            output_name = os.path.basename(
                image_path
            )

            output_path = os.path.join(
                stage3_images_folder,
                output_name
            )

            cv2.imwrite(
                output_path,
                output
            )

        # ====================================================
        # CSV
        # ====================================================

        row = {
            "image": image_name,
            "source_image": image_path,

            "left_angle_deg": "",
            "right_angle_deg": "",

            "common_angle_deg": "",

            "left_strength": "",
            "right_strength": "",

            "left_center_x": "",
            "left_center_y": "",

            "right_center_x": "",
            "right_center_y": "",

            "left_ok": "",
            "right_ok": ""
        }

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        left_orientation = (
            orientation["left"]
        )

        if left_orientation is not None:

            if (
                left_orientation.get(
                    "angle_deg"
                )
                is not None
            ):

                row[
                    "left_angle_deg"
                ] = (
                    f"{left_orientation['angle_deg']:.4f}"
                )

            row[
                "left_strength"
            ] = (
                f"{left_orientation['strength']:.6f}"
            )

            row[
                "left_ok"
            ] = int(
                left_orientation["ok"]
            )

        if left_pupil is not None:

            row[
                "left_center_x"
            ] = (
                f"{left_pupil['cx']:.3f}"
            )

            row[
                "left_center_y"
            ] = (
                f"{left_pupil['cy']:.3f}"
            )

        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        right_orientation = (
            orientation["right"]
        )

        if right_orientation is not None:

            if (
                right_orientation.get(
                    "angle_deg"
                )
                is not None
            ):

                row[
                    "right_angle_deg"
                ] = (
                    f"{right_orientation['angle_deg']:.4f}"
                )

            row[
                "right_strength"
            ] = (
                f"{right_orientation['strength']:.6f}"
            )

            row[
                "right_ok"
            ] = int(
                right_orientation["ok"]
            )

        if right_pupil is not None:

            row[
                "right_center_x"
            ] = (
                f"{right_pupil['cx']:.3f}"
            )

            row[
                "right_center_y"
            ] = (
                f"{right_pupil['cy']:.3f}"
            )

        # ----------------------------------------------------
        # COMMON ANGLE
        # ----------------------------------------------------

        if common_angle is not None:

            row[
                "common_angle_deg"
            ] = (
                f"{common_angle:.4f}"
            )

        all_csv_rows.append(
            row
        )

        processed += 1

        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        print()
        print(
            f"[OK] {os.path.basename(image_path)}"
        )

        print(
            f"     Left angle   : "
            f"{row['left_angle_deg']}"
        )

        print(
            f"     Right angle  : "
            f"{row['right_angle_deg']}"
        )

        print(
            f"     Common angle : "
            f"{row['common_angle_deg']}"
        )

    print()
    print(
        f"Images processed: {processed}"
    )

    if SAVE_STAGE3_IMAGES:

        print(
            "Stage 3 images:"
        )

        print(
            stage3_images_folder
        )


# ============================================================
# SAVE FINAL CSV
# ============================================================

def save_final_csv(
    root_folder,
    rows
):

    if not rows:

        print()
        print(
            "[WARNING] No angle results available."
        )

        return

    output_folder = os.path.join(
        root_folder,
        STAGE3_FOLDER_NAME
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output_csv = os.path.join(
        output_folder,
        STAGE3_CSV_NAME
    )

    fields = [
        "image",
        "source_image",

        "left_angle_deg",
        "right_angle_deg",

        "common_angle_deg",

        "left_strength",
        "right_strength",

        "left_center_x",
        "left_center_y",

        "right_center_x",
        "right_center_y",

        "left_ok",
        "right_ok"
    ]

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print("=" * 80)

    print(
        "STAGE 3 CSV SAVED"
    )

    print(
        output_csv
    )

    print(
        f"Total images: {len(rows)}"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)

    print(
        "STAGE 3 - PUPIL ORIENTATION ANGLE"
    )

    print("=" * 80)

    print()

    print(
        "ROOT:"
    )

    print(
        ROOT_HR_FOLDER
    )

    print()

    # --------------------------------------------------------
    # Check root
    # --------------------------------------------------------

    if not os.path.isdir(
        ROOT_HR_FOLDER
    ):

        print(
            "[ERROR] Root folder does not exist:"
        )

        print(
            ROOT_HR_FOLDER
        )

        return

    # --------------------------------------------------------
    # Find Stage 2 CSV files
    # --------------------------------------------------------

    print(
        "Searching for Stage 2 contour CSV files..."
    )

    stage2_csv_files = (
        find_stage2_csv_files(
            ROOT_HR_FOLDER
        )
    )

    print(
        f"Stage 2 CSV files found: "
        f"{len(stage2_csv_files)}"
    )

    if not stage2_csv_files:

        print()
        print(
            "[ERROR] No Stage 2 CSV found."
        )

        print()
        print(
            "Expected:"
        )

        print(
            "stage2_pupil_contour/"
        )

        print(
            "    pupil_raw_contour_points.csv"
        )

        return

    # --------------------------------------------------------
    # Find images recursively
    # --------------------------------------------------------

    print()

    print(
        "Searching for original images..."
    )

    all_image_files = (
        find_all_images(
            ROOT_HR_FOLDER
        )
    )

    print(
        f"Images found: "
        f"{len(all_image_files)}"
    )

    if not all_image_files:

        print()
        print(
            "[ERROR] No original images found."
        )

        return

    # --------------------------------------------------------
    # Process every Stage 2 dataset
    # --------------------------------------------------------

    all_rows = []

    for stage2_csv in stage2_csv_files:

        process_stage2_csv(
            stage2_csv,
            all_image_files,
            all_rows
        )

    # --------------------------------------------------------
    # Save combined CSV
    # --------------------------------------------------------

    save_final_csv(
        ROOT_HR_FOLDER,
        all_rows
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "STAGE 3 COMPLETED"
    )

    print("=" * 80)

    print()

    print(
        "Stage 3 uses:"
    )

    print(
        "Stage 2 raw pupil contour"
    )

    print(
        "        ↓"
    )

    print(
        "Original working intensity-weighted "
        "orientation calculation"
    )

    print(
        "        ↓"
    )

    print(
        "Pupil orientation angle"
    )

    print()

    print(
        "No Stage 1 input"
    )

    print(
        "No pupil re-detection"
    )

    print(
        "No YOLO"
    )

    print(
        "No ML"
    )

    print(
        "No Haar"
    )

    print(
        "No intensity profile"
    )

    print(
        "No slope calculation"
    )

    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()