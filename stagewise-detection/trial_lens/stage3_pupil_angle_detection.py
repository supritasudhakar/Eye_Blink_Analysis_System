#!/usr/bin/env python3
"""
STAGE 3: PUPIL INTENSITY-BASED ORIENTATION ANGLE

INPUTS
------
Stage 1:
    csv/results.csv
    masks/<image>_mask_left.png
    masks/<image>_mask_right.png

Stage 2:
    stage2_glint/csv/glint_results.csv
    stage2_glint/images/<image>_glint.png

OUTPUTS
-------
stage3_angle/images/<image>_angle.png
stage3_angle/csv/angle_results.csv

The script reuses the Stage 1 pupil masks and Stage 2 glint detections.
It does not run pupil detection or glint detection again.
No YOLO, ML, or Haar cascade is used.
"""

from pathlib import Path
import csv
import math
import os
import traceback

import cv2
import numpy as np


# ============================================================
# 2. ORIENTATION PARAMETERS
# ============================================================

# Keep the inner pupil region; exclude pixels too close to the rim.
ORIENTATION_INNER_FRAC = 0.85

# Smooth sensor noise before calculating intensity weights.
ORIENTATION_BLUR_SIGMA = 1.5

# Reject very weak directions if desired.
MIN_STRENGTH = 0.0

# Exclude a circular region around the Stage 2 glint centre.
GLINT_EXCLUDE_RADIUS_MIN = 5
GLINT_EXCLUDE_RADIUS_MAX = 12
GLINT_EXCLUDE_RADIUS_SCALE = 2.5

# True: both displayed arrows use the common circular-mean angle.
USE_COMMON_ANGLE_FOR_BOTH_EYES = True

# If True, the angle range is 0 to <360 degrees.
NORMALIZE_ANGLE_TO_360 = True


# ============================================================
# 3. VISUAL SETTINGS
# ============================================================

DRAW_PUPIL_BOUNDARY = True
DRAW_PUPIL_CENTER = True
DRAW_ORIENTATION_ARROW = True
DRAW_ANGLE_TEXT = True
DRAW_GLINT_CENTER = True

# OpenCV uses BGR colour order.
PUPIL_BOUNDARY_COLOR = (0, 255, 255)  # Yellow
PUPIL_CENTER_COLOR = (0, 255, 0)      # Green
ARROW_COLOR = (255, 255, 0)            # Cyan
ANGLE_TEXT_COLOR = (255, 255, 255)     # White
GLINT_COLOR = (0, 0, 255)             # Red

BOUNDARY_THICKNESS = 1
CENTER_RADIUS = 3
ARROW_THICKNESS = 2
ARROW_LENGTH_FRACTION = 0.75
ANGLE_TEXT_SCALE = 0.45
ANGLE_TEXT_THICKNESS = 1

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


# ============================================================
# 4. CSV AND IMAGE HELPERS
# ============================================================

def normalize_name(name):
    """Normalize an image filename for case-insensitive matching."""
    return Path(str(name).strip()).name.lower()


def read_csv_index(csv_path):
    """Read a CSV and index each row by its image filename."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV not found: {csv_path}")

    indexed = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError(f"CSV must contain an 'image' column: {csv_path}")
        for row in reader:
            if row.get("image", "").strip():
                indexed[normalize_name(row["image"])] = row
    return indexed


def csv_number(row, key):
    """Safely read a finite number from a CSV cell."""
    if row is None:
        return None
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def read_gray(image_path):
    """Read an original image as 8-bit grayscale."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if gray is None:
        raise ValueError(f"Cannot read original image: {image_path}")

    if gray.ndim == 3:
        if gray.shape[2] == 4:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if gray.dtype == np.uint16:
        gray = (gray / 256).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


# ============================================================
# 5. LOAD THE EXISTING STAGE 1 PUPIL MASK
# ============================================================

def load_pupil(mask_path, expected_shape):
    """Load a saved pupil mask and calculate its centre and contour."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    if mask.shape != expected_shape:
        raise ValueError(f"Mask/image dimensions differ: {mask_path.name}")

    binary = (mask > 0).astype(np.uint8)
    if int(np.count_nonzero(binary)) < 20:
        return None

    moments = cv2.moments(binary, binaryImage=True)
    if abs(moments["m00"]) < 1e-8:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    contours, _ = cv2.findContours(
        binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2).astype(np.float64)
    distances = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    radius = float(np.median(distances)) if distances.size else 0.0
    if radius <= 1:
        return None

    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_distance = float(distance_map.max())
    if max_distance <= 0:
        return None

    minimum_distance = (1.0 - ORIENTATION_INNER_FRAC) * max_distance
    inner_mask = (binary > 0) & (distance_map >= minimum_distance)

    return {
        "mask": binary,
        "inner_mask": inner_mask,
        "contour": contour,
        "cx": float(cx),
        "cy": float(cy),
        "radius": radius,
    }


# ============================================================
# 6. EXCLUDE THE ALREADY-DETECTED GLINT
# ============================================================

def remove_glint_region(inner_mask, glint_x, glint_y, glint_area):
    """Remove the existing Stage 2 glint and nearby pixels."""
    usable = inner_mask.astype(np.uint8).copy()

    if glint_x is None or glint_y is None:
        return usable > 0, 0

    estimated_radius = math.sqrt(max(float(glint_area or 0), 1.0) / math.pi)
    exclude_radius = int(round(GLINT_EXCLUDE_RADIUS_SCALE * estimated_radius))
    exclude_radius = max(
        GLINT_EXCLUDE_RADIUS_MIN,
        min(GLINT_EXCLUDE_RADIUS_MAX, exclude_radius),
    )

    gx = int(round(glint_x))
    gy = int(round(glint_y))

    cv2.circle(
        usable, (gx, gy), exclude_radius, 0, thickness=cv2.FILLED
    )
    return usable > 0, exclude_radius


# ============================================================
# 7. INTENSITY-BASED ORIENTATION
# ============================================================

def calculate_orientation(gray, pupil, glint_row, side):
    """Calculate the angle from pupil centre toward its intensity-weighted bright point."""
    glint_status = str(
        (glint_row or {}).get(f"{side}_status", "")
    ).strip().upper()

    if glint_status != "GLINT_FOUND":
        return {
            "status": "GLINT_NOT_FOUND",
            "angle_deg": None,
            "strength": None,
        }

    glint_x = csv_number(glint_row, f"{side}_x")
    glint_y = csv_number(glint_row, f"{side}_y")
    glint_area = csv_number(glint_row, f"{side}_area_px")

    if glint_x is None or glint_y is None:
        return {
            "status": "GLINT_CENTER_MISSING",
            "angle_deg": None,
            "strength": None,
        }

    blurred = cv2.GaussianBlur(
        gray.astype(np.float32),
        (0, 0),
        sigmaX=ORIENTATION_BLUR_SIGMA,
        sigmaY=ORIENTATION_BLUR_SIGMA,
    )

    usable_mask, exclusion_radius = remove_glint_region(
        pupil["inner_mask"], glint_x, glint_y, glint_area
    )

    ys, xs = np.nonzero(usable_mask)
    if len(xs) < 20:
        return {
            "status": "TOO_FEW_VALID_PIXELS",
            "angle_deg": None,
            "strength": None,
        }

    values = blurred[ys, xs].astype(np.float64)
    weights = np.clip(values - float(values.min()), 0.0, None)
    total_weight = float(weights.sum())

    if total_weight <= 1e-6:
        return {
            "status": "NO_INTENSITY_VARIATION",
            "angle_deg": None,
            "strength": 0.0,
        }

    bright_x = float(np.sum(xs * weights) / total_weight)
    bright_y = float(np.sum(ys * weights) / total_weight)

    vx = bright_x - pupil["cx"]
    vy = bright_y - pupil["cy"]
    magnitude = float(math.hypot(vx, vy))

    if magnitude < 1e-6:
        return {
            "status": "DIRECTION_UNDEFINED",
            "angle_deg": None,
            "strength": 0.0,
        }

    ux = vx / magnitude
    uy = vy / magnitude

    angle = math.degrees(math.atan2(uy, ux))
    if NORMALIZE_ANGLE_TO_360:
        angle %= 360.0

    strength = magnitude / max(pupil["radius"], 1e-6)
    if strength < MIN_STRENGTH:
        return {
            "status": "WEAK_ORIENTATION",
            "angle_deg": None,
            "strength": float(strength),
        }

    arrow_length = max(3.0, ARROW_LENGTH_FRACTION * pupil["radius"])

    return {
        "status": "ANGLE_CALCULATED",
        "angle_deg": float(angle),
        "raw_angle_deg": float(angle),
        "strength": float(strength),
        "dx": float(ux),
        "dy": float(uy),
        "bright_x": bright_x,
        "bright_y": bright_y,
        "glint_x": float(glint_x),
        "glint_y": float(glint_y),
        "glint_exclusion_radius": int(exclusion_radius),
        "arrow_start": (
            pupil["cx"] - ux * arrow_length,
            pupil["cy"] - uy * arrow_length,
        ),
        "arrow_end": (
            pupil["cx"] + ux * arrow_length,
            pupil["cy"] + uy * arrow_length,
        ),
    }


# ============================================================
# 8. COMMON ANGLE FOR BOTH EYES (OPTIONAL)
# ============================================================

def circular_mean_degrees(angles):
    """Average angles circularly."""
    valid = [
        float(a) for a in angles
        if a is not None and math.isfinite(float(a))
    ]
    if not valid:
        return None

    radians = np.radians(valid)
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))

    if abs(mean_sin) < 1e-12 and abs(mean_cos) < 1e-12:
        return valid[0]

    angle = math.degrees(math.atan2(mean_sin, mean_cos))
    return angle % 360.0 if NORMALIZE_ANGLE_TO_360 else angle


def set_common_orientation(orientations, pupils):
    """Optionally set a shared DISPLAY angle while preserving each raw angle."""
    if not USE_COMMON_ANGLE_FOR_BOTH_EYES:
        return orientations, None

    valid_angles = [
        orientations[side].get("raw_angle_deg")
        for side in ("left", "right")
        if orientations.get(side) is not None
        and orientations[side].get("raw_angle_deg") is not None
    ]

    common_angle = circular_mean_degrees(valid_angles)
    if common_angle is None:
        return orientations, None

    radians = math.radians(common_angle)
    ux, uy = math.cos(radians), math.sin(radians)

    for side in ("left", "right"):
        orientation = orientations.get(side)
        pupil = pupils.get(side)

        if (
            orientation is None
            or pupil is None
            or orientation.get("raw_angle_deg") is None
        ):
            continue

        length = max(3.0, ARROW_LENGTH_FRACTION * pupil["radius"])
        orientation["display_angle_deg"] = float(common_angle)
        orientation["dx"] = ux
        orientation["dy"] = uy
        orientation["arrow_start"] = (
            pupil["cx"] - ux * length,
            pupil["cy"] - uy * length,
        )
        orientation["arrow_end"] = (
            pupil["cx"] + ux * length,
            pupil["cy"] + uy * length,
        )

    return orientations, common_angle


# ============================================================
# 9. DRAW REFERENCE-STYLE ANNOTATIONS
# ============================================================

def draw_eye_annotation(image, pupil, orientation):
    """Draw the pupil boundary, centre, angle arrow, label and glint marker."""
    if pupil is None:
        return

    if DRAW_PUPIL_BOUNDARY:
        cv2.drawContours(
            image,
            [pupil["contour"]],
            -1,
            PUPIL_BOUNDARY_COLOR,
            BOUNDARY_THICKNESS,
            cv2.LINE_AA,
        )

    cx = int(round(pupil["cx"]))
    cy = int(round(pupil["cy"]))

    if DRAW_PUPIL_CENTER:
        cv2.circle(
            image, (cx, cy), CENTER_RADIUS,
            PUPIL_CENTER_COLOR, -1, cv2.LINE_AA,
        )

    if orientation is None or orientation.get("angle_deg") is None:
        return

    if DRAW_GLINT_CENTER:
        gx = orientation.get("glint_x")
        gy = orientation.get("glint_y")
        if gx is not None and gy is not None:
            cv2.drawMarker(
                image,
                (int(round(gx)), int(round(gy))),
                GLINT_COLOR,
                cv2.MARKER_CROSS,
                7,
                1,
                cv2.LINE_AA,
            )

    if DRAW_ORIENTATION_ARROW and "arrow_start" in orientation:
        start = tuple(int(round(v)) for v in orientation["arrow_start"])
        end = tuple(int(round(v)) for v in orientation["arrow_end"])
        cv2.arrowedLine(
            image, start, end, ARROW_COLOR,
            ARROW_THICKNESS, cv2.LINE_AA, tipLength=0.30,
        )

    if DRAW_ANGLE_TEXT:
        angle = orientation.get(
            "display_angle_deg", orientation.get("angle_deg")
        )
        radius = int(round(pupil["radius"]))
        tx = max(2, cx - radius)
        ty = max(15, cy - radius - 5)
        cv2.putText(
            image,
            f"{angle:.1f} deg",
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            ANGLE_TEXT_SCALE,
            ANGLE_TEXT_COLOR,
            ANGLE_TEXT_THICKNESS,
            cv2.LINE_AA,
        )


# ============================================================
# 10. PROCESS ONE IMAGE
# ============================================================

def process_image(image_path, stage1_row, stage2_row, mask_folder, stage2_images, output_images):
    """Calculate both eye angles using existing masks and glint CSV."""
    gray = read_gray(image_path)
    stem = image_path.stem

    visual_path = stage2_images / f"{stem}_glint.png"
    if visual_path.exists():
        visual = cv2.imread(str(visual_path), cv2.IMREAD_COLOR)
    else:
        visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if visual is None:
        visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if visual.shape[:2] != gray.shape:
        raise ValueError("Stage 2 annotated image dimensions do not match original image")

    result = {"image": image_path.name, "status": "CHECK"}
    pupils = {"left": None, "right": None}
    orientations = {"left": None, "right": None}

    for side in ("left", "right"):
        mask_path = mask_folder / f"{stem}_mask_{side}.png"

        if not mask_path.exists():
            result[f"{side}_status"] = "MASK_NOT_FOUND"
            result[f"{side}_stage1_found_flag"] = (
                stage1_row.get(f"{side}_found", "") if stage1_row else ""
            )
            continue

        pupil = load_pupil(mask_path, gray.shape)
        if pupil is None:
            result[f"{side}_status"] = "EMPTY_OR_INVALID_MASK"
            result[f"{side}_stage1_found_flag"] = (
                stage1_row.get(f"{side}_found", "") if stage1_row else ""
            )
            continue

        pupils[side] = pupil
        result[f"{side}_stage1_found_flag"] = (
            stage1_row.get(f"{side}_found", "") if stage1_row else ""
        )

        orientations[side] = calculate_orientation(
            gray, pupil, stage2_row, side
        )
        result[f"{side}_status"] = orientations[side]["status"]

    orientations, common_angle = set_common_orientation(orientations, pupils)
    result["common_angle_deg"] = (
        round(common_angle, 3) if common_angle is not None else ""
    )

    valid_eye_count = 0

    for side in ("left", "right"):
        pupil = pupils.get(side)
        orientation = orientations.get(side)

        draw_eye_annotation(visual, pupil, orientation)

        result[f"{side}_center_x"] = round(pupil["cx"], 3) if pupil else ""
        result[f"{side}_center_y"] = round(pupil["cy"], 3) if pupil else ""

        raw_angle = None
        display_angle = None

        if orientation is not None:
            raw_angle = orientation.get(
                "raw_angle_deg", orientation.get("angle_deg")
            )
            display_angle = orientation.get(
                "display_angle_deg", orientation.get("angle_deg")
            )

            result[f"{side}_strength"] = (
                round(orientation["strength"], 6)
                if orientation.get("strength") is not None else ""
            )
            result[f"{side}_glint_x"] = orientation.get("glint_x", "")
            result[f"{side}_glint_y"] = orientation.get("glint_y", "")
            result[f"{side}_glint_exclusion_radius"] = orientation.get(
                "glint_exclusion_radius", ""
            )
        else:
            result[f"{side}_strength"] = ""
            result[f"{side}_glint_x"] = ""
            result[f"{side}_glint_y"] = ""
            result[f"{side}_glint_exclusion_radius"] = ""

        result[f"{side}_raw_angle_deg"] = (
            round(raw_angle, 3) if raw_angle is not None else ""
        )
        result[f"{side}_angle_deg"] = (
            round(display_angle, 3) if display_angle is not None else ""
        )

        if raw_angle is not None:
            valid_eye_count += 1

    if valid_eye_count == 2:
        result["status"] = "OK"
    elif valid_eye_count == 1:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "NO_VALID_ANGLE"

    output_path = output_images / f"{stem}_angle.png"
    result["image_saved"] = int(cv2.imwrite(str(output_path), visual))

    return result


# ============================================================
# 11. MAIN (INTERACTIVE TERMINAL INPUT)
# ============================================================

def main():
    print()
    print("=" * 80)
    print("STAGE 3 - PUPIL INTENSITY-BASED ANGLE (INTERACTIVE INPUT)")
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
    input_paths_list = []
    for i in range(1, num_paths + 1):
        path_input = input(f"Enter input path {i} (subfolder or full path): ").strip().strip('"')
        
        if base_folder and not os.path.isabs(path_input):
            full_path = os.path.join(base_folder, path_input)
        else:
            full_path = path_input
            
        input_paths_list.append(full_path)

    input_folders = [
        Path(os.path.abspath(os.path.expanduser(path)))
        for path in input_paths_list
    ]

    for index, folder in enumerate(input_folders, start=1):
        if not folder.is_dir():
            print()
            print(f"[ERROR] Input path {index} does not exist:")
            print(folder)
            print()
            return

    for folder in input_folders:
        print()
        print("-" * 80)
        print(f"Processing folder: {folder}")
        print("-" * 80)

        original_folder = folder
        mask_folder = folder / "masks"
        stage1_csv = folder / "csv" / "results.csv"

        stage2_folder = folder / "stage2_glint"
        stage2_csv = stage2_folder / "csv" / "glint_results.csv"
        stage2_images = stage2_folder / "images"

        output_folder = folder / "stage3_angle"
        output_images = output_folder / "images"
        output_csv = output_folder / "csv" / "angle_results.csv"

        output_images.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        if not original_folder.exists():
            print(f"[WARNING] Dataset folder does not exist: {original_folder}")
            continue

        print("Stage 1 CSV:", stage1_csv)
        print("Stage 2 CSV:", stage2_csv)
        print("Pupil masks:", mask_folder)
        print("Stage 2 images:", stage2_images)

        try:
            stage1_rows = read_csv_index(stage1_csv)
            stage2_rows = read_csv_index(stage2_csv)
        except Exception as e:
            print(f"[WARNING] Skipping {folder} due to CSV error: {e}")
            continue

        image_files = [
            p for p in original_folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and not p.stem.endswith(("_annotated", "_overlay", "_glint", "_angle"))
            and "_mask_left" not in p.stem
            and "_mask_right" not in p.stem
        ]

        image_index = {normalize_name(p.name): p for p in image_files}
        results = []
        image_keys = sorted(stage2_rows.keys())
        total = len(image_keys)

        if total == 0:
            print(f"[WARNING] No image records found in Stage 2 glint CSV for: {folder}")
            continue

        for index_img, key in enumerate(image_keys, start=1):
            stage2_row = stage2_rows[key]
            image_path = image_index.get(key)

            if image_path is None:
                result = {
                    "image": key,
                    "status": "ORIGINAL_IMAGE_NOT_FOUND",
                    "left_status": "NOT_PROCESSED",
                    "right_status": "NOT_PROCESSED",
                }
                results.append(result)
                print(f"[{index_img}/{total}] {key}: ORIGINAL_IMAGE_NOT_FOUND")
                continue

            stage1_row = stage1_rows.get(key, {})

            try:
                result = process_image(
                    image_path, stage1_row, stage2_row,
                    mask_folder, stage2_images, output_images
                )
            except Exception as exc:
                print(
                    f"[ERROR] {image_path.name}: {type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                result = {
                    "image": image_path.name,
                    "status": f"ERROR: {type(exc).__name__}: {exc}",
                    "left_status": "ERROR",
                    "right_status": "ERROR",
                    "left_raw_angle_deg": "",
                    "right_raw_angle_deg": "",
                    "left_angle_deg": "",
                    "right_angle_deg": "",
                    "common_angle_deg": "",
                }

            results.append(result)

            print(
                f"[{index_img}/{total}] {image_path.name}: "
                f"Status={result.get('status', '-')}, "
                f"L={result.get('left_angle_deg', '-')}, "
                f"R={result.get('right_angle_deg', '-')}, "
                f"Common={result.get('common_angle_deg', '-')}, "
                f"L_status={result.get('left_status', '-')}, "
                f"R_status={result.get('right_status', '-')}"
            )

        columns = []
        for row in results:
            for field in row:
                if field not in columns:
                    columns.append(field)

        with open(output_csv, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nStage 3 completed for {folder.name}.")
        print("Images:", output_images)
        print("CSV:", output_csv)

    print()
    print("=" * 80)
    print("ALL STAGE 3 PROCESSING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()