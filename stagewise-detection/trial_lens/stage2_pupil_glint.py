#!/usr/bin/env python3
"""
STAGE 2: DETECT GLINT ONLY INSIDE THE STAGE 1 PUPIL MASKS

Stage 1 outputs used:
    annotated/<image>_annotated.png
    masks/<image>_mask_left.png
    masks/<image>_mask_right.png
    csv/results.csv

Important:
- Original grayscale pixels are used to measure glint brightness.
- The CSV tells us which pupils were detected.
- The pupil mask strictly limits the search region.
- Annotated images are used to display the result.
- No pupil segmentation is performed again.
"""

from pathlib import Path
import csv
import os

import cv2
import numpy as np


# ============================================================
# 2. GLINT PARAMETERS
# ============================================================

# Exclude pixels near the pupil boundary.
# Increase this if rim reflections are detected.
BOUNDARY_MARGIN = 2

# Candidate must be brighter than its local background.
MIN_LOCAL_CONTRAST = 12

# Candidate must also be sufficiently bright compared with
# the overall brightness inside the detected pupil.
MIN_ABOVE_MEDIAN = 20

# Percentile used to adapt the brightness threshold to the
# brightness of each pupil. This helps with exposure changes.
PUPIL_BRIGHT_PERCENTILE = 97.0

# Allowed connected-component area, in pixels.
MIN_GLINT_AREA = 2
MAX_GLINT_AREA = 300

# Reject highly elongated bright streaks.
MAX_ASPECT_RATIO = 3.5

# Local background estimation scale.
# This Gaussian blur estimates the surrounding brightness.
BACKGROUND_SIGMA = 5.0

# Glint annotation colour: red in OpenCV's BGR colour order.
GLINT_COLOR = (0, 0, 255)


# ============================================================
# 3. READ ORIGINAL IMAGE AS GRAYSCALE
# ============================================================

def read_gray(image_path):
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(
                image, cv2.COLOR_BGRA2GRAY
            )
        else:
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    return image


# ============================================================
# 4. READ THE STAGE 1 CSV
# ============================================================

def load_stage1_csv(csv_path):
    """
    Create a lookup table using the original image filename.

    left_found / right_found indicate whether Stage 1 found
    the corresponding pupil.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Stage 1 CSV not found: {csv_path}"
        )

    result = {}

    with open(
        csv_path, "r", newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError(
                "The Stage 1 CSV must contain an 'image' column."
            )

        for row in reader:
            result[row["image"]] = row

    return result


def pupil_was_found(csv_row, side):
    value = csv_row.get(f"{side}_found", "0")
    return str(value).strip().lower() in {
        "1", "1.0", "true", "yes"
    }


# ============================================================
# 5. LOAD THE EXISTING STAGE 1 PUPIL MASK
# ============================================================

def load_pupil_mask(mask_path, image_shape):
    """
    White pixels = detected pupil.
    Black pixels = outside the pupil.

    Erosion creates a smaller, safer region inside the pupil,
    reducing the chance of detecting a reflection on its rim.
    """

    mask = cv2.imread(
        str(mask_path), cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        return None, None

    if mask.shape != image_shape:
        raise ValueError(
            f"Mask dimensions do not match image: {mask_path.name}"
        )

    pupil_mask = (mask > 0).astype(np.uint8)

    if np.count_nonzero(pupil_mask) == 0:
        return pupil_mask, pupil_mask.copy()

    if BOUNDARY_MARGIN > 0:
        diameter = 2 * BOUNDARY_MARGIN + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (diameter, diameter)
        )

        safe_mask = cv2.erode(
            pupil_mask,
            kernel,
            iterations=1
        )
    else:
        safe_mask = pupil_mask.copy()

    return pupil_mask, safe_mask


# ============================================================
# 6. DETECT GLINT WITHIN ONE PUPIL ONLY
# ============================================================

def detect_glint_inside_pupil(gray, pupil_mask, safe_mask):
    """
    Search for a compact, bright, high-local-contrast component.

    All candidates are restricted to safe_mask. Therefore,
    a bright spot outside the supplied pupil mask is ignored.

    Returns:
        best_candidate: dictionary or None
        candidate_count: number of accepted candidates
    """

    pupil_values = gray[pupil_mask > 0]

    if pupil_values.size < 20:
        return None, 0

    pupil_median = float(np.median(pupil_values))

    percentile_value = float(
        np.percentile(
            pupil_values,
            PUPIL_BRIGHT_PERCENTILE
        )
    )

    bright_threshold = max(
        pupil_median + MIN_ABOVE_MEDIAN,
        percentile_value
    )

    bright_threshold = min(bright_threshold, 255.0)

    gray_float = gray.astype(np.float32)

    local_background = cv2.GaussianBlur(
        gray_float,
        (0, 0),
        sigmaX=BACKGROUND_SIGMA,
        sigmaY=BACKGROUND_SIGMA
    )

    local_contrast = gray_float - local_background

    candidate_pixels = (
        (gray_float >= bright_threshold) &
        (local_contrast >= MIN_LOCAL_CONTRAST) &
        (safe_mask > 0)
    )

    candidate_binary = candidate_pixels.astype(np.uint8)

    number, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            candidate_binary,
            connectivity=8
        )
    )

    candidates = []

    for label in range(1, number):

        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < MIN_GLINT_AREA or area > MAX_GLINT_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1)
        )

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        component = labels == label

        if np.any(component & (pupil_mask == 0)):
            continue

        component_brightness = gray_float[component]
        component_contrast = local_contrast[component]

        peak = float(component_brightness.max())
        mean_brightness = float(component_brightness.mean())
        mean_contrast = float(component_contrast.mean())

        if mean_contrast < MIN_LOCAL_CONTRAST:
            continue

        cx, cy = centroids[label]

        score = (
            1.5 * mean_contrast
            + 0.35 * peak
            + 0.25 * min(area, 30)
        )

        candidates.append({
            "x": round(float(cx), 2),
            "y": round(float(cy), 2),
            "area_px": area,
            "peak_brightness": round(peak, 2),
            "mean_brightness": round(mean_brightness, 2),
            "local_contrast": round(mean_contrast, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "score": round(float(score), 2),
            "brightness_threshold": round(
                bright_threshold, 2
            )
        })

    if not candidates:
        return None, 0

    best_candidate = max(
        candidates,
        key=lambda candidate: candidate["score"]
    )

    return best_candidate, len(candidates)


# ============================================================
# 7. PROCESS ONE ORIGINAL IMAGE
# ============================================================

def process_image(image_path, csv_row, annotated_folder, mask_folder, output_folder):

    gray = read_gray(image_path)

    annotated_path = (
        annotated_folder /
        f"{image_path.stem}_annotated.png"
    )

    if annotated_path.exists():
        display_image = cv2.imread(
            str(annotated_path), cv2.IMREAD_COLOR
        )
    else:
        display_image = cv2.imread(
            str(image_path), cv2.IMREAD_COLOR
        )

    if display_image is None:
        raise ValueError("Cannot read image for annotation.")

    if display_image.shape[:2] != gray.shape:
        raise ValueError(
            "Annotated image and original image dimensions differ."
        )

    row = {"image": image_path.name}

    for side in ("left", "right"):

        if not pupil_was_found(csv_row, side):
            row[f"{side}_status"] = "PUPIL_NOT_FOUND"
            continue

        mask_path = (
            mask_folder /
            f"{image_path.stem}_mask_{side}.png"
        )

        if not mask_path.exists():
            row[f"{side}_status"] = "MASK_NOT_FOUND"
            continue

        pupil_mask, safe_mask = load_pupil_mask(
            mask_path, gray.shape
        )

        if pupil_mask is None:
            row[f"{side}_status"] = "MASK_NOT_READABLE"
            continue

        if np.count_nonzero(safe_mask) < 20:
            row[f"{side}_status"] = "PUPIL_MASK_TOO_SMALL"
            continue

        glint, candidate_count = detect_glint_inside_pupil(
            gray, pupil_mask, safe_mask
        )

        row[f"{side}_candidates"] = candidate_count

        if glint is None:
            row[f"{side}_status"] = "NO_VALID_GLINT"
            continue

        gx = int(round(glint["x"]))
        gy = int(round(glint["y"]))

        if (
            gy < 0 or gy >= safe_mask.shape[0] or
            gx < 0 or gx >= safe_mask.shape[1] or
            safe_mask[gy, gx] == 0
        ):
            row[f"{side}_status"] = "CENTER_OUTSIDE_SAFE_MASK"
            continue

        cv2.circle(
            display_image,
            (gx, gy),
            4,
            GLINT_COLOR,
            1,
            cv2.LINE_AA
        )

        cv2.drawMarker(
            display_image,
            (gx, gy),
            GLINT_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=9,
            thickness=2,
            line_type=cv2.LINE_AA
        )

        row[f"{side}_status"] = "GLINT_FOUND"

        for key, value in glint.items():
            row[f"{side}_{key}"] = value

        outline_status = str(
            csv_row.get(f"{side}_outline", "")
        ).strip().lower()

        row[f"{side}_stage1_outline"] = (
            outline_status if outline_status else "not_recorded"
        )

    output_path = (
        output_folder / "images" /
        f"{image_path.stem}_glint.png"
    )

    cv2.imwrite(str(output_path), display_image)

    return row


# ============================================================
# 8. PROCESS ALL IMAGES AND SAVE CSV
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 2 - PUPIL GLINT DETECTION (INTERACTIVE INPUT)")
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

    # Step 3: Enter each subfolder or dataset path
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
        annotated_folder = folder / "annotated"
        mask_folder = folder / "masks"
        stage1_csv = folder / "csv" / "results.csv"
        output_folder = folder / "stage2_glint"

        output_folder.joinpath("images").mkdir(
            parents=True, exist_ok=True
        )

        output_folder.joinpath("csv").mkdir(
            parents=True, exist_ok=True
        )

        try:
            stage1_rows = load_stage1_csv(stage1_csv)
        except Exception as e:
            print(f"[WARNING] Skipping {folder} due to CSV error: {e}")
            continue

        supported_extensions = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        }

        image_files = sorted(
            path for path in original_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in supported_extensions
            and not path.stem.endswith(
                ("_annotated", "_overlay", "_glint")
            )
            and "_mask_left" not in path.stem
            and "_mask_right" not in path.stem
        )

        if not image_files:
            print(f"[WARNING] No original images found in: {original_folder}")
            continue

        rows = []

        for index_img, image_path in enumerate(image_files, start=1):

            csv_row = stage1_rows.get(image_path.name)

            if csv_row is None:
                result = {
                    "image": image_path.name,
                    "status": "IMAGE_NOT_FOUND_IN_STAGE1_CSV"
                }
            else:
                try:
                    result = process_image(
                        image_path, csv_row, 
                        annotated_folder, mask_folder, output_folder
                    )
                except Exception as exc:
                    result = {
                        "image": image_path.name,
                        "status": f"ERROR: {exc}"
                    }

            rows.append(result)

            print(
                f"[{index_img}/{len(image_files)}] "
                f"{image_path.name}: "
                f"L={result.get('left_status', 'CHECK')} | "
                f"R={result.get('right_status', 'CHECK')}"
            )

        columns = ["image"]

        for result in rows:
            for key in result:
                if key not in columns:
                    columns.append(key)

        output_csv = (
            output_folder / "csv" / "glint_results.csv"
        )

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nStage 2 completed for {folder.name}.")
        print(f"Annotated results: {output_folder / 'images'}")
        print(f"Glint CSV: {output_csv}")

    print()
    print("=" * 80)
    print("ALL STAGE 2 PROCESSING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()