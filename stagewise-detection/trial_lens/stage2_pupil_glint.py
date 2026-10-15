
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

import cv2
import numpy as np


# ============================================================
# 1. PATHS: CHANGE THESE TO YOUR ACTUAL STAGE 1 FOLDER
# ============================================================

# This folder contains your original images AND the Stage 1
# output subfolders, as in your provided pupil detection code.
#
# If your current dataset is the Sph_-3.5 folder, replace the
# example path below with that actual dataset folder.
BASE_FOLDER = Path(
    )

ORIGINAL_FOLDER = BASE_FOLDER
ANNOTATED_FOLDER = BASE_FOLDER / "annotated"
MASK_FOLDER = BASE_FOLDER / "masks"
STAGE1_CSV = BASE_FOLDER / "csv" / "results.csv"

# Keep Stage 2 results separate.
OUTPUT_FOLDER = BASE_FOLDER / "stage2_glint"


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

    # A robust measure of the usual pupil brightness.
    pupil_median = float(np.median(pupil_values))

    # Adapt the threshold to this image's illumination.
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

    # Estimate the surrounding brightness at each pixel.
    # Small, sharp reflections stand out from this background.
    gray_float = gray.astype(np.float32)

    local_background = cv2.GaussianBlur(
        gray_float,
        (0, 0),
        sigmaX=BACKGROUND_SIGMA,
        sigmaY=BACKGROUND_SIGMA
    )

    local_contrast = gray_float - local_background

    # IMPORTANT: restrict every candidate to the eroded
    # pupil mask. Outside reflections cannot become candidates.
    candidate_pixels = (
        (gray_float >= bright_threshold) &
        (local_contrast >= MIN_LOCAL_CONTRAST) &
        (safe_mask > 0)
    )

    candidate_binary = candidate_pixels.astype(np.uint8)

    # Find separate connected bright components.
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

        # Additional validation against the original pupil mask.
        # The component must lie completely inside its safe region.
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

        # Ranking score: favour stronger local contrast and
        # brighter reflections. This is a heuristic, not a
        # guarantee that a candidate is the true corneal glint.
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

def process_image(image_path, csv_row):

    # Original image supplies real pixel brightness.
    gray = read_gray(image_path)

    # Start final display from the Stage 1 annotated image.
    annotated_path = (
        ANNOTATED_FOLDER /
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

        # CSV determines whether Stage 1 found this pupil.
        if not pupil_was_found(csv_row, side):
            row[f"{side}_status"] = "PUPIL_NOT_FOUND"
            continue

        mask_path = (
            MASK_FOLDER /
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

        # Detect a glint exclusively inside this pupil's safe mask.
        glint, candidate_count = detect_glint_inside_pupil(
            gray, pupil_mask, safe_mask
        )

        row[f"{side}_candidates"] = candidate_count

        if glint is None:
            row[f"{side}_status"] = "NO_VALID_GLINT"
            continue

        gx = int(round(glint["x"]))
        gy = int(round(glint["y"]))

        # Final safety check: the marker center must be inside the
        # eroded pupil mask, not outside the pupil boundary.
        if (
            gy < 0 or gy >= safe_mask.shape[0] or
            gx < 0 or gx >= safe_mask.shape[1] or
            safe_mask[gy, gx] == 0
        ):
            row[f"{side}_status"] = "CENTER_OUTSIDE_SAFE_MASK"
            continue

        # Draw the detected glint on the existing annotated image.
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

        # Record whether Stage 1 reported a fallback outline.
        # A fallback pupil boundary should be manually checked.
        outline_status = str(
            csv_row.get(f"{side}_outline", "")
        ).strip().lower()

        row[f"{side}_stage1_outline"] = (
            outline_status if outline_status else "not_recorded"
        )

    # Save final image with existing pupil outlines and glint markers.
    output_path = (
        OUTPUT_FOLDER / "images" /
        f"{image_path.stem}_glint.png"
    )

    cv2.imwrite(str(output_path), display_image)

    return row


# ============================================================
# 8. PROCESS ALL IMAGES AND SAVE CSV
# ============================================================

def main():

    OUTPUT_FOLDER.joinpath("images").mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT_FOLDER.joinpath("csv").mkdir(
        parents=True, exist_ok=True
    )

    stage1_rows = load_stage1_csv(STAGE1_CSV)

    supported_extensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
    }

    # Original files are expected directly in ORIGINAL_FOLDER.
    # Stage 1 outputs are stored in their own subfolders.
    image_files = sorted(
        path for path in ORIGINAL_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
        and not path.stem.endswith(
            ("_annotated", "_overlay", "_glint")
        )
        and "_mask_left" not in path.stem
        and "_mask_right" not in path.stem
    )

    if not image_files:
        print("ERROR: No original images found.")
        print(f"Check ORIGINAL_FOLDER: {ORIGINAL_FOLDER}")
        return

    rows = []

    for index, image_path in enumerate(image_files, start=1):

        csv_row = stage1_rows.get(image_path.name)

        if csv_row is None:
            result = {
                "image": image_path.name,
                "status": "IMAGE_NOT_FOUND_IN_STAGE1_CSV"
            }
        else:
            try:
                result = process_image(image_path, csv_row)
            except Exception as exc:
                result = {
                    "image": image_path.name,
                    "status": f"ERROR: {exc}"
                }

        rows.append(result)

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}: "
            f"L={result.get('left_status', 'CHECK')} | "
            f"R={result.get('right_status', 'CHECK')}"
        )

    # Create columns for every result field.
    columns = ["image"]

    for result in rows:
        for key in result:
            if key not in columns:
                columns.append(key)

    output_csv = (
        OUTPUT_FOLDER / "csv" / "glint_results.csv"
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

    print("\nStage 2 completed.")
    print(f"Annotated results: {OUTPUT_FOLDER / 'images'}")
    print(f"Glint CSV: {output_csv}")


if __name__ == "__main__":
    main()
