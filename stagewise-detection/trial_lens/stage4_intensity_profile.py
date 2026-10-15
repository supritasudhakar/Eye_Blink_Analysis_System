#!/usr/bin/env python3
"""
STAGE 4: INTENSITY PROFILE AND SLOPE ALONG THE STAGE 3 ANGLE

Inputs from the current pupil/glint/angle pipeline:
    Stage 1:
        masks/<image>_mask_left.png
        masks/<image>_mask_right.png
        csv/results.csv

    Stage 2:
        stage2_glint/csv/glint_results.csv

    Stage 3:
        stage3_angle/csv/angle_results.csv

    Original images in BASE_FOLDER.

Outputs:
    stage4_intensity_profile/
        graphs/<image>_left.png
        graphs/<image>_right.png
        angle_intensity_values.csv  # one row per averaged profile point
        slope_summary.csv           # one row per image/eye

Method follows the supplied reference code:
    - Use the Stage 3 angle directly.
    - Sample grayscale values along a line through the pupil.
    - Keep sample positions inside the saved Stage 1 pupil mask.
    - Average neighboring profile samples.
    - Fit one linear regression and report slope and R-squared.
    - Do NOT remove, ignore, or reject glint intensity values.

The detected glint coordinates are recorded for reference only. They are
NOT excluded from the profile, consistent with the supplied reference code.
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. PATHS -- EDIT BASE_FOLDER FOR YOUR DATASET
# ============================================================

BASE_FOLDER = Path(
    )

# Stage 1 outputs
ORIGINAL_FOLDER = BASE_FOLDER
MASK_FOLDER = BASE_FOLDER / "masks"
STAGE1_CSV = BASE_FOLDER / "csv" / "results.csv"

# Stage 2 output
STAGE2_CSV = BASE_FOLDER / "stage2_glint" / "csv" / "glint_results.csv"

# Stage 3 output
STAGE3_CSV = BASE_FOLDER / "stage3_angle" / "csv" / "angle_results.csv"

# Stage 4 outputs
OUTPUT_FOLDER = BASE_FOLDER / "stage4_intensity_profile"
GRAPH_FOLDER = OUTPUT_FOLDER / "graphs"
PROFILE_CSV = OUTPUT_FOLDER / "angle_intensity_values.csv"
SUMMARY_CSV = OUTPUT_FOLDER / "slope_summary.csv"


# ============================================================
# 2. PROFILE PARAMETERS
# ============================================================

# Number of positions sampled along the Stage 3 angle.
INTENSITY_PROFILE_SAMPLES = 101

# Scan this fraction of the pupil radius on either side of center.
# Example: 0.75 means scan from -0.75*r to +0.75*r.
INTENSITY_PROFILE_HALF_LENGTH_FRAC = 0.75

# Average each group of neighboring profile points.
# 5 means up to five neighboring samples become one profile value.
AVERAGE_PROFILE_PIXELS = 5

# Create graphs for left and right eyes separately.
SAVE_GRAPHS = True

# Optional extra smoothness is intentionally not applied: the supplied
# reference fits the slope to the averaged raw intensity values.

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def normalize_name(name):
    return Path(str(name).strip()).name.lower()


def read_csv_index(csv_path):
    """Return CSV rows indexed by normalized image filename."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV not found: {csv_path}")

    rows = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError(f"CSV must have an 'image' column: {csv_path}")
        for row in reader:
            name = normalize_name(row.get("image", ""))
            if name:
                rows[name] = row
    return rows


def get_float(row, field):
    if row is None:
        return None
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def read_gray(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if gray is None:
        raise ValueError(f"Cannot read image: {image_path}")

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


def get_original_images(folder):
    """Collect only original images located directly inside BASE_FOLDER."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.stem.lower().endswith(("_annotated", "_overlay", "_glint", "_angle"))
        and "_mask_left" not in p.stem.lower()
        and "_mask_right" not in p.stem.lower()
    )


# ============================================================
# 4. LOAD EXISTING PUPIL MASK AND GEOMETRY
# ============================================================

def load_pupil_mask(mask_path, image_shape, preferred_center=None):
    """Load Stage 1 mask and compute its pupil boundary and radius."""
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None
    if mask_img.shape != image_shape:
        raise ValueError(f"Mask/image size mismatch: {mask_path.name}")

    binary = (mask_img > 0).astype(np.uint8)
    if int(binary.sum()) < 20:
        return None

    contours, _ = cv2.findContours(
        binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(binary, binaryImage=True)
    if abs(moments["m00"]) < 1e-8:
        return None

    calculated_x = moments["m10"] / moments["m00"]
    calculated_y = moments["m01"] / moments["m00"]

    # Prefer Stage 3's recorded center so the sample line passes through
    # the same pupil center used when Stage 3 calculated the angle.
    if preferred_center and preferred_center[0] is not None and preferred_center[1] is not None:
        cx, cy = preferred_center
    else:
        cx, cy = calculated_x, calculated_y

    points = contour.reshape(-1, 2).astype(np.float64)
    radius_values = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    radius = float(np.median(radius_values)) if len(radius_values) else 0.0
    if radius <= 1.0:
        return None

    return {
        "mask": binary,
        "contour": contour,
        "cx": float(cx),
        "cy": float(cy),
        "radius": radius,
    }


# ============================================================
# 5. AVERAGE NEIGHBORING PROFILE POINTS
# ============================================================

def average_profile_points(distances, xs, ys, intensities, group_size):
    """Average each sequential group without removing bright glint samples."""
    if len(intensities) == 0:
        return {
            "distances": np.array([], dtype=float),
            "xs": np.array([], dtype=float),
            "ys": np.array([], dtype=float),
            "intensities": np.array([], dtype=float),
        }

    group_size = max(1, int(group_size))
    out_d, out_x, out_y, out_i = [], [], [], []

    for start in range(0, len(intensities), group_size):
        end = min(start + group_size, len(intensities))
        out_d.append(float(np.mean(distances[start:end])))
        out_x.append(float(np.mean(xs[start:end])))
        out_y.append(float(np.mean(ys[start:end])))
        out_i.append(float(np.mean(intensities[start:end])))

    return {
        "distances": np.asarray(out_d, dtype=np.float64),
        "xs": np.asarray(out_x, dtype=np.float64),
        "ys": np.asarray(out_y, dtype=np.float64),
        "intensities": np.asarray(out_i, dtype=np.float64),
    }


# ============================================================
# 6. EXTRACT PROFILE AND CALCULATE SLOPE
# ============================================================

def extract_intensity_profile(gray, pupil, angle_deg):
    """
    Sample original grayscale intensity along the Stage 3 angle.

    Glint handling intentionally follows the supplied reference code:
    no glint removal, no spike rejection, and no glint-based point filtering.
    The only spatial restriction is that sampled points must be inside the
    Stage 1 pupil mask and inside the image bounds.
    """
    if pupil is None or angle_deg is None:
        return None

    cx, cy, radius = pupil["cx"], pupil["cy"], pupil["radius"]
    if radius <= 1:
        return None

    angle_rad = math.radians(float(angle_deg))
    unit_x = math.cos(angle_rad)
    unit_y = math.sin(angle_rad)

    half_length = INTENSITY_PROFILE_HALF_LENGTH_FRAC * radius
    distances = np.linspace(
        -half_length,
        half_length,
        INTENSITY_PROFILE_SAMPLES
    )

    xs = cx + distances * unit_x
    ys = cy + distances * unit_y

    height, width = gray.shape[:2]
    image_valid = (
        (xs >= 0) & (xs <= width - 1) &
        (ys >= 0) & (ys <= height - 1)
    )
    distances, xs, ys = distances[image_valid], xs[image_valid], ys[image_valid]

    if len(xs) < 3:
        return None

    # Restrict profile positions to the saved pupil region.
    # This does not filter pixels by brightness: a glint inside the mask stays.
    xi = np.clip(np.rint(xs).astype(int), 0, width - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, height - 1)
    in_pupil = pupil["mask"][yi, xi] > 0

    distances, xs, ys = distances[in_pupil], xs[in_pupil], ys[in_pupil]
    if len(xs) < 3:
        return None

    # cv2.remap samples sub-pixel grayscale intensities from the original image.
    # No glint removal or bright-spike rejection is performed.
    intensities = cv2.remap(
        gray,
        xs.astype(np.float32).reshape(1, -1),
        ys.astype(np.float32).reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float64).reshape(-1)

    if len(intensities) < 3:
        return None

    averaged = average_profile_points(
        distances, xs, ys, intensities, AVERAGE_PROFILE_PIXELS
    )

    distances = averaged["distances"]
    xs = averaged["xs"]
    ys = averaged["ys"]
    intensities = averaged["intensities"]

    if len(intensities) < 2:
        return None

    dark_index = int(np.argmin(intensities))
    bright_index = int(np.argmax(intensities))

    # Least-squares linear fit: intensity = slope * distance + intercept.
    slope, intercept = np.polyfit(distances, intensities, 1)
    fitted = slope * distances + intercept

    ss_res = float(np.sum((intensities - fitted) ** 2))
    ss_tot = float(np.sum((intensities - np.mean(intensities)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    return {
        "angle_deg": float(angle_deg),
        "distances": distances,
        "xs": xs,
        "ys": ys,
        "intensities": intensities,
        "dark_index": dark_index,
        "bright_index": bright_index,
        "dark_intensity": float(intensities[dark_index]),
        "bright_intensity": float(intensities[bright_index]),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "sample_count": int(len(intensities)),
        "pupil_center_x": float(cx),
        "pupil_center_y": float(cy),
        "pupil_radius_px": float(radius),
    }


# ============================================================
# 7. SAVE PROFILE GRAPH
# ============================================================

def save_graph(profile, output_path, image_name, eye):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distances = profile["distances"]
    intensities = profile["intensities"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        distances,
        intensities,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Intensity profile (glint included)",
    )

    dark_i = profile["dark_index"]
    bright_i = profile["bright_index"]

    plt.scatter(
        [distances[dark_i]], [intensities[dark_i]], s=70,
        label=f"Dark point = {intensities[dark_i]:.1f}",
    )
    plt.scatter(
        [distances[bright_i]], [intensities[bright_i]], s=70,
        label=f"Bright point = {intensities[bright_i]:.1f}",
    )

    # Draw the overall best-fit regression line.
    order = np.argsort(distances)
    sorted_distances = distances[order]
    fitted = profile["slope"] * sorted_distances + profile["intercept"]
    plt.plot(
        sorted_distances,
        fitted,
        linewidth=2,
        label=f"Overall slope = {profile['slope']:.4f}",
    )

    plt.xlabel("Distance along Stage 3 angle (pixels)")
    plt.ylabel("Grayscale intensity (0-255)")
    plt.title(
        f"{eye.upper()} Eye Intensity Profile\n"
        f"{image_name} | Angle = {profile['angle_deg']:.2f} deg\n"
        f"Slope = {profile['slope']:.4f} intensity/pixel | R² = {profile['r2']:.4f}"
    )
    plt.ylim(0, 255)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


# ============================================================
# 8. RUN STAGE 4 FOR THE DATASET
# ============================================================

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    if SAVE_GRAPHS:
        GRAPH_FOLDER.mkdir(parents=True, exist_ok=True)

    for required in (STAGE1_CSV, STAGE2_CSV, STAGE3_CSV):
        if not required.exists():
            raise FileNotFoundError(
                f"Required input file not found:\n{required}\n"
                "Check BASE_FOLDER and confirm Stage 1, 2, and 3 have run."
            )

    stage1_rows = read_csv_index(STAGE1_CSV)
    stage2_rows = read_csv_index(STAGE2_CSV)
    stage3_rows = read_csv_index(STAGE3_CSV)

    image_files = get_original_images(ORIGINAL_FOLDER)
    image_index = {normalize_name(path.name): path for path in image_files}

    print("=" * 78)
    print("STAGE 4 - INTENSITY PROFILE AND SLOPE")
    print(f"Dataset: {BASE_FOLDER}")
    print(f"Original images found: {len(image_files)}")
    print(f"Stage 1 CSV rows: {len(stage1_rows)}")
    print(f"Stage 2 glint CSV rows: {len(stage2_rows)}")
    print(f"Stage 3 angle CSV rows: {len(stage3_rows)}")
    print("Glint intensity is included; no glint removal or spike rejection.")
    print("=" * 78)

    profile_rows = []
    summary_rows = []

    # Stage 3 angle rows define which images and eye angles are processed.
    for index, name_key in enumerate(sorted(stage3_rows), start=1):
        stage3_row = stage3_rows[name_key]
        stage1_row = stage1_rows.get(name_key)
        stage2_row = stage2_rows.get(name_key)
        image_path = image_index.get(name_key)

        if image_path is None:
            print(f"[{index}/{len(stage3_rows)}] {name_key}: ORIGINAL_IMAGE_NOT_FOUND")
            for side in ("left", "right"):
                summary_rows.append({
                    "image": name_key,
                    "eye": side,
                    "status": "ORIGINAL_IMAGE_NOT_FOUND",
                })
            continue

        try:
            gray = read_gray(image_path)
        except Exception as exc:
            print(f"[{index}/{len(stage3_rows)}] {image_path.name}: ERROR {exc}")
            for side in ("left", "right"):
                summary_rows.append({
                    "image": image_path.name,
                    "eye": side,
                    "status": f"IMAGE_READ_ERROR: {exc}",
                })
            continue

        print(f"[{index}/{len(stage3_rows)}] {image_path.name}")

        for side in ("left", "right"):
            angle = get_float(stage3_row, f"{side}_angle_deg")

            mask_path = MASK_FOLDER / f"{image_path.stem}_mask_{side}.png"

            base_record = {
                "image": image_path.name,
                "source_image": str(image_path),
                "eye": side,
                "angle_deg": angle if angle is not None else "",
                "glint_status": (stage2_row or {}).get(f"{side}_status", ""),
                "glint_x": get_float(stage2_row, f"{side}_x") if stage2_row else None,
                "glint_y": get_float(stage2_row, f"{side}_y") if stage2_row else None,
            }

            if stage1_row is None:
                summary_rows.append({**base_record, "status": "STAGE1_CSV_ROW_MISSING"})
                print(f"    {side.upper()}: Stage 1 CSV row missing")
                continue

            if angle is None:
                summary_rows.append({**base_record, "status": "ANGLE_MISSING"})
                print(f"    {side.upper()}: ANGLE_MISSING")
                continue

            if not mask_path.exists():
                summary_rows.append({**base_record, "status": "PUPIL_MASK_NOT_FOUND"})
                print(f"    {side.upper()}: PUPIL_MASK_NOT_FOUND")
                continue

            center_x = get_float(stage3_row, f"{side}_center_x")
            center_y = get_float(stage3_row, f"{side}_center_y")
            preferred_center = (center_x, center_y) if center_x is not None and center_y is not None else None

            try:
                pupil = load_pupil_mask(mask_path, gray.shape, preferred_center)
                if pupil is None:
                    summary_rows.append({**base_record, "status": "EMPTY_OR_INVALID_MASK"})
                    print(f"    {side.upper()}: EMPTY_OR_INVALID_MASK")
                    continue

                profile = extract_intensity_profile(gray, pupil, angle)
                if profile is None:
                    summary_rows.append({**base_record, "status": "PROFILE_EXTRACTION_FAILED"})
                    print(f"    {side.upper()}: PROFILE_EXTRACTION_FAILED")
                    continue

                graph_path = GRAPH_FOLDER / f"{image_path.stem}_{side}.png"
                if SAVE_GRAPHS:
                    save_graph(profile, graph_path, image_path.name, side)

                summary = {
                    **base_record,
                    "status": "OK",
                    "pupil_center_x": round(profile["pupil_center_x"], 4),
                    "pupil_center_y": round(profile["pupil_center_y"], 4),
                    "pupil_radius_px": round(profile["pupil_radius_px"], 4),
                    "sample_count": profile["sample_count"],
                    "dark_intensity": round(profile["dark_intensity"], 4),
                    "bright_intensity": round(profile["bright_intensity"], 4),
                    "slope": round(profile["slope"], 8),
                    "r2": round(profile["r2"], 8),
                    "graph_path": str(graph_path) if SAVE_GRAPHS else "",
                }
                summary_rows.append(summary)

                # One CSV record per averaged profile point. The slope and R²
                # are repeated for each point, matching the supplied code style.
                for point_index in range(len(profile["distances"])):
                    profile_rows.append({
                        "image": image_path.name,
                        "source_image": str(image_path),
                        "eye": side,
                        "angle_deg": f"{profile['angle_deg']:.4f}",
                        "point_index": point_index,
                        "distance_px": f"{profile['distances'][point_index]:.4f}",
                        "x": f"{profile['xs'][point_index]:.4f}",
                        "y": f"{profile['ys'][point_index]:.4f}",
                        "intensity": f"{profile['intensities'][point_index]:.4f}",
                        "dark_intensity": f"{profile['dark_intensity']:.4f}",
                        "bright_intensity": f"{profile['bright_intensity']:.4f}",
                        "slope": f"{profile['slope']:.8f}",
                        "intercept": f"{profile['intercept']:.8f}",
                        "r2": f"{profile['r2']:.8f}",
                        "glint_status": base_record["glint_status"],
                        "glint_x": base_record["glint_x"] if base_record["glint_x"] is not None else "",
                        "glint_y": base_record["glint_y"] if base_record["glint_y"] is not None else "",
                    })

                print(
                    f"    {side.upper()}: angle={angle:.2f} deg, "
                    f"slope={profile['slope']:.6f}, R²={profile['r2']:.4f}, "
                    f"points={profile['sample_count']}"
                )

            except Exception as exc:
                summary_rows.append({
                    **base_record,
                    "status": f"ERROR: {type(exc).__name__}: {exc}",
                })
                print(f"    {side.upper()}: ERROR {type(exc).__name__}: {exc}")

    # Save per-profile-point data.
    profile_fields = [
        "image", "source_image", "eye", "angle_deg", "point_index",
        "distance_px", "x", "y", "intensity", "dark_intensity",
        "bright_intensity", "slope", "intercept", "r2",
        "glint_status", "glint_x", "glint_y",
    ]
    with open(PROFILE_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=profile_fields)
        writer.writeheader()
        writer.writerows(profile_rows)

    # Save compact per-image/per-eye summary, including failures.
    summary_fields = [
        "image", "source_image", "eye", "status", "angle_deg",
        "glint_status", "glint_x", "glint_y", "pupil_center_x",
        "pupil_center_y", "pupil_radius_px", "sample_count",
        "dark_intensity", "bright_intensity", "slope", "r2", "graph_path",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    success_count = sum(1 for row in summary_rows if row.get("status") == "OK")

    print("\n" + "=" * 78)
    print("STAGE 4 COMPLETED")
    print(f"Successful image-eye profiles: {success_count}/{len(summary_rows)}")
    print(f"Graphs: {GRAPH_FOLDER}")
    print(f"Profile-point CSV: {PROFILE_CSV}")
    print(f"Slope summary CSV: {SUMMARY_CSV}")
    print("Glint samples were NOT removed from the intensity profile.")
    print("=" * 78)


if __name__ == "__main__":
    main()
