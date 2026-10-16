#!/usr/bin/env python3
"""
STAGE 5: AVERAGE PROFILE POINTS AND FIT ONE STRAIGHT LINE

This version matches the requested graph style:
  - averaged points only (no raw-profile curve)
  - one least-squares straight line fitted to the averaged points
  - darkest and brightest averaged points highlighted
  - angle, number of ignored glint points, averaging size, and R-squared
    displayed in the graph title

INPUT:
  <dataset>/stage4_intensity_profile/angle_intensity_values.csv

EXPECTED CSV FIELDS include:
  image, eye, angle_deg, distance_px, intensity

OUTPUT, saved inside each dataset:
  stage5_average_fit/
      graphs/<image>_<eye>_average_fit.png
      average_5_pixel_points.csv
      stage5_fit_information.csv
      stage5_processing_status.csv

Important:
  This follows the supplied Stage 5 reference: it reads the Stage 4 profile
  values, detects and excludes a bright spike region, averages exactly five
  consecutive valid Stage 4 rows at a time, then fits a line to those averaged
  points. Averaging groups never cross an ignored glint region. Incomplete
  groups of fewer than five points are skipped.
"""

import csv
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 2. FOLDER / FILE NAMES
# ============================================================
STAGE4_FOLDER_NAME = "stage4_intensity_profile"
STAGE4_CSV_NAME = "angle_intensity_values.csv"
STAGE5_FOLDER_NAME = "stage5_average_fit"
GRAPH_FOLDER_NAME = "graphs"
AVERAGE_POINTS_CSV = "average_5_pixel_points.csv"
FIT_INFORMATION_CSV = "stage5_fit_information.csv"
STATUS_CSV = "stage5_processing_status.csv"


# ============================================================
# 3. PROFILE AVERAGING AND GLINT-SPIKE PARAMETERS
# ============================================================
# Exactly five adjacent Stage 4 rows -> one Stage 5 average point.
AVERAGE_PROFILE_PIXELS = 5

# A potential glint spike must reach at least this intensity.
MIN_BRIGHT_VALUE = 100.0

# The point must rise this many intensity levels above its neighbours.
GLINT_SPIKE_MIN_RISE = 20.0

# Number of values inspected on each side of a potential spike.
GLINT_SPIKE_NEIGHBOR_POINTS = 2

# Expand the ignored glint region by this many Stage 4 profile samples.
GLINT_IGNORE_PADDING = 2


# ============================================================
# 4. GRAPH SETTINGS
# ============================================================
FIGURE_SIZE = (12, 7)
GRAPH_DPI = 160
Y_AXIS_MIN = 0
Y_AXIS_MAX = 255


# ============================================================
# 5. GENERAL HELPERS
# ============================================================
def find_column(fieldnames, candidates):
    """Find a CSV column regardless of spaces/capitalization."""
    if not fieldnames:
        return None
    lookup = {
        str(name).strip().lower().replace(" ", "_"): name
        for name in fieldnames
    }
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None


def to_float(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def normalize_eye(value):
    value = str(value or "").strip().lower()
    if value in {"left", "le", "l", "left_eye", "left eye"}:
        return "left"
    if value in {"right", "re", "r", "right_eye", "right eye"}:
        return "right"
    return ""


def normalize_image_name(value):
    return Path(str(value or "").strip()).name


def find_stage4_csv_in_path(folder_path):
    """Find Stage 4 CSV files given a specific input path."""
    found = []
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return found

    # Check direct location
    direct_csv = folder_path / STAGE4_FOLDER_NAME / STAGE4_CSV_NAME
    if direct_csv.is_file():
        found.append(direct_csv)
    elif folder_path.name.lower() == STAGE4_FOLDER_NAME.lower() and (folder_path / STAGE4_CSV_NAME).is_file():
        found.append(folder_path / STAGE4_CSV_NAME)
    else:
        # Search recursively inside the folder
        for current_root, dirs, files in os.walk(folder_path):
            dirs[:] = [
                name for name in dirs
                if name.lower() != STAGE5_FOLDER_NAME.lower()
            ]
            if Path(current_root).name.lower() != STAGE4_FOLDER_NAME.lower():
                continue
            for name in files:
                if name.lower() == STAGE4_CSV_NAME.lower():
                    found.append(Path(current_root) / name)
        
        if not found:
            # Fallback search for angle_intensity_values.csv anywhere
            for path in folder_path.glob(f"**/{STAGE4_CSV_NAME}"):
                found.append(path)

    return sorted(set(found))


def read_stage4_csv(csv_path):
    """Read Stage 4 samples and group by (image filename, eye)."""
    groups = {}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []

        image_col = find_column(fields, ["image", "source_image", "filename", "file"])
        eye_col = find_column(fields, ["eye", "side"])
        angle_col = find_column(fields, ["angle_deg", "profile_angle_deg", "common_angle_deg", "angle"])
        distance_col = find_column(fields, ["distance_px", "distance", "profile_distance", "x_distance", "position"])
        intensity_col = find_column(fields, ["intensity", "gray_intensity", "grayscale_intensity", "average_intensity"])

        missing = []
        if image_col is None:
            missing.append("image")
        if eye_col is None:
            missing.append("eye")
        if distance_col is None:
            missing.append("distance_px")
        if intensity_col is None:
            missing.append("intensity")
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(missing)}. "
                f"Found: {fields}"
            )

        for row in reader:
            image_name = normalize_image_name(row.get(image_col))
            eye = normalize_eye(row.get(eye_col))
            distance = to_float(row.get(distance_col))
            intensity = to_float(row.get(intensity_col))
            angle = to_float(row.get(angle_col)) if angle_col else None

            if not image_name or eye not in {"left", "right"}:
                continue
            if distance is None or intensity is None:
                continue

            key = (image_name.lower(), eye)
            if key not in groups:
                groups[key] = {
                    "image": image_name,
                    "eye": eye,
                    "distances": [],
                    "intensities": [],
                    "angles": [],
                }
            groups[key]["distances"].append(distance)
            groups[key]["intensities"].append(intensity)
            if angle is not None:
                groups[key]["angles"].append(angle)

    for record in groups.values():
        order = np.argsort(np.asarray(record["distances"], dtype=float))
        record["distances"] = np.asarray(record["distances"], dtype=float)[order]
        record["intensities"] = np.asarray(record["intensities"], dtype=float)[order]
        record["angle"] = float(np.median(record["angles"])) if record["angles"] else None

    return groups


# ============================================================
# 6. DETECT BRIGHT SPIKE / GLINT ALONG THE PROFILE
# ============================================================
def is_bright_profile_spike(intensities, index):
    """True when a bright sample sharply rises above nearby samples."""
    n = len(intensities)
    k = GLINT_SPIKE_NEIGHBOR_POINTS
    if index < k or index >= n - k:
        return False

    current = float(intensities[index])
    neighbours = np.concatenate([
        intensities[index - k:index],
        intensities[index + 1:index + 1 + k],
    ])
    if neighbours.size == 0:
        return False

    local_median = float(np.median(neighbours))
    return (
        current >= MIN_BRIGHT_VALUE
        and current - local_median >= GLINT_SPIKE_MIN_RISE
    )


def detect_glint_region(intensities):
    """Return ignored mask, glint start/end, and a diagnostic status."""
    values = np.asarray(intensities, dtype=np.float64).reshape(-1)
    ignored = np.zeros(len(values), dtype=bool)
    k = GLINT_SPIKE_NEIGHBOR_POINTS

    if len(values) < 2 * k + 1:
        return ignored, None, None, "PROFILE_TOO_SHORT_FOR_SPIKE_CHECK"

    candidate_indices = [
        i for i in range(len(values))
        if is_bright_profile_spike(values, i)
    ]
    if not candidate_indices:
        return ignored, None, None, "NO_BRIGHT_SPIKE_FOUND"

    candidate_array = np.asarray(candidate_indices, dtype=int)
    breaks = np.where(np.diff(candidate_array) > 1)[0] + 1
    groups = np.split(candidate_array, breaks)
    best_group = max(groups, key=lambda group: float(np.mean(values[group])))

    start = max(0, int(best_group[0]) - GLINT_IGNORE_PADDING)
    end = min(len(values) - 1, int(best_group[-1]) + GLINT_IGNORE_PADDING)
    ignored[start:end + 1] = True
    return ignored, start, end, "BRIGHT_SPIKE_IGNORED"


# ============================================================
# 7. AVERAGE EXACTLY FIVE CONSECUTIVE VALID PROFILE ROWS
# ============================================================
def average_valid_profile_points(distances, intensities, ignored):
    """Average complete five-point groups; never cross an ignored glint gap."""
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    ignored = np.asarray(ignored, dtype=bool).reshape(-1)

    if not (len(distances) == len(intensities) == len(ignored)):
        raise ValueError("Distance, intensity and ignored arrays must have equal lengths.")

    valid_indices = np.flatnonzero(~ignored)
    if len(valid_indices) == 0:
        return []

    split_at = np.where(np.diff(valid_indices) != 1)[0] + 1
    runs = np.split(valid_indices, split_at)

    averaged = []
    for run in runs:
        for start in range(0, len(run), AVERAGE_PROFILE_PIXELS):
            group = run[start:start + AVERAGE_PROFILE_PIXELS]
            if len(group) != AVERAGE_PROFILE_PIXELS:
                continue

            averaged.append({
                "average_distance_px": float(np.mean(distances[group])),
                "average_intensity": float(np.mean(intensities[group])),
                "pixels_used": AVERAGE_PROFILE_PIXELS,
                "profile_start_index": int(group[0]),
                "profile_end_index": int(group[-1]),
            })

    return averaged


# ============================================================
# 8. STRAIGHT-LINE FIT: y = slope*x + intercept
# ============================================================
def fit_average_line(average_points):
    """Least-squares fit to averaged points and calculate R-squared."""
    if len(average_points) < 2:
        return None

    x = np.asarray([p["average_distance_px"] for p in average_points], dtype=float)
    y = np.asarray([p["average_intensity"] for p in average_points], dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or len(np.unique(x)) < 2:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "point_count": int(len(x)),
    }


# ============================================================
# 9. GRAPH IN THE SAME STYLE AS THE USER'S EXAMPLE
# ============================================================
def save_average_fit_graph(average_points, fit, output_path, image_name,
                           eye, angle, ignored_count, spike_status):
    """Plot averaged points, the fitted line, and the dark/bright points."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray([p["average_distance_px"] for p in average_points], dtype=float)
    y = np.asarray([p["average_intensity"] for p in average_points], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 7))

    if len(x):
        ax.scatter(
            x, y, s=60, marker="o",
            label=f"Averaged intensity ({AVERAGE_PROFILE_PIXELS} pixels)",
            zorder=3,
        )

        dark_index = int(np.argmin(y))
        bright_index = int(np.argmax(y))

        ax.scatter(
            [x[dark_index]], [y[dark_index]], s=90,
            label=f"Dark = {y[dark_index]:.1f}", zorder=4,
        )
        ax.scatter(
            [x[bright_index]], [y[bright_index]], s=90,
            label=f"Bright = {y[bright_index]:.1f}", zorder=4,
        )

        if fit is not None:
            line_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            line_y = fit["slope"] * line_x + fit["intercept"]
            ax.plot(
                line_x,
                line_y,
                linewidth=2.0,
                label=(
                    f"Linear fit: y = {fit['slope']:.4f}x "
                    f"+ {fit['intercept']:.4f}"
                ),
                zorder=2,
            )

    if fit is not None:
        r2 = fit["r_squared"]
        ax.plot([], [], linestyle="none", label=f"R² = {r2:.4f}")
        fit_text = f"R² = {r2:.4f}"
    else:
        fit_text = "Straight-line fit unavailable (need at least 2 averaged points)"

    angle_text = f"{angle:.1f}°" if angle is not None else "N/A"
    ax.set_title(
        f"{eye.capitalize()} eye | Angle = {angle_text} | "
        f"Glint points ignored = {ignored_count} | "
        f"Average = {AVERAGE_PROFILE_PIXELS} pixels | {fit_text}"
    )
    ax.set_xlabel(
        "Distance along detected angle direction (pixels)\n"
        "Dark side → Bright side"
    )
    ax.set_ylabel("Average grayscale intensity (0–255)")
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=GRAPH_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 10. PROCESS ONE IMAGE + EYE
# ============================================================
def process_profile(profile, dataset_name, average_rows, fit_rows, status_rows,
                    graph_folder):
    image_name = profile["image"]
    eye = profile["eye"]
    distances = profile["distances"]
    intensities = profile["intensities"]
    angle = profile["angle"]

    ignored, glint_start, glint_end, glint_status = detect_glint_region(intensities)
    ignored_count = int(np.count_nonzero(ignored))
    averaged = average_valid_profile_points(distances, intensities, ignored)
    fit = fit_average_line(averaged)

    graph_path = graph_folder / f"{Path(image_name).stem}_{eye}_average_fit.png"
    save_average_fit_graph(
        averaged, fit, graph_path, image_name, eye, angle,
        ignored_count, glint_status,
    )

    status = "OK" if fit is not None else "INSUFFICIENT_AVERAGED_POINTS"
    status_rows.append({
        "image": image_name,
        "eye": eye,
        "profile_samples": len(intensities),
        "angle_deg": "" if angle is None else f"{angle:.4f}",
        "glint_detection_status": glint_status,
        "glint_start_index": "" if glint_start is None else glint_start,
        "glint_end_index": "" if glint_end is None else glint_end,
        "glint_points_ignored": ignored_count,
        "averaged_points": len(averaged),
        "fit_status": status,
        "graph_path": str(graph_path),
    })

    for index, point in enumerate(averaged):
        average_rows.append({
            "dataset": dataset_name,
            "image": image_name,
            "eye": eye,
            "angle_deg": "" if angle is None else f"{angle:.4f}",
            "average_index": index,
            "average_distance_px": f"{point['average_distance_px']:.5f}",
            "average_intensity": f"{point['average_intensity']:.5f}",
            "pixels_used": point["pixels_used"],
            "profile_start_index": point["profile_start_index"],
            "profile_end_index": point["profile_end_index"],
            "glint_points_ignored": ignored_count,
        })

    if fit is not None:
        y = np.asarray([p["average_intensity"] for p in averaged], dtype=float)
        fit_rows.append({
            "dataset": dataset_name,
            "image": image_name,
            "eye": eye,
            "angle_deg": "" if angle is None else f"{angle:.4f}",
            "average_points": fit["point_count"],
            "average_pixels_per_point": AVERAGE_PROFILE_PIXELS,
            "glint_detection_status": glint_status,
            "glint_points_ignored": ignored_count,
            "dark_intensity": f"{float(np.min(y)):.4f}",
            "bright_intensity": f"{float(np.max(y)):.4f}",
            "fit_slope": f"{fit['slope']:.8f}",
            "fit_intercept": f"{fit['intercept']:.8f}",
            "r_squared": f"{fit['r_squared']:.8f}",
            "graph_path": str(graph_path),
        })
        print(
            f"[OK] {eye.upper():5s} {image_name} | "
            f"points={fit['point_count']} | slope={fit['slope']:.6f} | "
            f"R²={fit['r_squared']:.4f} | glint ignored={ignored_count}"
        )
    else:
        print(
            f"[SKIP FIT] {eye.upper():5s} {image_name} | "
            f"averaged points={len(averaged)} | glint status={glint_status}. "
            f"Diagnostic graph saved."
        )


# ============================================================
# 11. SAVE CSV FILES
# ============================================================
def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 12. MAIN (INTERACTIVE TERMINAL INPUT)
# ============================================================
def main():
    print()
    print("=" * 80)
    print("STAGE 5 - AVERAGE PROFILE POINTS AND FIT (INTERACTIVE INPUT)")
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

    print()
    print(f"Averaging: exactly {AVERAGE_PROFILE_PIXELS} consecutive valid Stage 4 rows")
    print("Glint handling: detect bright spike, ignore its padded region, continue after it")
    print("Fit: least-squares line y = slope*x + intercept, using averaged points only")

    stage4_csvs = []
    for folder in input_folders:
        stage4_csvs.extend(find_stage4_csv_in_path(folder))

    stage4_csvs = sorted(list(set(stage4_csvs)))

    if not stage4_csvs:
        print("[ERROR] No stage4_intensity_profile/angle_intensity_values.csv files found in the specified paths.")
        return

    print(f"Stage 4 CSV files found: {len(stage4_csvs)}")

    total_profiles = 0
    successful_fits = 0

    for stage4_csv in stage4_csvs:
        dataset_folder = stage4_csv.parent.parent
        dataset_name = dataset_folder.name
        output_folder = dataset_folder / STAGE5_FOLDER_NAME
        graph_folder = output_folder / GRAPH_FOLDER_NAME
        graph_folder.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 78)
        print(f"Dataset: {dataset_name}")
        print(f"Stage 4 CSV: {stage4_csv}")

        try:
            groups = read_stage4_csv(stage4_csv)
        except Exception as exc:
            print(f"[ERROR] Could not read {stage4_csv}: {type(exc).__name__}: {exc}")
            continue

        if not groups:
            print("[WARNING] No valid image/eye profiles found in this CSV.")
            continue

        average_rows = []
        fit_rows = []
        status_rows = []

        for key in sorted(groups.keys()):
            total_profiles += 1
            process_profile(
                groups[key], dataset_name,
                average_rows, fit_rows, status_rows,
                graph_folder,
            )

        save_csv(
            output_folder / AVERAGE_POINTS_CSV,
            average_rows,
            [
                "dataset", "image", "eye", "angle_deg", "average_index",
                "average_distance_px", "average_intensity", "pixels_used",
                "profile_start_index", "profile_end_index", "glint_points_ignored",
            ],
        )
        save_csv(
            output_folder / FIT_INFORMATION_CSV,
            fit_rows,
            [
                "dataset", "image", "eye", "angle_deg", "average_points",
                "average_pixels_per_point", "glint_detection_status",
                "glint_points_ignored", "dark_intensity", "bright_intensity",
                "fit_slope", "fit_intercept", "r_squared", "graph_path",
            ],
        )
        save_csv(
            output_folder / STATUS_CSV,
            status_rows,
            [
                "image", "eye", "profile_samples", "angle_deg",
                "glint_detection_status", "glint_start_index", "glint_end_index",
                "glint_points_ignored", "averaged_points", "fit_status", "graph_path",
            ],
        )

        successful_fits += len(fit_rows)
        print(f"Saved graphs: {graph_folder}")
        print(f"Saved average points: {output_folder / AVERAGE_POINTS_CSV}")
        print(f"Saved fit information: {output_folder / FIT_INFORMATION_CSV}")
        print(f"Saved processing status: {output_folder / STATUS_CSV}")

    print("\n" + "=" * 78)
    print("STAGE 5 COMPLETED")
    print(f"Profiles processed: {total_profiles}")
    print(f"Successful straight-line fits: {successful_fits}")
    print("=" * 78)


if __name__ == "__main__":
    main()