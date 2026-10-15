#!/usr/bin/env python3
"""
STAGE 6 - FINAL SLOPE DETECTION

Based on the user's Stage 6 reference code, adapted to the current pipeline.

INPUT per dataset:
    stage5_average_fit/average_5_pixel_points.csv

The CSV must contain:
    image, eye, average_distance_px, average_intensity
and preferably angle_deg.

WHAT THIS SCRIPT DOES:
    1. Reads Stage 5 averaged profile points.
    2. Groups the points by image and eye.
    3. Fits y = slope*x + intercept using linear regression.
    4. Calculates R-squared (R²).
    5. Saves individual profile slope results.
    6. Summarizes mean/median/min/max/std slope for each dataset and eye.
    7. Creates left/right slope comparison graphs across input datasets.
    8. Creates left/right slope-vs-angle graphs.

Stage 5 is responsible for finding/ignoring the glint spike and averaging
five consecutive valid samples. Stage 6 uses those averaged points and does
not run pupil, glint, or intensity-profile detection again.

No YOLO, ML, Haar, or model files are used.
"""

import csv
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. ROOT FOLDER
# ============================================================
# This is the parent folder containing your lens datasets.
# Update this if your datasets are stored somewhere else.
ROOT_HR_FOLDER = Path(
    
)


# ============================================================
# 2. INPUT PATHS
# ============================================================
# You can supply dataset folders here. The old reference code used paths
# ending in '\\annotated'; this version supports either the dataset folder
# itself OR its 'annotated' subfolder.
#
# The list below contains the seven dataset paths from your reference code.
# Each entry can point to the dataset folder OR its annotated subfolder.
# To add another dataset, add one more ROOT_HR_FOLDER / "folder_name" entry.
# To automatically discover ALL datasets instead, set INPUT_PATHS = [].
INPUT_PATHS = [
    # Dataset 1
    ROOT_HR_FOLDER / "" 
]


# ============================================================
# 3. FILE / OUTPUT NAMES
# ============================================================
STAGE5_FOLDER_NAME = "stage5_average_fit"
STAGE5_AVERAGE_CSV = "average_5_pixel_points.csv"

OUTPUT_FOLDER_NAME = "stage6_slope_detection"
GRAPH_FOLDER_NAME = "graphs"
SLOPE_CSV_NAME = "slope_results.csv"
SUMMARY_CSV_NAME = "slope_summary.csv"
STATUS_CSV_NAME = "slope_processing_status.csv"

AVERAGE_PIXELS = 5  # Stage 5 should have averaged exactly five valid samples.
FIGURE_SIZE = (12, 7)
GRAPH_DPI = 160


# ============================================================
# 4. SMALL HELPERS
# ============================================================
def normalize_name(value):
    return Path(str(value or "").strip()).name.lower()


def find_column(fieldnames, possible_names):
    """Find CSV columns despite capitalization or spaces."""
    if not fieldnames:
        return None
    normalized = {
        str(field).strip().lower().replace(" ", "_"): field
        for field in fieldnames
    }
    for name in possible_names:
        key = name.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def normalize_eye(value):
    value = str(value or "").strip().lower()
    if value in {"left", "le", "l", "left_eye", "left eye"}:
        return "left"
    if value in {"right", "re", "r", "right_eye", "right eye"}:
        return "right"
    return ""


def to_float(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def resolve_dataset_folder(input_path):
    """Accept either the dataset folder or its annotated subfolder."""
    path = Path(input_path).expanduser()
    if path.name.lower() == "annotated":
        parent = path.parent
        if (parent / STAGE5_FOLDER_NAME / STAGE5_AVERAGE_CSV).is_file():
            return parent
    if (path / STAGE5_FOLDER_NAME / STAGE5_AVERAGE_CSV).is_file():
        return path
    # Recursive fallback supports folder layouts where stage output is nested.
    if path.exists():
        found = sorted(path.rglob(STAGE5_AVERAGE_CSV))
        found = [p for p in found if p.parent.name.lower() == STAGE5_FOLDER_NAME.lower()
                 and OUTPUT_FOLDER_NAME.lower() not in {part.lower() for part in p.parts}]
        if found:
            return found[0].parent.parent
    # If caller supplied an annotated directory, search its parent too.
    if path.name.lower() == "annotated" and path.parent.exists():
        found = sorted(path.parent.rglob(STAGE5_AVERAGE_CSV))
        found = [p for p in found if p.parent.name.lower() == STAGE5_FOLDER_NAME.lower()
                 and OUTPUT_FOLDER_NAME.lower() not in {part.lower() for part in p.parts}]
        if found:
            return found[0].parent.parent
    return path


def find_stage5_csv(dataset_folder):
    """Locate Stage 5 average points CSV within the selected dataset."""
    dataset_folder = Path(dataset_folder)
    direct = dataset_folder / STAGE5_FOLDER_NAME / STAGE5_AVERAGE_CSV
    if direct.is_file():
        return direct

    if not dataset_folder.exists():
        return None

    matches = sorted(dataset_folder.rglob(STAGE5_AVERAGE_CSV))
    matches = [p for p in matches
               if p.parent.name.lower() == STAGE5_FOLDER_NAME.lower()
               and OUTPUT_FOLDER_NAME.lower() not in {part.lower() for part in p.parts}]
    return matches[0] if matches else None


def discover_datasets(root_folder):
    """Find every dataset folder containing the expected Stage 5 CSV."""
    root_folder = Path(root_folder)
    if not root_folder.exists():
        return []
    matches = []
    for csv_path in root_folder.rglob(STAGE5_AVERAGE_CSV):
        parts_lower = {part.lower() for part in csv_path.parts}
        if OUTPUT_FOLDER_NAME.lower() in parts_lower:
            continue
        if csv_path.parent.name.lower() != STAGE5_FOLDER_NAME.lower():
            continue
        matches.append(csv_path.parent.parent)
    # Unique, stable order
    unique = {str(p.resolve()): p for p in matches}
    return [unique[k] for k in sorted(unique)]


# ============================================================
# 5. READ STAGE 5 AVERAGED POINTS
# ============================================================
def read_stage5_csv(csv_path):
    """Read averaged samples from Stage 5, adapting to common column names."""
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []

        image_col = find_column(fields, ["image", "source_image", "filename", "file"])
        eye_col = find_column(fields, ["eye", "side"])
        distance_col = find_column(fields, [
            "average_distance_px", "distance_px", "distance", "average_distance"
        ])
        intensity_col = find_column(fields, [
            "average_intensity", "intensity", "gray_intensity", "grayscale_intensity"
        ])
        angle_col = find_column(fields, ["angle_deg", "angle", "profile_angle_deg"])
        pixels_col = find_column(fields, ["pixels_used", "average_pixels", "pixels_per_point"])

        missing = []
        if image_col is None:
            missing.append("image")
        if eye_col is None:
            missing.append("eye")
        if distance_col is None:
            missing.append("average_distance_px")
        if intensity_col is None:
            missing.append("average_intensity")
        if missing:
            raise ValueError(
                f"Stage 5 CSV is missing required column(s) {missing}. "
                f"Found columns: {fields}"
            )

        for line_number, row in enumerate(reader, start=2):
            image = str(row.get(image_col, "")).strip()
            eye = normalize_eye(row.get(eye_col, ""))
            distance = to_float(row.get(distance_col))
            intensity = to_float(row.get(intensity_col))
            angle = to_float(row.get(angle_col)) if angle_col else None
            pixels_used = to_float(row.get(pixels_col)) if pixels_col else None

            if not image or eye not in {"left", "right"}:
                continue
            if distance is None or intensity is None:
                continue

            rows.append({
                "image": Path(image).name,
                "eye": eye,
                "distance": distance,
                "intensity": intensity,
                "angle": angle,
                "pixels_used": int(pixels_used) if pixels_used is not None else AVERAGE_PIXELS,
                "csv_line": line_number,
            })
    return rows


def group_profiles(rows):
    groups = {}
    for row in rows:
        key = (normalize_name(row["image"]), row["eye"])
        groups.setdefault(key, []).append(row)
    for key in groups:
        groups[key].sort(key=lambda item: item["distance"])
    return groups


# ============================================================
# 6. FIT STRAIGHT LINE AND CALCULATE R-SQUARED
# ============================================================
def calculate_slope(profile_rows):
    """Fit intensity = slope * distance + intercept."""
    if len(profile_rows) < 2:
        return None, "FEWER_THAN_2_POINTS"

    x = np.asarray([row["distance"] for row in profile_rows], dtype=np.float64)
    y = np.asarray([row["intensity"] for row in profile_rows], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 2:
        return None, "FEWER_THAN_2_VALID_POINTS"
    if len(np.unique(x)) < 2:
        return None, "DISTANCE_VALUES_NOT_DISTINCT"

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "points": int(len(x)),
    }, "OK"


# ============================================================
# 7. PROCESS ONE DATASET
# ============================================================
def process_dataset(dataset_folder, slope_rows, status_rows):
    dataset_folder = Path(dataset_folder)
    # If the supplied path ends in "annotated", use its parent dataset
    # name in summaries so multiple datasets are not all labelled "annotated".
    dataset_name = (
        dataset_folder.parent.name
        if dataset_folder.name.lower() == "annotated"
        else dataset_folder.name
    )
    stage5_csv = find_stage5_csv(dataset_folder)

    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print(f"FOLDER:  {dataset_folder}")

    if stage5_csv is None:
        message = f"Stage 5 CSV not found: {STAGE5_FOLDER_NAME}/{STAGE5_AVERAGE_CSV}"
        print(f"[WARNING] {message}")
        status_rows.append({
            "path": dataset_name,
            "input_path": str(dataset_folder),
            "image": "",
            "eye": "",
            "status": "STAGE5_CSV_NOT_FOUND",
            "details": message,
        })
        return

    print(f"Stage 5 CSV: {stage5_csv}")

    try:
        rows = read_stage5_csv(stage5_csv)
    except Exception as exc:
        status_rows.append({
            "path": dataset_name,
            "input_path": str(dataset_folder),
            "image": "",
            "eye": "",
            "status": "CSV_READ_ERROR",
            "details": str(exc),
        })
        print(f"[ERROR] Could not read Stage 5 CSV: {exc}")
        return

    if not rows:
        status_rows.append({
            "path": dataset_name,
            "input_path": str(dataset_folder),
            "image": "",
            "eye": "",
            "status": "NO_VALID_ROWS",
            "details": "No rows contained a valid image, eye, distance and intensity.",
        })
        print("[WARNING] No valid averaged points in CSV.")
        return

    groups = group_profiles(rows)
    print(f"Image-eye profiles found: {len(groups)}")

    for (normalized_image, eye), profile_rows in sorted(groups.items()):
        image_name = profile_rows[0]["image"]
        fit, status = calculate_slope(profile_rows)
        angle_values = [row["angle"] for row in profile_rows if row["angle"] is not None]
        angle = float(np.mean(angle_values)) if angle_values else None
        points_used = len(profile_rows)
        pixels_values = [row["pixels_used"] for row in profile_rows]
        pixels_used = int(round(float(np.median(pixels_values)))) if pixels_values else AVERAGE_PIXELS

        if fit is None:
            status_rows.append({
                "path": dataset_name,
                "input_path": str(dataset_folder),
                "image": image_name,
                "eye": eye,
                "status": status,
                "details": f"Cannot fit line using {points_used} averaged points.",
            })
            print(f"[SKIP] {eye.upper():<5} {image_name}: {status}")
            continue

        result = {
            "path": dataset_name,
            "input_path": str(dataset_folder),
            "image": image_name,
            "eye": eye,
            "angle_deg": angle,
            "average_pixels": pixels_used,
            "average_points": fit["points"],
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "r_squared": fit["r_squared"],
            "status": "OK",
        }
        slope_rows.append(result)
        status_rows.append({
            "path": dataset_name,
            "input_path": str(dataset_folder),
            "image": image_name,
            "eye": eye,
            "status": "OK",
            "details": f"Fitted {fit['points']} averaged points.",
        })
        angle_text = f"{angle:.2f}" if angle is not None else "N/A"
        print(
            f"[OK] {eye.upper():<5} {image_name} | angle={angle_text} | "
            f"points={fit['points']} | slope={fit['slope']:.8f} | "
            f"R²={fit['r_squared']:.4f}"
        )


# ============================================================
# 8. SUMMARIZE SLOPES BY DATASET AND EYE
# ============================================================
def create_path_summary(slope_rows):
    summary = []
    paths = sorted({row["path"] for row in slope_rows})

    for path_name in paths:
        for eye in ("left", "right"):
            entries = [r for r in slope_rows if r["path"] == path_name and r["eye"] == eye]
            if not entries:
                continue
            values = np.asarray([r["slope"] for r in entries], dtype=np.float64)
            summary.append({
                "path": path_name,
                "eye": eye,
                "number_of_profiles": int(len(values)),
                "mean_slope": float(np.mean(values)),
                "median_slope": float(np.median(values)),
                "min_slope": float(np.min(values)),
                "max_slope": float(np.max(values)),
                "std_slope": float(np.std(values)),
            })
    return summary


# ============================================================
# 9. CSV OUTPUT HELPERS
# ============================================================
def save_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_slope_results(path, rows):
    fields = [
        "path", "input_path", "image", "eye", "angle_deg", "average_pixels",
        "average_points", "slope", "intercept", "r_squared", "status"
    ]
    save_csv(path, rows, fields)


def save_summary(path, rows):
    fields = [
        "path", "eye", "number_of_profiles", "mean_slope", "median_slope",
        "min_slope", "max_slope", "std_slope"
    ]
    save_csv(path, rows, fields)


def save_status(path, rows):
    fields = ["path", "input_path", "image", "eye", "status", "details"]
    save_csv(path, rows, fields)


# ============================================================
# 10. PLOT MEAN SLOPE COMPARISON
# ============================================================
def create_slope_graph(summary, eye, output_path):
    data = [row for row in summary if row["eye"] == eye]
    if not data:
        print(f"[WARNING] No valid {eye} eye slope data; graph not created.")
        return False

    data.sort(key=lambda row: row["path"])
    labels = [row["path"] for row in data]
    slopes = [row["mean_slope"] for row in data]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(labels))
    ax.plot(x, slopes, marker="o", linewidth=2, markersize=8, label="Mean slope")
    for i, value in enumerate(slopes):
        ax.annotate(f"{value:.5f}", (x[i], value), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Input dataset")
    ax.set_ylabel("Slope of intensity profile (intensity/pixel)")
    ax.set_title(f"{eye.capitalize()} Eye — Mean Slope Comparison Across Datasets")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=GRAPH_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[GRAPH] {output_path}")
    return True


# ============================================================
# 11. PLOT SLOPE VS ANGLE
# ============================================================
def create_slope_vs_angle_graph(slope_rows, eye, output_path):
    data = [
        row for row in slope_rows
        if row["eye"] == eye and row["angle_deg"] is not None
        and math.isfinite(float(row["angle_deg"]))
    ]
    if not data:
        print(f"[WARNING] No {eye} eye angle data; slope-vs-angle graph not created.")
        return False

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for dataset_name in sorted({row["path"] for row in data}):
        entries = [row for row in data if row["path"] == dataset_name]
        angles = [row["angle_deg"] for row in entries]
        slopes = [row["slope"] for row in entries]
        ax.scatter(angles, slopes, s=45, label=dataset_name)

    ax.set_xlabel("Pupil orientation angle (degrees)")
    ax.set_ylabel("Slope of intensity profile (intensity/pixel)")
    ax.set_title(f"{eye.capitalize()} Eye — Slope vs Orientation Angle")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=GRAPH_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[GRAPH] {output_path}")
    return True


# ============================================================
# 12. MAIN
# ============================================================
def main():
    print("\n" + "=" * 80)
    print("STAGE 6 — FINAL SLOPE DETECTION")
    print("=" * 80)
    print(f"Root folder: {ROOT_HR_FOLDER}")

    if INPUT_PATHS:
        datasets = []
        for input_value in INPUT_PATHS:
            supplied = Path(input_value)
            if not supplied.exists():
                print(f"[WARNING] Input path does not exist: {supplied}")
                continue
            dataset = resolve_dataset_folder(supplied)
            if dataset not in datasets:
                datasets.append(dataset)
    else:
        datasets = discover_datasets(ROOT_HR_FOLDER)

    # Keep only datasets that actually have Stage 5 points.
    valid_datasets = []
    for dataset in datasets:
        csv_path = find_stage5_csv(dataset)
        if csv_path is None:
            print(f"[WARNING] Stage 5 average CSV missing under: {dataset}")
        elif dataset not in valid_datasets:
            valid_datasets.append(dataset)

    if len(valid_datasets) < 1:
        print("\n[ERROR] No Stage 5 average_5_pixel_points.csv files found.")
        print("Expected: <dataset>\\stage5_average_fit\\average_5_pixel_points.csv")
        print("Check ROOT_HR_FOLDER or add valid folders to INPUT_PATHS.")
        return

    print(f"Datasets to process: {len(valid_datasets)}")

    output_root = ROOT_HR_FOLDER / OUTPUT_FOLDER_NAME
    graph_root = output_root / GRAPH_FOLDER_NAME
    graph_root.mkdir(parents=True, exist_ok=True)

    slope_rows = []
    status_rows = []

    for dataset in valid_datasets:
        process_dataset(dataset, slope_rows, status_rows)

    if not slope_rows:
        print("\n[ERROR] No slope values were calculated.")
        status_path = output_root / STATUS_CSV_NAME
        save_status(status_path, status_rows)
        print(f"Processing diagnostics saved: {status_path}")
        return

    summary = create_path_summary(slope_rows)

    slope_csv_path = output_root / SLOPE_CSV_NAME
    summary_csv_path = output_root / SUMMARY_CSV_NAME
    status_csv_path = output_root / STATUS_CSV_NAME

    save_slope_results(slope_csv_path, slope_rows)
    save_summary(summary_csv_path, summary)
    save_status(status_csv_path, status_rows)

    create_slope_graph(
        summary, "left", graph_root / "left_eye_slope_comparison.png"
    )
    create_slope_graph(
        summary, "right", graph_root / "right_eye_slope_comparison.png"
    )
    create_slope_vs_angle_graph(
        slope_rows, "left", graph_root / "left_eye_slope_vs_angle.png"
    )
    create_slope_vs_angle_graph(
        slope_rows, "right", graph_root / "right_eye_slope_vs_angle.png"
    )

    print("\n" + "=" * 80)
    print("STAGE 6 COMPLETED")
    print("=" * 80)
    print(f"Valid image-eye slope profiles: {len(slope_rows)}")
    print(f"Slope CSV:   {slope_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Status CSV:  {status_csv_path}")
    print(f"Graphs:      {graph_root}")
    print("\nSlope = change in average grayscale intensity per pixel of profile distance.")
    print("R² describes how closely the straight line fits the averaged points.")


if __name__ == "__main__":
    main()
