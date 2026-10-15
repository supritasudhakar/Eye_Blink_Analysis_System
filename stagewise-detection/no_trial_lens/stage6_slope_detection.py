# ============================================================
# stage6_slope_detection.py
#
# STAGE 6 - FINAL SLOPE DETECTION
#
# INPUT:
#     Stage 5 average_5_pixel_points.csv
#
# PROCEDURE:
#
#     Stage 5:
#         Ignore glint
#         Average every 5 pixels
#         Produce averaged intensity points
#
#     Stage 6:
#         Read averaged points
#         Fit straight line
#         Calculate slope
#         Calculate R²
#         Compare multiple input paths
#         Generate LEFT and RIGHT slope graphs
#
# NO:
#     YOLO
#     ML
#     Haar
#     best.pt
#
# ============================================================

import os
import csv
import glob

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# ROOT FOLDER
# ============================================================

ROOT_HR_FOLDER = (
    )


# ============================================================
# INPUT PATHS
#
# Add 2, 3, 4 or more dataset folders here.
#
# Example:
#
# pupil_image
#     ├── sachin
#     ├── rakshita
#     ├── shiya
#
# Each folder should contain:
#
# stage5_average_fit
#     └── average_5_pixel_points.csv
#
# ============================================================

INPUT_PATHS = [

    os.path.join(
        ROOT_HR_FOLDER,
        " "
    )

    
    
       


    

]


# ============================================================
# STAGE 5 FILE
# ============================================================

STAGE5_FOLDER_NAME = "stage5_average_fit"

STAGE5_AVERAGE_CSV = "average_5_pixel_points.csv"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FOLDER_NAME = "stage6_slope_detection"

GRAPH_FOLDER_NAME = "graphs"

SLOPE_CSV_NAME = "slope_results.csv"

SUMMARY_CSV_NAME = "slope_summary.csv"


# ============================================================
# AVERAGE PIXELS
#
# Stage 5 uses 5 pixels.
# This is recorded in the output for verification.
# ============================================================

AVERAGE_PIXELS = 5


# ============================================================
# GRAPH
# ============================================================

FIGURE_SIZE = (12, 7)

GRAPH_DPI = 160


# ============================================================
# FIND STAGE 5 CSV
# ============================================================

def find_stage5_csv(folder):

    # --------------------------------------------------------
    # First check the normal expected location
    # --------------------------------------------------------

    direct_path = os.path.join(
        folder,
        STAGE5_FOLDER_NAME,
        STAGE5_AVERAGE_CSV
    )

    if os.path.isfile(
        direct_path
    ):

        return direct_path

    # --------------------------------------------------------
    # Otherwise search recursively
    # --------------------------------------------------------

    pattern = os.path.join(
        folder,
        "**",
        STAGE5_AVERAGE_CSV
    )

    files = glob.glob(
        pattern,
        recursive=True
    )

    valid_files = []

    for path in files:

        parts = [
            p.lower()
            for p in os.path.normpath(
                path
            ).split(os.sep)
        ]

        if (
            STAGE5_FOLDER_NAME.lower()
            in parts
        ):

            valid_files.append(
                path
            )

    if not valid_files:

        return None

    return sorted(
        valid_files
    )[0]


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    fieldnames,
    possible_names
):

    if not fieldnames:

        return None

    normalized = {}

    for field in fieldnames:

        key = (
            str(field)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        normalized[key] = field

    for name in possible_names:

        key = (
            name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if key in normalized:

            return normalized[key]

    return None


# ============================================================
# NORMALIZE EYE
# ============================================================

def normalize_eye(value):

    if value is None:

        return ""

    value = (
        str(value)
        .strip()
        .lower()
    )

    if value in (
        "left",
        "le",
        "l",
        "left_eye",
        "left eye"
    ):

        return "left"

    if value in (
        "right",
        "re",
        "r",
        "right_eye",
        "right eye"
    ):

        return "right"

    return value


# ============================================================
# READ STAGE 5 AVERAGED POINTS
# ============================================================

def read_stage5_csv(
    csv_path
):

    rows = []

    try:

        with open(
            csv_path,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(
                f
            )

            fieldnames = (
                reader.fieldnames
            )

            if not fieldnames:

                print(
                    "[WARNING] Empty CSV:"
                )

                print(
                    csv_path
                )

                return rows

            # ------------------------------------------------
            # Columns
            # ------------------------------------------------

            image_col = find_column(
                fieldnames,
                [
                    "image",
                    "source_image",
                    "filename"
                ]
            )

            eye_col = find_column(
                fieldnames,
                [
                    "eye",
                    "side"
                ]
            )

            distance_col = find_column(
                fieldnames,
                [
                    "average_distance_px",
                    "distance_px",
                    "distance",
                    "average_distance"
                ]
            )

            intensity_col = find_column(
                fieldnames,
                [
                    "average_intensity",
                    "intensity",
                    "gray_intensity"
                ]
            )

            angle_col = find_column(
                fieldnames,
                [
                    "angle_deg",
                    "angle"
                ]
            )

            if image_col is None:

                print(
                    "[WARNING] Image column missing:"
                )

                print(
                    csv_path
                )

                return rows

            if eye_col is None:

                print(
                    "[WARNING] Eye column missing:"
                )

                print(
                    csv_path
                )

                return rows

            if distance_col is None:

                print(
                    "[WARNING] Distance column missing:"
                )

                print(
                    csv_path
                )

                return rows

            if intensity_col is None:

                print(
                    "[WARNING] Intensity column missing:"
                )

                print(
                    csv_path
                )

                return rows

            # ------------------------------------------------
            # Read rows
            # ------------------------------------------------

            for row in reader:

                try:

                    image_name = str(
                        row.get(
                            image_col,
                            ""
                        )
                    ).strip()

                    eye = normalize_eye(
                        row.get(
                            eye_col,
                            ""
                        )
                    )

                    if not image_name:

                        continue

                    if eye not in (
                        "left",
                        "right"
                    ):

                        continue

                    distance = float(
                        row[
                            distance_col
                        ]
                    )

                    intensity = float(
                        row[
                            intensity_col
                        ]
                    )

                    if angle_col is not None:

                        angle_text = str(
                            row.get(
                                angle_col,
                                ""
                            )
                        ).strip()

                        if angle_text:

                            angle = float(
                                angle_text
                            )

                        else:

                            angle = None

                    else:

                        angle = None

                    rows.append(
                        {
                            "image":
                                image_name,

                            "eye":
                                eye,

                            "distance":
                                distance,

                            "intensity":
                                intensity,

                            "angle":
                                angle
                        }
                    )

                except (
                    ValueError,
                    TypeError,
                    KeyError
                ):

                    continue

    except Exception as e:

        print(
            "[ERROR] Could not read:"
        )

        print(
            csv_path
        )

        print(
            e
        )

    return rows


# ============================================================
# GROUP BY IMAGE + EYE
# ============================================================

def group_profile_rows(
    rows
):

    groups = {}

    for row in rows:

        key = (
            row["image"],
            row["eye"]
        )

        if key not in groups:

            groups[key] = []

        groups[key].append(
            row
        )

    # --------------------------------------------------------
    # Sort every profile by distance
    # --------------------------------------------------------

    for key in groups:

        groups[key].sort(
            key=lambda x:
            x["distance"]
        )

    return groups


# ============================================================
# CALCULATE SLOPE
#
# y = mx + c
#
# m = slope
#
# x = averaged distance
# y = averaged intensity
# ============================================================

def calculate_slope(
    rows
):

    if len(rows) < 2:

        return None

    x = np.asarray(
        [
            row["distance"]
            for row in rows
        ],
        dtype=np.float64
    )

    y = np.asarray(
        [
            row["intensity"]
            for row in rows
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]

    y = y[valid]

    if len(x) < 2:

        return None

    # --------------------------------------------------------
    # Straight line fit
    # --------------------------------------------------------

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    slope = float(
        slope
    )

    intercept = float(
        intercept
    )

    # --------------------------------------------------------
    # Predicted intensity
    # --------------------------------------------------------

    predicted = (
        slope * x
        +
        intercept
    )

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    ss_res = float(
        np.sum(
            (
                y
                -
                predicted
            ) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                y
                -
                np.mean(y)
            ) ** 2
        )
    )

    if ss_tot > 1e-12:

        r_squared = (
            1.0
            -
            ss_res / ss_tot
        )

    else:

        r_squared = 1.0

    return {
        "slope":
            slope,

        "intercept":
            intercept,

        "r_squared":
            float(
                r_squared
            ),

        "points":
            len(x)
    }


# ============================================================
# PROCESS ONE INPUT PATH
# ============================================================

def process_one_path(
    input_path,
    slope_rows
):

    dataset_name = os.path.basename(
        os.path.normpath(
            input_path
        )
    )

    print()
    print(
        "-" * 75
    )

    print(
        f"INPUT PATH: {dataset_name}"
    )

    print(
        input_path
    )

    # --------------------------------------------------------
    # Find Stage 5 CSV
    # --------------------------------------------------------

    stage5_csv = find_stage5_csv(
        input_path
    )

    if stage5_csv is None:

        print(
            "[WARNING] Stage 5 CSV not found."
        )

        print(
            "Expected:"
        )

        print(
            os.path.join(
                STAGE5_FOLDER_NAME,
                STAGE5_AVERAGE_CSV
            )
        )

        return []

    print()
    print(
        "Stage 5 CSV:"
    )

    print(
        stage5_csv
    )

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    rows = read_stage5_csv(
        stage5_csv
    )

    if not rows:

        print(
            "[WARNING] No valid Stage 5 points."
        )

        return []

    # --------------------------------------------------------
    # Group image + eye
    # --------------------------------------------------------

    groups = group_profile_rows(
        rows
    )

    print(
        f"Profiles found: {len(groups)}"
    )

    path_results = []

    # --------------------------------------------------------
    # Process each image + eye
    # --------------------------------------------------------

    for (
        image_name,
        eye
    ), profile_rows in groups.items():

        result = calculate_slope(
            profile_rows
        )

        if result is None:

            print(
                f"[SKIP] "
                f"{eye.upper()} "
                f"{image_name}"
            )

            continue

        angle_values = [
            row["angle"]
            for row in profile_rows
            if row["angle"] is not None
        ]

        if angle_values:

            angle = float(
                np.mean(
                    angle_values
                )
            )

        else:

            angle = None

        slope_result = {
            "path":
                dataset_name,

            "input_path":
                input_path,

            "image":
                image_name,

            "eye":
                eye,

            "angle_deg":
                angle,

            "average_pixels":
                AVERAGE_PIXELS,

            "average_points":
                result["points"],

            "slope":
                result["slope"],

            "intercept":
                result["intercept"],

            "r_squared":
                result["r_squared"]
        }

        slope_rows.append(
            slope_result
        )

        path_results.append(
            slope_result
        )

        print(
            f"[OK] "
            f"{eye.upper():<5} "
            f"{os.path.basename(image_name)} "
            f"| points = "
            f"{result['points']} "
            f"| slope = "
            f"{result['slope']:.8f} "
            f"| R² = "
            f"{result['r_squared']:.4f}"
        )

    return path_results


# ============================================================
# CALCULATE PATH SUMMARY
#
# For every input path:
#
#     LEFT  -> mean slope
#     RIGHT -> mean slope
#
# This allows multiple paths to be compared.
# ============================================================

def create_path_summary(
    slope_rows
):

    summary = []

    paths = sorted(
        set(
            row["path"]
            for row in slope_rows
        )
    )

    for path in paths:

        for eye in (
            "left",
            "right"
        ):

            values = [
                row["slope"]
                for row in slope_rows
                if (
                    row["path"] == path
                    and
                    row["eye"] == eye
                )
            ]

            if not values:

                continue

            values = np.asarray(
                values,
                dtype=np.float64
            )

            summary.append(
                {
                    "path":
                        path,

                    "eye":
                        eye,

                    "number_of_profiles":
                        len(values),

                    "mean_slope":
                        float(
                            np.mean(values)
                        ),

                    "median_slope":
                        float(
                            np.median(values)
                        ),

                    "min_slope":
                        float(
                            np.min(values)
                        ),

                    "max_slope":
                        float(
                            np.max(values)
                        ),

                    "std_slope":
                        float(
                            np.std(values)
                        )
                }
            )

    return summary


# ============================================================
# SAVE SLOPE CSV
# ============================================================

def save_slope_csv(
    path,
    rows
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    fieldnames = [
        "path",
        "input_path",
        "image",
        "eye",
        "angle_deg",
        "average_pixels",
        "average_points",
        "slope",
        "intercept",
        "r_squared"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:

            output_row = dict(
                row
            )

            writer.writerow(
                output_row
            )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

def save_summary_csv(
    path,
    rows
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    fieldnames = [
        "path",
        "eye",
        "number_of_profiles",
        "mean_slope",
        "median_slope",
        "min_slope",
        "max_slope",
        "std_slope"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# PLOT MULTI-PATH SLOPES
#
# Separate graph:
#
#     LEFT EYE
#
# and
#
#     RIGHT EYE
#
# Each path gets one mean slope point.
# ============================================================

def create_slope_graph(
    summary,
    eye,
    output_path
):

    data = [
        row
        for row in summary
        if row["eye"] == eye
    ]

    if not data:

        print(
            f"[WARNING] No {eye} eye slope data."
        )

        return

    # --------------------------------------------------------
    # Sort by path
    # --------------------------------------------------------

    data.sort(
        key=lambda row:
        row["path"]
    )

    labels = [
        row["path"]
        for row in data
    ]

    slopes = [
        row["mean_slope"]
        for row in data
    ]

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    plt.figure(
        figsize=FIGURE_SIZE
    )

    x = np.arange(
        len(labels)
    )

    plt.plot(
        x,
        slopes,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Mean slope"
    )

    # --------------------------------------------------------
    # Show values on graph
    # --------------------------------------------------------

    for i, value in enumerate(
        slopes
    ):

        plt.text(
            i,
            value,
            f"{value:.5f}",
            ha="center",
            va="bottom"
        )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    plt.xticks(
        x,
        labels,
        rotation=30,
        ha="right"
    )

    plt.xlabel(
        "Input path"
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    plt.ylabel(
        "Slope of intensity profile"
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    plt.title(
        f"{eye.capitalize()} Eye - "
        f"Slope Comparison Across Input Paths"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    plt.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    plt.legend()

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=GRAPH_DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[GRAPH] {eye.upper()}:"
    )

    print(
        output_path
    )


# ============================================================
# OPTIONAL:
# SLOPE VS ANGLE GRAPH
#
# This graph shows individual profiles from all paths.
# ============================================================

def create_slope_vs_angle_graph(
    slope_rows,
    eye,
    output_path
):

    data = [
        row
        for row in slope_rows
        if (
            row["eye"] == eye
            and
            row["angle_deg"] is not None
        )
    ]

    if not data:

        print(
            f"[WARNING] "
            f"No angle data for {eye}."
        )

        return

    plt.figure(
        figsize=FIGURE_SIZE
    )

    # --------------------------------------------------------
    # Plot each path separately
    # --------------------------------------------------------

    paths = sorted(
        set(
            row["path"]
            for row in data
        )
    )

    for path in paths:

        path_data = [
            row
            for row in data
            if row["path"] == path
        ]

        angles = [
            row["angle_deg"]
            for row in path_data
        ]

        slopes = [
            row["slope"]
            for row in path_data
        ]

        plt.scatter(
            angles,
            slopes,
            s=45,
            label=path
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xlabel(
        "Pupil orientation angle (degrees)"
    )

    plt.ylabel(
        "Slope of intensity profile"
    )

    plt.title(
        f"{eye.capitalize()} Eye - "
        f"Slope vs Pupil Angle"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=GRAPH_DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[GRAPH] {eye.upper()} "
        f"slope vs angle:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "STAGE 6 - FINAL SLOPE DETECTION"
    )
    print("=" * 80)
    print()

    print(
        f"Number of input paths: "
        f"{len(INPUT_PATHS)}"
    )

    print(
        f"Average pixels: "
        f"{AVERAGE_PIXELS}"
    )

    print()

    # ========================================================
    # CHECK INPUT PATH COUNT
    # ========================================================

    if len(INPUT_PATHS) < 2:

        print(
            "[ERROR] Please provide at least 2 input paths."
        )

        print(
            "Edit INPUT_PATHS in the code."
        )

        return

    # ========================================================
    # OUTPUT
    # ========================================================

    output_root = os.path.join(
        ROOT_HR_FOLDER,
        OUTPUT_FOLDER_NAME
    )

    graph_root = os.path.join(
        output_root,
        GRAPH_FOLDER_NAME
    )

    os.makedirs(
        graph_root,
        exist_ok=True
    )

    # ========================================================
    # ALL SLOPE RESULTS
    # ========================================================

    slope_rows = []

    # ========================================================
    # PROCESS EVERY PATH
    # ========================================================

    for input_path in INPUT_PATHS:

        if not os.path.isdir(
            input_path
        ):

            print()
            print(
                "[WARNING] Input path does not exist:"
            )

            print(
                input_path
            )

            continue

        process_one_path(
            input_path,
            slope_rows
        )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not slope_rows:

        print()
        print(
            "[ERROR] No slope values calculated."
        )

        print()
        print(
            "Check that every input path contains:"
        )

        print(
            "stage5_average_fit"
        )

        print(
            "    └── average_5_pixel_points.csv"
        )

        return

    # ========================================================
    # CREATE SUMMARY
    # ========================================================

    summary = create_path_summary(
        slope_rows
    )

    # ========================================================
    # SAVE SLOPE RESULTS
    # ========================================================

    slope_csv_path = os.path.join(
        output_root,
        SLOPE_CSV_NAME
    )

    save_slope_csv(
        slope_csv_path,
        slope_rows
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_csv_path = os.path.join(
        output_root,
        SUMMARY_CSV_NAME
    )

    save_summary_csv(
        summary_csv_path,
        summary
    )

    # ========================================================
    # CREATE LEFT GRAPH
    # ========================================================

    left_graph = os.path.join(
        graph_root,
        "left_eye_slope_comparison.png"
    )

    create_slope_graph(
        summary,
        "left",
        left_graph
    )

    # ========================================================
    # CREATE RIGHT GRAPH
    # ========================================================

    right_graph = os.path.join(
        graph_root,
        "right_eye_slope_comparison.png"
    )

    create_slope_graph(
        summary,
        "right",
        right_graph
    )

    # ========================================================
    # CREATE SLOPE VS ANGLE GRAPHS
    # ========================================================

    left_angle_graph = os.path.join(
        graph_root,
        "left_eye_slope_vs_angle.png"
    )

    create_slope_vs_angle_graph(
        slope_rows,
        "left",
        left_angle_graph
    )

    right_angle_graph = os.path.join(
        graph_root,
        "right_eye_slope_vs_angle.png"
    )

    create_slope_vs_angle_graph(
        slope_rows,
        "right",
        right_angle_graph
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STAGE 6 COMPLETED"
    )
    print("=" * 80)
    print()

    print(
        f"Total slope profiles: "
        f"{len(slope_rows)}"
    )

    print()

    print(
        "PATH SUMMARY"
    )

    print(
        "-" * 80
    )

    for row in summary:

        print(
            f"{row['path']:<20} "
            f"{row['eye'].upper():<6} "
            f"profiles="
            f"{row['number_of_profiles']:<5} "
            f"mean slope="
            f"{row['mean_slope']:.8f}"
        )

    print()

    print(
        "Slope CSV:"
    )

    print(
        slope_csv_path
    )

    print()

    print(
        "Summary CSV:"
    )

    print(
        summary_csv_path
    )

    print()

    print(
        "Graphs:"
    )

    print(
        graph_root
    )

    print()

    print(
        "LEFT:"
    )

    print(
        left_graph
    )

    print()

    print(
        "RIGHT:"
    )

    print(
        right_graph
    )

    print()

    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()