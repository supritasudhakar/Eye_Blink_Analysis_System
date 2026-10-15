# ============================================================
# stage5_average_fit.py
#
# STAGE 5
#
# INPUT:
#     Stage 4 angle_intensity_values.csv
#
# PROCEDURE:
#     1. Read complete Stage 4 intensity profile
#     2. Detect bright glint / bright spike
#     3. Ignore the complete glint region
#     4. Continue processing after the glint
#     5. Average neighbouring VALID pixels
#     6. EXACTLY 5 pixels -> 1 average point
#     7. Fit straight line to averaged points
#     8. Calculate R² for graph quality
#     9. Save graph
#
# IMPORTANT:
#     - Averaging groups NEVER cross an ignored glint region.
#     - Stage 5 is NOT the final slope calculation stage.
#     - No YOLO
#     - No ML
#     - No Haar
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
# STAGE 4
# ============================================================

STAGE4_FOLDER_NAME = "stage4_intensity_profile"

STAGE4_CSV_NAME = "angle_intensity_values.csv"


# ============================================================
# STAGE 5 OUTPUT
# ============================================================

STAGE5_FOLDER_NAME = "stage5_average_fit"

GRAPH_FOLDER_NAME = "graphs"


AVERAGE_POINTS_CSV = (
    "average_5_pixel_points.csv"
)

FIT_INFORMATION_CSV = (
    "stage5_fit_information.csv"
)


# ============================================================
# AVERAGING
# ============================================================

# IMPORTANT:
#
# EXACTLY 5 VALID PIXELS
# are used to create one averaged point.
#
AVERAGE_PROFILE_PIXELS = 5


# ============================================================
# GLINT / BRIGHT SPIKE PARAMETERS
# ============================================================

# Brightness threshold.
#
# This is intentionally configurable because IR images
# can have different exposure levels.

MIN_BRIGHT_VALUE = 100.0


# Minimum rise above local neighbouring pixels.

GLINT_SPIKE_MIN_RISE = 20.0


# Number of neighbouring pixels used to estimate
# the local background.

GLINT_SPIKE_NEIGHBOR_POINTS = 2


# After detecting a bright spike, expand the ignored
# region on both sides.
#
# Example:
#
# detected point = 50
# padding = 2
#
# ignored:
# 48,49,50,51,52
#
GLINT_IGNORE_PADDING = 2


# ============================================================
# GRAPH
# ============================================================

FIGURE_SIZE = (12, 7)

GRAPH_DPI = 160


# ============================================================
# FIND STAGE 4 CSV FILES
# ============================================================

def find_stage4_csv_files(root_folder):

    pattern = os.path.join(
        root_folder,
        "**",
        STAGE4_CSV_NAME
    )

    files = glob.glob(
        pattern,
        recursive=True
    )

    result = []

    for path in files:

        parts = [
            p.lower()
            for p in os.path.normpath(
                path
            ).split(os.sep)
        ]

        if (
            STAGE4_FOLDER_NAME.lower()
            in
            parts
        ):

            result.append(path)

    return sorted(
        set(result)
    )


# ============================================================
# FIND CSV COLUMN
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
# READ STAGE 4 CSV
# ============================================================

def read_stage4_csv(csv_path):

    rows = []

    try:

        with open(
            csv_path,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(f)

            fieldnames = reader.fieldnames

            if not fieldnames:

                print(
                    "[WARNING] CSV has no columns:"
                )

                print(csv_path)

                return rows

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            image_col = find_column(
                fieldnames,
                [
                    "image",
                    "source_image",
                    "filename",
                    "file"
                ]
            )

            # ------------------------------------------------
            # EYE
            # ------------------------------------------------

            eye_col = find_column(
                fieldnames,
                [
                    "eye",
                    "side"
                ]
            )

            # ------------------------------------------------
            # ANGLE
            # ------------------------------------------------

            angle_col = find_column(
                fieldnames,
                [
                    "angle_deg",
                    "profile_angle_deg",
                    "common_angle_deg"
                ]
            )

            # ------------------------------------------------
            # DISTANCE
            # ------------------------------------------------

            distance_col = find_column(
                fieldnames,
                [
                    "distance_px",
                    "distance",
                    "profile_distance",
                    "x_distance",
                    "position"
                ]
            )

            # ------------------------------------------------
            # INTENSITY
            # ------------------------------------------------

            intensity_col = find_column(
                fieldnames,
                [
                    "intensity",
                    "gray_intensity",
                    "grayscale_intensity",
                    "average_intensity"
                ]
            )

            if image_col is None:

                print(
                    "[WARNING] Image column not found:"
                )

                print(csv_path)

                return rows

            if eye_col is None:

                print(
                    "[WARNING] Eye column not found:"
                )

                print(csv_path)

                return rows

            if intensity_col is None:

                print(
                    "[WARNING] Intensity column not found:"
                )

                print(csv_path)

                return rows

            # ------------------------------------------------
            # READ
            # ------------------------------------------------

            for row_number, row in enumerate(
                reader
            ):

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

                    intensity_text = str(
                        row.get(
                            intensity_col,
                            ""
                        )
                    ).strip()

                    if not intensity_text:

                        continue

                    intensity = float(
                        intensity_text
                    )

                    # ----------------------------------------
                    # DISTANCE
                    # ----------------------------------------

                    if distance_col is not None:

                        distance_text = str(
                            row.get(
                                distance_col,
                                ""
                            )
                        ).strip()

                        if distance_text:

                            distance = float(
                                distance_text
                            )

                        else:

                            distance = float(
                                row_number
                            )

                    else:

                        distance = float(
                            row_number
                        )

                    # ----------------------------------------
                    # ANGLE
                    # ----------------------------------------

                    angle = None

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

                    rows.append(
                        {
                            "image":
                                image_name,

                            "eye":
                                eye,

                            "angle":
                                angle,

                            "distance":
                                distance,

                            "intensity":
                                intensity
                        }
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

    except Exception as e:

        print(
            "[ERROR] Could not read:"
        )

        print(csv_path)

        print(e)

    return rows


# ============================================================
# GROUP BY IMAGE + EYE
# ============================================================

def group_profiles(rows):

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
    # Sort by distance
    # --------------------------------------------------------

    for key in groups:

        groups[key].sort(
            key=lambda item:
            item["distance"]
        )

    return groups


# ============================================================
# DETECT BRIGHT SPIKE
# ============================================================

def is_bright_profile_spike(
    intensities,
    index
):

    n = len(
        intensities
    )

    k = (
        GLINT_SPIKE_NEIGHBOR_POINTS
    )

    # Not enough pixels on both sides.

    if (
        index < k
        or
        index >= n - k
    ):

        return False

    current = float(
        intensities[index]
    )

    # Left neighbours

    left_values = (
        intensities[
            index - k:index
        ]
    )

    # Right neighbours

    right_values = (
        intensities[
            index + 1:
            index + 1 + k
        ]
    )

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
# DETECT COMPLETE GLINT REGION
# ============================================================

def detect_glint_region(
    intensities
):

    intensities = np.asarray(
        intensities,
        dtype=np.float64
    )

    n = len(
        intensities
    )

    ignored = np.zeros(
        n,
        dtype=bool
    )

    if n < 5:

        return (
            ignored,
            None,
            None
        )

    # --------------------------------------------------------
    # Find all bright spike candidates
    # --------------------------------------------------------

    candidates = []

    for i in range(n):

        if (
            intensities[i]
            <
            MIN_BRIGHT_VALUE
        ):

            continue

        if is_bright_profile_spike(
            intensities,
            i
        ):

            candidates.append(i)

    if not candidates:

        return (
            ignored,
            None,
            None
        )

    # --------------------------------------------------------
    # Group consecutive candidates
    # --------------------------------------------------------

    candidate_array = np.asarray(
        candidates,
        dtype=int
    )

    breaks = np.where(
        np.diff(candidate_array) > 1
    )[0] + 1

    groups = np.split(
        candidate_array,
        breaks
    )

    # --------------------------------------------------------
    # Select strongest bright group
    #
    # The glint is normally the strongest local bright
    # region along the detected angle.
    # --------------------------------------------------------

    best_group = None

    best_score = -np.inf

    for group in groups:

        if len(group) == 0:

            continue

        score = float(
            np.mean(
                intensities[group]
            )
        )

        # Prefer stronger and reasonably sized groups.

        if score > best_score:

            best_score = score

            best_group = group

    if best_group is None:

        return (
            ignored,
            None,
            None
        )

    # --------------------------------------------------------
    # Expand complete glint region
    # --------------------------------------------------------

    start = max(
        0,
        int(best_group[0])
        -
        GLINT_IGNORE_PADDING
    )

    end = min(
        n - 1,
        int(best_group[-1])
        +
        GLINT_IGNORE_PADDING
    )

    ignored[
        start:
        end + 1
    ] = True

    return (
        ignored,
        start,
        end
    )


# ============================================================
# AVERAGE VALID PROFILE PIXELS
#
# THIS IS THE SAME LOGIC AS THE WORKING CODE.
#
# Important:
#     A group NEVER crosses an ignored glint region.
#
# Exactly 5 valid neighbouring pixels are averaged.
# ============================================================

def average_valid_profile_pixels(
    distances,
    intensities,
    ignored,
    group_size=5
):

    distances = np.asarray(
        distances,
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

    empty = {
        "distances":
            np.array(
                [],
                dtype=np.float64
            ),

        "intensities":
            np.array(
                [],
                dtype=np.float64
            )
    }

    n = len(
        intensities
    )

    if n == 0:

        return empty

    # --------------------------------------------------------
    # Only valid pixels
    # --------------------------------------------------------

    valid_indices = np.flatnonzero(
        ~ignored
    )

    if len(valid_indices) == 0:

        return empty

    # --------------------------------------------------------
    # Split valid pixels into continuous runs.
    #
    # This is VERY important.
    #
    # If glint is:
    #
    # 45 46 47 48 49
    #
    # the averaging must NOT create a group such as:
    #
    # 43 44 50 51 52
    #
    # --------------------------------------------------------

    breaks = np.where(
        np.diff(valid_indices) != 1
    )[0] + 1

    runs = np.split(
        valid_indices,
        breaks
    )

    avg_distances = []

    avg_intensities = []

    # --------------------------------------------------------
    # Process every continuous valid run
    # --------------------------------------------------------

    for run in runs:

        if len(run) == 0:

            continue

        # ----------------------------------------------------
        # EXACTLY 5 PIXELS PER GROUP
        # ----------------------------------------------------

        for start_index in range(
            0,
            len(run),
            group_size
        ):

            group = run[
                start_index:
                start_index + group_size
            ]

            if len(group) == 0:

                continue

            # ------------------------------------------------
            # Important:
            #
            # Only COMPLETE groups of 5 are used.
            #
            # This matches the working code's
            # neighbouring-pixel averaging behaviour.
            # ------------------------------------------------

            if len(group) != group_size:

                continue

            avg_distance = float(
                np.mean(
                    distances[group]
                )
            )

            avg_intensity = float(
                np.mean(
                    intensities[group]
                )
            )

            avg_distances.append(
                avg_distance
            )

            avg_intensities.append(
                avg_intensity
            )

    return {
        "distances":
            np.asarray(
                avg_distances,
                dtype=np.float64
            ),

        "intensities":
            np.asarray(
                avg_intensities,
                dtype=np.float64
            )
    }


# ============================================================
# FIT STRAIGHT LINE
#
# SAME BASIC FIT USED BY WORKING CODE:
#
#     y = slope*x + intercept
#
# Stage 5 uses the fitted line for visualization.
# ============================================================

def fit_straight_line(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    if len(x) < 2:

        return None

    if len(x) != len(y):

        return None

    # --------------------------------------------------------
    # Straight-line fit
    # --------------------------------------------------------

    fit_slope, fit_intercept = np.polyfit(
        x,
        y,
        1
    )

    fit_slope = float(
        fit_slope
    )

    fit_intercept = float(
        fit_intercept
    )

    # --------------------------------------------------------
    # Predicted values
    # --------------------------------------------------------

    fitted_values = (
        fit_slope * x
        +
        fit_intercept
    )

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    ss_res = float(
        np.sum(
            (
                y
                -
                fitted_values
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

        fit_r2 = (
            1.0
            -
            ss_res / ss_tot
        )

    else:

        fit_r2 = 1.0

    return {
        "slope":
            fit_slope,

        "intercept":
            fit_intercept,

        "r2":
            float(fit_r2)
    }


# ============================================================
# CREATE GRAPH
#
# This follows the graph style of your working
# glint_straight_line.py.
# ============================================================

def save_average_fit_graph(
    profile,
    output_path,
    image_name,
    eye,
    glint_count
):

    x = np.asarray(
        profile["distances"],
        dtype=float
    ).reshape(-1)

    y = np.asarray(
        profile["intensities"],
        dtype=float
    ).reshape(-1)

    if (
        x.size == 0
        or
        y.size == 0
        or
        x.size != y.size
    ):

        return False

    # --------------------------------------------------------
    # Dark / bright points
    # --------------------------------------------------------

    dark_index = int(
        np.argmin(y)
    )

    bright_index = int(
        np.argmax(y)
    )

    # --------------------------------------------------------
    # Fit information
    # --------------------------------------------------------

    fit = profile["fit"]

    slope = fit["slope"]

    intercept = fit["intercept"]

    r2 = fit["r2"]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=FIGURE_SIZE
    )

    # --------------------------------------------------------
    # AVERAGED POINTS
    # --------------------------------------------------------

    plt.scatter(
        x,
        y,
        s=55,
        label=(
            f"Averaged intensity "
            f"({AVERAGE_PROFILE_PIXELS} pixels)"
        )
    )

    # --------------------------------------------------------
    # STRAIGHT-LINE FIT
    # --------------------------------------------------------

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
        f"y = {slope:.4f}x "
        f"+ {intercept:.4f}"
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
        [x[dark_index]],
        [y[dark_index]],
        s=75,
        label=(
            f"Dark = "
            f"{y[dark_index]:.1f}"
        )
    )

    # --------------------------------------------------------
    # BRIGHT POINT
    # --------------------------------------------------------

    plt.scatter(
        [x[bright_index]],
        [y[bright_index]],
        s=75,
        label=(
            f"Bright = "
            f"{y[bright_index]:.1f}"
        )
    )

    # --------------------------------------------------------
    # X AXIS
    # --------------------------------------------------------

    plt.xlabel(
        "Distance along detected angle direction (pixels)\n"
        "Dark side → Bright side"
    )

    # --------------------------------------------------------
    # Y AXIS
    # --------------------------------------------------------

    plt.ylabel(
        "Average grayscale intensity (0–255)"
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    angle = profile.get(
        "angle"
    )

    if angle is None:

        angle_text = "N/A"

    else:

        angle_text = (
            f"{angle:.1f}°"
        )

    title = (
        f"{eye.capitalize()} eye | "
        f"Angle = {angle_text} | "
        f"Glint points ignored = "
        f"{glint_count} | "
        f"Average = "
        f"{AVERAGE_PROFILE_PIXELS} pixels | "
        f"R² = {r2:.4f}"
    )

    plt.title(
        title
    )

    # --------------------------------------------------------
    # Y RANGE
    # --------------------------------------------------------

    plt.ylim(
        0,
        255
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    plt.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    # Add R² as a separate legend line.

    handles, labels = (
        plt.gca()
        .get_legend_handles_labels()
    )

    from matplotlib.lines import Line2D

    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="",
            label=f"R² = {r2:.4f}"
        )
    )

    labels.append(
        f"R² = {r2:.4f}"
    )

    plt.legend(
        handles,
        labels
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
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

    return True


# ============================================================
# PROCESS ONE IMAGE + EYE
# ============================================================

def process_profile(
    profile_rows,
    image_name,
    eye,
    average_rows,
    fit_rows,
    graph_root
):

    if len(
        profile_rows
    ) < AVERAGE_PROFILE_PIXELS:

        return False

    # --------------------------------------------------------
    # Sort by distance
    # --------------------------------------------------------

    profile_rows = sorted(
        profile_rows,
        key=lambda row:
        row["distance"]
    )

    distances = np.asarray(
        [
            row["distance"]
            for row in profile_rows
        ],
        dtype=np.float64
    )

    intensities = np.asarray(
        [
            row["intensity"]
            for row in profile_rows
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Angle
    # --------------------------------------------------------

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

    # ========================================================
    # STEP 1
    #
    # DETECT GLINT
    # ========================================================

    (
        ignored,
        glint_start,
        glint_end
    ) = detect_glint_region(
        intensities
    )

    glint_count = int(
        np.count_nonzero(
            ignored
        )
    )

    # ========================================================
    # STEP 2
    #
    # AVERAGE 5 VALID PIXELS
    #
    # IMPORTANT:
    # groups cannot cross glint.
    # ========================================================

    averaged = (
        average_valid_profile_pixels(
            distances,
            intensities,
            ignored,
            group_size=AVERAGE_PROFILE_PIXELS
        )
    )

    avg_distances = (
        averaged["distances"]
    )

    avg_intensities = (
        averaged["intensities"]
    )

    if len(
        avg_intensities
    ) < 2:

        print(
            f"[SKIP] Not enough averaged points:"
            f" {image_name} | {eye}"
        )

        return False

    # ========================================================
    # STEP 3
    #
    # FIT STRAIGHT LINE
    # ========================================================

    fit = fit_straight_line(
        avg_distances,
        avg_intensities
    )

    if fit is None:

        return False

    # ========================================================
    # STORE AVERAGE POINTS
    # ========================================================

    for index in range(
        len(avg_distances)
    ):

        average_rows.append(
            {
                "image":
                    image_name,

                "eye":
                    eye,

                "angle_deg":
                    ""
                    if angle is None
                    else f"{angle:.4f}",

                "average_index":
                    index,

                "average_distance_px":
                    f"{avg_distances[index]:.4f}",

                "average_intensity":
                    f"{avg_intensities[index]:.4f}",

                "pixels_used":
                    AVERAGE_PROFILE_PIXELS,

                "glint_points_ignored":
                    glint_count,

                "glint_start_index":
                    ""
                    if glint_start is None
                    else glint_start,

                "glint_end_index":
                    ""
                    if glint_end is None
                    else glint_end
            }
        )

    # ========================================================
    # DARK / BRIGHT
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

    dark_intensity = float(
        avg_intensities[
            dark_index
        ]
    )

    bright_intensity = float(
        avg_intensities[
            bright_index
        ]
    )

    # ========================================================
    # SAVE FIT INFORMATION
    #
    # This is useful for Stage 6.
    # ========================================================

    fit_rows.append(
        {
            "image":
                image_name,

            "eye":
                eye,

            "angle_deg":
                ""
                if angle is None
                else f"{angle:.4f}",

            "average_points":
                len(
                    avg_intensities
                ),

            "average_pixels":
                AVERAGE_PROFILE_PIXELS,

            "glint_points_ignored":
                glint_count,

            "dark_intensity":
                f"{dark_intensity:.4f}",

            "bright_intensity":
                f"{bright_intensity:.4f}",

            "fit_slope":
                f"{fit['slope']:.8f}",

            "fit_intercept":
                f"{fit['intercept']:.8f}",

            "r_squared":
                f"{fit['r2']:.8f}"
        }
    )

    # ========================================================
    # GRAPH
    # ========================================================

    base_name = os.path.splitext(
        os.path.basename(
            image_name
        )
    )[0]

    safe_eye = eye

    graph_name = (
        f"{base_name}_"
        f"{safe_eye}_"
        f"average_fit.png"
    )

    graph_path = os.path.join(
        graph_root,
        GRAPH_FOLDER_NAME,
        graph_name
    )

    graph_profile = {
        "distances":
            avg_distances,

        "intensities":
            avg_intensities,

        "angle":
            angle,

        "fit":
            fit
    }

    save_average_fit_graph(
        graph_profile,
        graph_path,
        image_name,
        eye,
        glint_count
    )

    return True


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    output_path,
    rows,
    fieldnames
):

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True
    )

    with open(
        output_path,
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
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "STAGE 5 - AVERAGE 5 PIXELS + STRAIGHT LINE FIT"
    )
    print("=" * 75)
    print()

    print(
        "Root:"
    )

    print(
        ROOT_HR_FOLDER
    )

    print()

    print(
        f"Average pixels/group : "
        f"{AVERAGE_PROFILE_PIXELS}"
    )

    print(
        "Glint handling       : IGNORE and CONTINUE"
    )

    print(
        "Line fitting         : averaged points"
    )

    print(
        "Final slope stage    : Stage 6"
    )

    print()

    # ========================================================
    # CHECK ROOT
    # ========================================================

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

    # ========================================================
    # FIND STAGE 4 FILES
    # ========================================================

    stage4_files = (
        find_stage4_csv_files(
            ROOT_HR_FOLDER
        )
    )

    if not stage4_files:

        print(
            "[ERROR] No Stage 4 CSV found."
        )

        print()

        print(
            "Expected folder:"
        )

        print(
            "stage4_intensity_profile"
        )

        print()

        print(
            "Expected CSV:"
        )

        print(
            "angle_intensity_values.csv"
        )

        return

    print(
        f"Stage 4 CSV files found: "
        f"{len(stage4_files)}"
    )

    print()

    # ========================================================
    # STAGE 5 OUTPUT
    # ========================================================

    output_root = os.path.join(
        ROOT_HR_FOLDER,
        STAGE5_FOLDER_NAME
    )

    graph_root = output_root

    os.makedirs(
        graph_root,
        exist_ok=True
    )

    average_csv_path = os.path.join(
        output_root,
        AVERAGE_POINTS_CSV
    )

    fit_csv_path = os.path.join(
        output_root,
        FIT_INFORMATION_CSV
    )

    # ========================================================
    # STORAGE
    # ========================================================

    average_rows = []

    fit_rows = []

    total_profiles = 0

    successful_profiles = 0

    # ========================================================
    # PROCESS ALL STAGE 4 CSV FILES
    # ========================================================

    for stage4_csv in stage4_files:

        print(
            "-" * 75
        )

        print(
            "Processing:"
        )

        print(
            stage4_csv
        )

        print()

        rows = read_stage4_csv(
            stage4_csv
        )

        if not rows:

            print(
                "[WARNING] No valid rows."
            )

            continue

        groups = group_profiles(
            rows
        )

        print(
            f"Profiles found: "
            f"{len(groups)}"
        )

        # ----------------------------------------------------
        # PROCESS IMAGE + EYE
        # ----------------------------------------------------

        for (
            image_name,
            eye
        ), profile_rows in groups.items():

            total_profiles += 1

            before = len(
                fit_rows
            )

            success = process_profile(
                profile_rows,
                image_name,
                eye,
                average_rows,
                fit_rows,
                graph_root
            )

            if success:

                successful_profiles += 1

                new_rows = fit_rows[
                    before:
                ]

                if new_rows:

                    result = new_rows[0]

                    print(
                        f"[OK] "
                        f"{eye.upper():<5} "
                        f"{os.path.basename(image_name)} "
                        f"| glint ignored = "
                        f"{result['glint_points_ignored']} "
                        f"| average = "
                        f"{AVERAGE_PROFILE_PIXELS} "
                        f"pixels "
                        f"| averaged points = "
                        f"{result['average_points']} "
                        f"| R² = "
                        f"{float(result['r_squared']):.4f}"
                    )

            else:

                print(
                    f"[SKIP] "
                    f"{eye.upper():<5} "
                    f"{os.path.basename(image_name)}"
                )

        print()

    # ========================================================
    # SAVE AVERAGE CSV
    # ========================================================

    save_csv(
        average_csv_path,
        average_rows,
        [
            "image",
            "eye",
            "angle_deg",
            "average_index",
            "average_distance_px",
            "average_intensity",
            "pixels_used",
            "glint_points_ignored",
            "glint_start_index",
            "glint_end_index"
        ]
    )

    # ========================================================
    # SAVE FIT CSV
    # ========================================================

    save_csv(
        fit_csv_path,
        fit_rows,
        [
            "image",
            "eye",
            "angle_deg",
            "average_points",
            "average_pixels",
            "glint_points_ignored",
            "dark_intensity",
            "bright_intensity",
            "fit_slope",
            "fit_intercept",
            "r_squared"
        ]
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print(
        "STAGE 5 COMPLETED"
    )
    print("=" * 75)
    print()

    print(
        f"Total profiles      : "
        f"{total_profiles}"
    )

    print(
        f"Successful profiles : "
        f"{successful_profiles}"
    )

    print(
        f"Average size        : "
        f"{AVERAGE_PROFILE_PIXELS} pixels"
    )

    print()

    print(
        "Graphs:"
    )

    print(
        os.path.join(
            output_root,
            GRAPH_FOLDER_NAME
        )
    )

    print()

    print(
        "Average points CSV:"
    )

    print(
        average_csv_path
    )

    print()

    print(
        "Fit information CSV:"
    )

    print(
        fit_csv_path
    )

    print()

    print(
        "Stage 5 fits the straight line."
    )

    print(
        "Stage 6 can calculate/report the final slope."
    )

    print()

    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()