import os
import csv
import math
import cv2
import numpy as np


# ============================================================
# STAGE 4
# INTENSITY PROFILE ALONG STAGE 3 DETECTED ANGLE
# ============================================================
#
# INPUT:
#
#   Stage 2:
#       pupil_raw_contour_points.csv
#
#   Stage 3:
#       pupil_angles.csv
#
#   Original images
#
# OUTPUT:
#
#   stage4_intensity_profile/
#       graphs/
#           image_left.png
#           image_right.png
#
#       angle_intensity_values.csv
#
# ============================================================


# ============================================================
# STAGE FOLDER NAMES
# ============================================================

STAGE2_FOLDER = "stage2_raw_contour"

STAGE3_FOLDER = "stage3_angle"

STAGE4_FOLDER = "stage4_intensity_profile"


# ============================================================
# CSV FILE NAMES
# ============================================================

STAGE2_CSV_NAME = "pupil_raw_contour_points.csv"

STAGE3_CSV_NAME = "pupil_angles.csv"

STAGE4_CSV_NAME = "angle_intensity_values.csv"


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
# PROFILE SETTINGS
# ============================================================

INTENSITY_PROFILE_SAMPLES = 101

INTENSITY_PROFILE_HALF_LENGTH_FRAC = 0.75

AVERAGE_PROFILE_PIXELS = 5


# ============================================================
# OUTPUT SETTINGS
# ============================================================

SAVE_GRAPHS = True

SAVE_CSV = True


# ============================================================
# FIND FILES RECURSIVELY
# ============================================================

def find_files_recursive(
    root_folder,
    target_name
):

    found_files = []

    for current_root, dirs, files in os.walk(
        root_folder
    ):

        dirs[:] = [
            d for d in dirs
            if d.lower() != STAGE4_FOLDER.lower()
        ]

        for file_name in files:

            if file_name.lower() == target_name.lower():

                found_files.append(
                    os.path.join(
                        current_root,
                        file_name
                    )
                )

    return sorted(found_files)


# ============================================================
# FIND ORIGINAL IMAGES RECURSIVELY
# ============================================================

def find_all_images(
    root_folder
):

    image_files = []

    for current_root, dirs, files in os.walk(
        root_folder
    ):

        dirs[:] = [
            d for d in dirs
            if d.lower() not in (
                STAGE2_FOLDER.lower(),
                STAGE3_FOLDER.lower(),
                STAGE4_FOLDER.lower()
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

def normalize_name(
    name
):

    return os.path.basename(
        str(name)
    ).strip().lower()


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

def build_image_index(
    image_files
):

    index = {}

    for path in image_files:

        name = normalize_name(
            os.path.basename(path)
        )

        if name not in index:

            index[name] = []

        index[name].append(
            path
        )

    return index


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(
    image_name,
    image_index,
    dataset_folder
):

    key = normalize_name(
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

    if len(local_candidates) == 1:

        return local_candidates[0]

    if local_candidates:

        local_candidates.sort(
            key=lambda p: len(
                os.path.relpath(
                    p,
                    dataset_folder
                ).split(
                    os.sep
                )
            )
        )

        return local_candidates[0]

    if len(candidates) == 1:

        return candidates[0]

    return candidates[0]


# ============================================================
# LOAD STAGE 2 RAW PUPIL CONTOURS
# ============================================================

def load_stage2_contours(
    csv_path
):

    contours = {}

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            image_name = normalize_name(
                row.get(
                    "image",
                    ""
                )
            )

            eye = row.get(
                "eye",
                ""
            ).strip().upper()

            if eye in (
                "LEFT",
                "LE"
            ):

                eye = "left"

            elif eye in (
                "RIGHT",
                "RE"
            ):

                eye = "right"

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
                image_name,
                eye
            )

            if key not in contours:

                contours[key] = []

            contours[key].append(
                (
                    x,
                    y
                )
            )

    return contours


# ============================================================
# LOAD STAGE 3 ANGLES
# ============================================================

def load_stage3_angles(
    csv_path
):

    angles = {}

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            image_name = normalize_name(
                row.get(
                    "image",
                    ""
                )
            )

            if not image_name:

                continue

            def read_float(
                key
            ):

                value = row.get(
                    key,
                    ""
                )

                if value in (
                    "",
                    None
                ):

                    return None

                try:

                    return float(
                        value
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    return None

            angles[image_name] = {

                "left_angle":
                    read_float(
                        "left_angle_deg"
                    ),

                "right_angle":
                    read_float(
                        "right_angle_deg"
                    ),

                "common_angle":
                    read_float(
                        "common_angle_deg"
                    )
            }

    return angles


# ============================================================
# CALCULATE PUPIL CENTER
# ============================================================

def calculate_pupil_center(
    contour
):

    points = np.asarray(
        contour,
        dtype=np.float32
    )

    if len(points) < 3:

        return None

    points_for_moments = points.reshape(
        (-1, 1, 2)
    )

    moments = cv2.moments(
        points_for_moments
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
                points[:, 0]
            )
        )

        cy = float(
            np.mean(
                points[:, 1]
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

    points = np.asarray(
        contour,
        dtype=np.float32
    )

    cx, cy = center

    distances = np.hypot(
        points[:, 0] - cx,
        points[:, 1] - cy
    )

    if len(distances) == 0:

        return 0.0

    return float(
        np.median(
            distances
        )
    )


# ============================================================
# CREATE PUPIL MASK
# ============================================================

def create_pupil_mask(
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

    points = np.asarray(
        contour,
        dtype=np.float32
    )

    points = np.round(
        points
    ).astype(
        np.int32
    )

    points = points.reshape(
        (-1, 1, 2)
    )

    cv2.fillPoly(
        mask,
        [points],
        255
    )

    return mask


# ============================================================
# AVERAGE PROFILE PIXELS
# ============================================================

def average_profile_pixels(
    distances,
    xs,
    ys,
    intensities,
    group_size
):

    if len(
        intensities
    ) == 0:

        return {
            "distances": np.array([]),
            "xs": np.array([]),
            "ys": np.array([]),
            "intensities": np.array([])
        }

    output_distances = []
    output_xs = []
    output_ys = []
    output_intensities = []

    for start in range(
        0,
        len(intensities),
        group_size
    ):

        end = min(
            start + group_size,
            len(intensities)
        )

        output_distances.append(
            float(
                np.mean(
                    distances[
                        start:end
                    ]
                )
            )
        )

        output_xs.append(
            float(
                np.mean(
                    xs[
                        start:end
                    ]
                )
            )
        )

        output_ys.append(
            float(
                np.mean(
                    ys[
                        start:end
                    ]
                )
            )
        )

        output_intensities.append(
            float(
                np.mean(
                    intensities[
                        start:end
                    ]
                )
            )
        )

    return {
        "distances": np.asarray(
            output_distances,
            dtype=np.float64
        ),

        "xs": np.asarray(
            output_xs,
            dtype=np.float64
        ),

        "ys": np.asarray(
            output_ys,
            dtype=np.float64
        ),

        "intensities": np.asarray(
            output_intensities,
            dtype=np.float64
        )
    }


# ============================================================
# EXTRACT INTENSITY PROFILE
# ============================================================

def extract_intensity_profile(
    gray,
    contour,
    angle_deg
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

    cx, cy = center

    radius = calculate_pupil_radius(
        contour,
        center
    )

    if radius <= 1:

        return None

    angle_rad = math.radians(
        float(angle_deg)
    )

    dx = math.cos(
        angle_rad
    )

    dy = math.sin(
        angle_rad
    )

    half_length = (
        INTENSITY_PROFILE_HALF_LENGTH_FRAC
        *
        radius
    )

    distances = np.linspace(
        -half_length,
        half_length,
        INTENSITY_PROFILE_SAMPLES
    )

    xs = (
        cx
        +
        distances * dx
    )

    ys = (
        cy
        +
        distances * dy
    )

    height, width = gray.shape[:2]

    image_valid = (
        (xs >= 0)
        &
        (xs <= width - 1)
        &
        (ys >= 0)
        &
        (ys <= height - 1)
    )

    distances = distances[
        image_valid
    ]

    xs = xs[
        image_valid
    ]

    ys = ys[
        image_valid
    ]

    if len(xs) < 3:

        return None

    pupil_mask = create_pupil_mask(
        gray,
        contour
    )

    inside_pupil = []

    for x, y in zip(
        xs,
        ys
    ):

        xi = int(
            round(x)
        )

        yi = int(
            round(y)
        )

        if (
            0 <= xi < width
            and
            0 <= yi < height
            and
            pupil_mask[yi, xi] > 0
        ):

            inside_pupil.append(
                True
            )

        else:

            inside_pupil.append(
                False
            )

    inside_pupil = np.asarray(
        inside_pupil,
        dtype=bool
    )

    distances = distances[
        inside_pupil
    ]

    xs = xs[
        inside_pupil
    ]

    ys = ys[
        inside_pupil
    ]

    if len(xs) < 3:

        return None

    intensities = np.asarray(
        cv2.remap(
            gray,
            xs.astype(
                np.float32
            ),
            ys.astype(
                np.float32
            ),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        ),
        dtype=np.float64
    ).reshape(-1)

    if len(
        intensities
    ) < 3:

        return None

    averaged = average_profile_pixels(
        distances,
        xs,
        ys,
        intensities,
        AVERAGE_PROFILE_PIXELS
    )

    distances = averaged[
        "distances"
    ]

    xs = averaged[
        "xs"
    ]

    ys = averaged[
        "ys"
    ]

    intensities = averaged[
        "intensities"
    ]

    if len(
        intensities
    ) < 2:

        return None

    dark_index = int(
        np.argmin(
            intensities
        )
    )

    bright_index = int(
        np.argmax(
            intensities
        )
    )

    dark_intensity = float(
        intensities[
            dark_index
        ]
    )

    bright_intensity = float(
        intensities[
            bright_index
        ]
    )

    slope = None
    intercept = None
    r2 = None

    if len(
        intensities
    ) >= 2:

        slope, intercept = np.polyfit(
            distances,
            intensities,
            1
        )

        fitted = (
            slope
            *
            distances
            +
            intercept
        )

        ss_res = np.sum(
            (
                intensities
                -
                fitted
            ) ** 2
        )

        ss_tot = np.sum(
            (
                intensities
                -
                np.mean(
                    intensities
                )
            ) ** 2
        )

        if ss_tot > 1e-12:

            r2 = (
                1.0
                -
                ss_res / ss_tot
            )

        else:

            r2 = 1.0

    return {

        "angle_deg":
            float(angle_deg),

        "distances":
            distances,

        "xs":
            xs,

        "ys":
            ys,

        "intensities":
            intensities,

        "dark_index":
            dark_index,

        "bright_index":
            bright_index,

        "dark_intensity":
            dark_intensity,

        "bright_intensity":
            bright_intensity,

        "slope":
            None
            if slope is None
            else float(slope),

        "intercept":
            None
            if intercept is None
            else float(intercept),

        "r2":
            None
            if r2 is None
            else float(r2)
    }


# ============================================================
# SAVE GRAPH
# ============================================================

def save_graph(
    profile,
    output_path,
    image_name,
    eye
):

    import matplotlib

    matplotlib.use(
        "Agg"
    )

    import matplotlib.pyplot as plt

    distances = np.asarray(
        profile["distances"],
        dtype=float
    )

    intensities = np.asarray(
        profile["intensities"],
        dtype=float
    )

    if len(
        distances
    ) == 0:

        return

    dark_index = int(
        profile["dark_index"]
    )

    bright_index = int(
        profile["bright_index"]
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        distances,
        intensities,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Intensity profile"
    )

    plt.scatter(
        [
            distances[
                dark_index
            ]
        ],
        [
            intensities[
                dark_index
            ]
        ],
        s=70,
        label=(
            f"Dark point = "
            f"{intensities[dark_index]:.1f}"
        )
    )

    plt.scatter(
        [
            distances[
                bright_index
            ]
        ],
        [
            intensities[
                bright_index
            ]
        ],
        s=70,
        label=(
            f"Bright point = "
            f"{intensities[bright_index]:.1f}"
        )
    )

    slope = profile.get(
        "slope"
    )

    intercept = profile.get(
        "intercept"
    )

    if (
        slope is not None
        and
        intercept is not None
    ):

        order = np.argsort(
            distances
        )

        sorted_distances = (
            distances[
                order
            ]
        )

        fitted = (
            slope
            *
            sorted_distances
            +
            intercept
        )

        plt.plot(
            sorted_distances,
            fitted,
            linewidth=2,
            label=(
                f"Overall slope = "
                f"{slope:.4f}"
            )
        )

    plt.xlabel(
        "Distance along Stage 3 angle (pixels)"
    )

    plt.ylabel(
        "Grayscale intensity"
    )

    plt.title(
        f"{eye.upper()} Eye Intensity Profile\n"
        f"Angle = {profile['angle_deg']:.2f}°"
    )

    plt.ylim(
        0,
        255
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=160
    )

    plt.close()


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    stage3_csv_path,
    stage2_csv_path,
    image_index,
    all_rows
):

    print()
    print("=" * 80)
    print("STAGE 4 DATASET")
    print("=" * 80)

    print(
        "Stage 3:"
    )

    print(
        stage3_csv_path
    )

    print(
        "Stage 2:"
    )

    print(
        stage2_csv_path
    )

    stage3_folder = os.path.dirname(
        stage3_csv_path
    )

    dataset_folder = os.path.dirname(
        stage3_folder
    )

    contours = load_stage2_contours(
        stage2_csv_path
    )

    angles = load_stage3_angles(
        stage3_csv_path
    )

    output_folder = os.path.join(
        dataset_folder,
        STAGE4_FOLDER
    )

    graph_folder = os.path.join(
        output_folder,
        "graphs"
    )

    if SAVE_GRAPHS:

        os.makedirs(
            graph_folder,
            exist_ok=True
        )

    processed_profiles = 0

    for image_name, angle_data in angles.items():

        image_path = find_image(
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

        gray = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if gray is None:

            print(
                f"[WARNING] Cannot read: "
                f"{image_path}"
            )

            continue

        display_name = os.path.basename(
            image_path
        )

        normalized_image_name = normalize_name(
            image_name
        )

        left_angle = angle_data.get(
            "left_angle"
        )

        left_contour = contours.get(
            (
                normalized_image_name,
                "left"
            )
        )

        if (
            left_angle is not None
            and
            left_contour is not None
        ):

            profile = extract_intensity_profile(
                gray,
                left_contour,
                left_angle
            )

            if profile is not None:

                if SAVE_GRAPHS:

                    graph_name = (
                        os.path.splitext(
                            display_name
                        )[0]
                        +
                        "_left.png"
                    )

                    graph_path = os.path.join(
                        graph_folder,
                        graph_name
                    )

                    save_graph(
                        profile,
                        graph_path,
                        display_name,
                        "left"
                    )

                for i in range(
                    len(
                        profile["distances"]
                    )
                ):

                    all_rows.append(
                        {
                            "image":
                                display_name,

                            "source_image":
                                image_path,

                            "eye":
                                "left",

                            "angle_deg":
                                f"{profile['angle_deg']:.4f}",

                            "distance_px":
                                f"{profile['distances'][i]:.4f}",

                            "x":
                                f"{profile['xs'][i]:.4f}",

                            "y":
                                f"{profile['ys'][i]:.4f}",

                            "intensity":
                                f"{profile['intensities'][i]:.4f}",

                            "dark_intensity":
                                f"{profile['dark_intensity']:.4f}",

                            "bright_intensity":
                                f"{profile['bright_intensity']:.4f}",

                            "slope":
                                ""
                                if profile["slope"] is None
                                else
                                f"{profile['slope']:.6f}",

                            "r2":
                                ""
                                if profile["r2"] is None
                                else
                                f"{profile['r2']:.6f}"
                        }
                    )

                processed_profiles += 1

                print(
                    f"[OK] LEFT  | "
                    f"{display_name} | "
                    f"angle = "
                    f"{left_angle:.2f}°"
                )

        right_angle = angle_data.get(
            "right_angle"
        )

        right_contour = contours.get(
            (
                normalized_image_name,
                "right"
            )
        )

        if (
            right_angle is not None
            and
            right_contour is not None
        ):

            profile = extract_intensity_profile(
                gray,
                right_contour,
                right_angle
            )

            if profile is not None:

                if SAVE_GRAPHS:

                    graph_name = (
                        os.path.splitext(
                            display_name
                        )[0]
                        +
                        "_right.png"
                    )

                    graph_path = os.path.join(
                        graph_folder,
                        graph_name
                    )

                    save_graph(
                        profile,
                        graph_path,
                        display_name,
                        "right"
                    )

                for i in range(
                    len(
                        profile["distances"]
                    )
                ):

                    all_rows.append(
                        {
                            "image":
                                display_name,

                            "source_image":
                                image_path,

                            "eye":
                                "right",

                            "angle_deg":
                                f"{profile['angle_deg']:.4f}",

                            "distance_px":
                                f"{profile['distances'][i]:.4f}",

                            "x":
                                f"{profile['xs'][i]:.4f}",

                            "y":
                                f"{profile['ys'][i]:.4f}",

                            "intensity":
                                f"{profile['intensities'][i]:.4f}",

                            "dark_intensity":
                                f"{profile['dark_intensity']:.4f}",

                            "bright_intensity":
                                f"{profile['bright_intensity']:.4f}",

                            "slope":
                                ""
                                if profile["slope"] is None
                                else
                                f"{profile['slope']:.6f}",

                            "r2":
                                ""
                                if profile["r2"] is None
                                else
                                f"{profile['r2']:.6f}"
                        }
                    )

                processed_profiles += 1

                print(
                    f"[OK] RIGHT | "
                    f"{display_name} | "
                    f"angle = "
                    f"{right_angle:.2f}°"
                )

    print()
    print(
        f"Profiles generated: "
        f"{processed_profiles}"
    )


# ============================================================
# SAVE FINAL CSV
# ============================================================

def save_csv(
    root_folder,
    rows
):

    if not rows:

        print()
        print(
            "[WARNING] No Stage 4 data generated."
        )

        return

    output_folder = os.path.join(
        root_folder,
        STAGE4_FOLDER
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output_csv = os.path.join(
        output_folder,
        STAGE4_CSV_NAME
    )

    fields = [
        "image",
        "source_image",
        "eye",
        "angle_deg",
        "distance_px",
        "x",
        "y",
        "intensity",
        "dark_intensity",
        "bright_intensity",
        "slope",
        "r2"
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
    print(
        "Stage 4 CSV saved:"
    )

    print(
        output_csv
    )


# ============================================================
# MAIN (INTERACTIVE TERMINAL INPUT)
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 4 - INTENSITY PROFILE (INTERACTIVE INPUT)")
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
        os.path.abspath(os.path.expanduser(path))
        for path in input_paths_list
    ]

    for index, folder in enumerate(input_folders, start=1):
        if not os.path.isdir(folder):
            print()
            print(f"[ERROR] Input path {index} does not exist:")
            print(folder)
            print()
            return

    stage2_csvs = []
    stage3_csvs = []
    image_files = []

    for folder in input_folders:
        s2_found = find_files_recursive(folder, STAGE2_CSV_NAME)
        if s2_found:
            stage2_csvs.extend(s2_found)
        else:
            p = os.path.join(folder, STAGE2_FOLDER, STAGE2_CSV_NAME)
            if os.path.isfile(p):
                stage2_csvs.append(p)

        s3_found = find_files_recursive(folder, STAGE3_CSV_NAME)
        if s3_found:
            stage3_csvs.extend(s3_found)
        else:
            p = os.path.join(folder, STAGE3_FOLDER, STAGE3_CSV_NAME)
            if os.path.isfile(p):
                stage3_csvs.append(p)

        img_found = find_all_images(folder)
        if img_found:
            image_files.extend(img_found)

    stage2_csvs = sorted(set(stage2_csvs))
    stage3_csvs = sorted(set(stage3_csvs))
    image_files = sorted(set(image_files))

    print()
    print(f"Stage 2 CSV files found : {len(stage2_csvs)}")
    print(f"Stage 3 CSV files found : {len(stage3_csvs)}")
    print(f"Original images found   : {len(image_files)}")

    if not stage2_csvs:
        print()
        print("[ERROR] Stage 2 pupil_raw_contour_points.csv was not found.")
        return

    if not stage3_csvs:
        print()
        print("[ERROR] Stage 3 pupil_angles.csv was not found. Run stage3_angle.py first.")
        return

    if not image_files:
        print()
        print("[ERROR] Original images not found.")
        return

    image_index = build_image_index(image_files)

    all_rows = []

    for stage3_csv in stage3_csvs:

        stage3_folder = os.path.dirname(
            stage3_csv
        )

        dataset_folder = os.path.dirname(
            stage3_folder
        )

        expected_stage2_csv = os.path.join(
            dataset_folder,
            STAGE2_FOLDER,
            STAGE2_CSV_NAME
        )

        if os.path.isfile(
            expected_stage2_csv
        ):

            stage2_csv = (
                expected_stage2_csv
            )

        else:

            matching_stage2 = []

            for candidate in stage2_csvs:

                candidate_stage2_folder = (
                    os.path.dirname(
                        candidate
                    )
                )

                candidate_dataset_folder = (
                    os.path.dirname(
                        candidate_stage2_folder
                    )
                )

                if os.path.abspath(
                    candidate_dataset_folder
                ) == os.path.abspath(
                    dataset_folder
                ):

                    matching_stage2.append(
                        candidate
                    )

            if not matching_stage2:

                print()
                print(
                    "[WARNING] Matching Stage 2 CSV "
                    "not found:"
                )

                print(
                    stage3_csv
                )

                continue

            stage2_csv = (
                matching_stage2[0]
            )

        process_dataset(
            stage3_csv,
            stage2_csv,
            image_index,
            all_rows
        )

    if SAVE_CSV and all_rows:
        for folder in input_folders:
            save_csv(folder, all_rows)

    print()
    print("=" * 80)
    print("STAGE 4 COMPLETED")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()