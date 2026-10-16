# Blink Analysis System
A Python-based project for stage-wise eye image processing and blink analysis using OpenCV and computer vision techniques.

## Overview
The Blink Analysis System is designed to process eye images through separate stages. Each stage performs a specific image-processing task to support eye analysis and blink-related research.

The project uses Python and OpenCV to process images, extract eye-related information, and analyze image intensity patterns.

## Features
- Stage-wise image processing.
- Eye image analysis using Python and OpenCV.
- Pupil boundary detection.
- Corneal glint detection.
- Intensity profile extraction.
- Angle and slope analysis.
- Separate Python scripts for individual processing stages.

*Note: Keep only the features that are implemented and tested in your current code.\*

## Technologies Used

- **Python** — programming language.
- **OpenCV** — image processing.
- **NumPy** — numerical and array operations.
- **Matplotlib** — visualization and plotting, if used by the scripts.

## Project Structure

Blink\_Analysis\_System/
│
├── blink\_analysis\_main.py
├── README.md
│
└── stage\_wise\_blink\_analysis/
   │   └── no\_trial\_lens/
   │       ├── stage1\_glint\_detection.py
   │       ├── stage2\_pupil\_boundary.py
   │       ├── stage3\_angle\_detection.py
   │       ├──  stage4\_intensity\_profile.py
   │       ├──  stage5\_average\_fit.py
   │       └──  stage6\_slope\_detection.py
   │
   └── trial\_lens/
        ├── stage1\_pupil\_segmentation.py
        ├── stage2\_pupil\_glint.py
        ├── stage3\_pupil\_angle\_detection.py
        ├── stage4\_intensity\_profile.py
        ├── stage5\_average\_fit.py
        └── stage6\_slope\_detection.py

*The structure above reflects the currently known filenames. Update it if your folders or scripts differ.\*

## Installation
### 1. Install Python
  Install a compatible Python version and verify it:
  python --version

### 2. Clone the repository
  git clone https://github.com/YOUR\_USERNAME/Blink\_Analysis\_System\_V2.git

### 3. Open the project folder
  cd Blink\_Analysis\_System\_V2

### 4. Install dependencies
Install the packages required by your scripts:
  pip install opencv-python numpy matplotlib

If your project has a "requirements.txt" file, install dependencies using:
  pip install -r requirements.txt

## How to Run
Run the main script from the project directory:
  python blink\_analysis\_main.py

Individual stage scripts can be executed separately if they are designed to run independently.
For example:
python stage\_wise\_blink\_analysis/no\_trial\_lens/stage1\_glint.py

Make sure the input image paths and output folders are configured correctly before running the scripts.

## Processing Workflow
The intended workflow is organized into separate stages:



1. Glint Detection — identify the relevant bright reflection in the eye image.

2. Pupil Detection — identify the pupil region or boundary.

3. Angle Analysis — calculate the direction or angle from image information.

4. Intensity Profile Analysis — examine intensity values across the selected region.

5. Average fitting — Analyzes the intensity profiles and fits a straight line to the relevant data to estimate an average   trend.

6. Slope Detection — Calculates and visualizes the final slope from the processed intensity data. The slope results can be used to compare patterns across eye images.

The exact execution order and output of each stage depend on the implementation of the scripts.

## Input and Output

-**Input:** Eye images in the formats supported by the scripts.

- **Processing:** Image processing and numerical analysis.

- **Output:** Processed images, measurements, profiles, or plots, depending on the configured stages.

Ensure that the input paths, output paths, and required image formats are correctly configured.

## Version Control
This project uses Git for version control and GitHub for repository hosting.
#### Common commands:
git status
git add .
git commit -m "Update blink analysis"
git push

## Future Improvements

- Improve detection robustness under different lighting conditions.
- Add validation and error handling for image paths.
- Standardize configuration settings across processing stages.
- Add automated tests and sample outputs.
- Document performance and accuracy using measured results.







