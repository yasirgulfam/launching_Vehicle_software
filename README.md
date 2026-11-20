Launch Vehicle Test Bed – Data Analysis Software

Project Overview

A Python-based data acquisition and analysis application developed for a Launch Vehicle Test Bed project.

The software processes test videos, extracts measurement readings, performs thrust calculations, and generates graphical representations of test results.

The application was developed to support the analysis of multiple test runs.

Features

- Video frame extraction using OpenCV.
- Region of Interest (ROI) selection and image cropping.
- Frame timestamp and data management.
- CSV-based measurement data handling.
- Conversion of mass readings into force.
- Calculation of peak thrust, average thrust, burn time and total impulse.
- Thrust curve generation and visualization.
- Support for multiple test runs.

Technology Stack

- Python
- OpenCV
- NumPy
- Matplotlib
- CSV and JSON

Project Structure

launching_Vehicle_software/
│
├── main.py
├── extraction_data.py
├── graph.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── run1/
│   ├── frames.csv
│   ├── readings1.csv
│   ├── roi.json
│   ├── run_1_results.json
│   ├── run_1_thrust_curve.png
│   └── summary.csv
│
└── run2/
    ├── frames.csv
    ├── readings2.csv
    ├── roi.json
    ├── run_2_results.json
    ├── run_2_thrust_curve.png
    └── summary.csv

Software Architecture

1. main.py

Main application responsible for coordinating the software workflow and processing operations.

2. extraction_data.py

Handles video frame extraction, ROI-based image processing and extraction of measurement-related data.

3. graph.py

Processes measurement data, performs thrust calculations and generates thrust curves and analysis outputs.

Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL

Navigate into the project directory:

cd launching_Vehicle_software

Install the required dependencies:

pip install -r requirements.txt

Execution

Check the available command-line arguments:

python main.py --help

Use the arguments displayed by the application to execute the required processing workflow.

Outputs

The software generates:

- Extracted frame data (CSV).
- Measurement reading files (CSV).
- ROI configuration files (JSON).
- Test result files (JSON).
- Thrust curves (PNG).
- Summary data (CSV).

Project Applications

This software was developed as part of an academic Launch Vehicle Test Bed project, with the objective of supporting experimental data processing and post-test thrust analysis.

Future Improvements

- Automated measurement extraction.
- Improved data validation.
- Real-time data acquisition integration.
- Enhanced visualization and reporting.