# Malware Detection Tool

## Overview

This Malware Detection Tool is designed to scan files and folders for potential malware using a pre-trained machine learning model. It provides a user-friendly graphical interface to facilitate file selection and display scan results.

## Features

- **File Scanning:** Users can select individual files to scan for malware.
- **Folder Scanning:** Users can select a folder to scan all files within it for malware.
- **Results Display:** The tool provides immediate feedback on whether malware has been detected in the scanned files.

## Prerequisites

Before running the tool, ensure you have the following:

- Python 3.x installed on your system.
- Required libraries:
  - `joblib`
  - `tkinter` (comes with Python standard library)
  - `pefile` (install using `pip install pefile`)

## Installation

1. Clone the repository or download the source code files.
2. Install the required libraries:
```bash
pip install pefile
```
3. Place the trained model file `malware_model.pkl` in the same directory as the script.

## Usage

To run the Malware Detection Tool:

1. Open a terminal or command prompt.
2. Navigate to the directory containing the script.
3. Run the script:
```bash
python malware_detection_tool.py
```
4. Use the GUI to select a file or folder to scan for malware.

## How It Works

1. **Feature Extraction:** The tool extracts features from Portable Executable (PE) files, including:
   - File size
   - Number of sections
   - Entropy of sections

2. **Prediction:** The extracted features are fed into a pre-trained machine learning model to classify the file as either malware or safe.

3. **Results:** The user is informed if malware was detected in the scanned file(s).

## Error Handling

- If the `pefile` library is not installed, the tool will prompt the user to install it and exit.
- If the model file `malware_model.pkl` is missing, the user will receive an error message instructing them to train the model.

## Future Enhancements

- Integration of an API for enhanced malware detection (e.g., VirusTotal).
- Additional features for malware removal.
- Real-time scanning capabilities.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers of the libraries used in this project.
- Inspiration from various sources in the field of cybersecurity and machine learning.
