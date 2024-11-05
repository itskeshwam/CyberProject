# Malware Detection Tool

A machine learning-based tool to detect potential malware in files. This project is aimed at developing a prototype malware detection software using features extracted from executable files. The tool scans files based on extracted features, identifies whether the file is legitimate or malware, and displays the results through a simple graphical user interface (GUI).

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)


---

## Overview

This malware detection tool was created as a college project to provide a basic antivirus-like functionality with on-demand malware scanning capabilities. Using machine learning, it identifies potential malware based on a set of features extracted from executable files, providing insights into the likelihood of malware presence. The project is structured to support adding more advanced functionalities, including real-time protection and integration with external APIs like VirusTotal.

## Features

- **Machine Learning Detection**: Uses a Random Forest Classifier trained on features extracted from a set of known malware and legitimate files.
- **GUI for Scanning**: Simple GUI for selecting and scanning files.
- **On-Demand Scanning**: Allows users to scan files manually by selecting them through the file browser.
- **Modular Design**: Organized code for easy expansion and modification.
  
## Dataset

The model is trained on a dataset of executable files with features extracted for classification. Each file has the following features:

- **Basic File Properties**: Size, alignment, number of sections, entropy of sections.
- **Executable-Specific Attributes**: Import/Export counts, DLL characteristics, address offsets, subsystem info.
- **Resource Information**: Entropy and sizes of resources in the file.
- **Labels**: `legitimate` flag indicating if the file is malware or legitimate.

Example format:
```plaintext
| Name           | md5                           | Machine | SizeOfOptionalHeader | Characteristics | ... | legitimate |
|----------------|-------------------------------|---------|----------------------|-----------------|-----|------------|
| memtest.exe    | 631ea355665f28d4707448e442fbf5b8 | 332     | 224                  | 258             | ... | 1          |
```

## Model Training
The train_model.py script trains the malware detection model and saves it for use in the scanning tool.

1. Load Dataset: Loads MalwareData.csv, removing non-feature columns such as Name and md5.
2. Train-Test Split: Splits the dataset into training and testing subsets.
3. Random Forest Classifier: Fits a Random Forest model to classify files as malware or legitimate.
4. Save Model: Exports the trained model as malware_model.pkl and feature list as selected_features.pkl.

```python
# Training example:
python train_model.py
```

## File Structure
```bash
Cyber Project/
├── build/
│   ├── malware_detection_script/
│   ├── dist/
│       └── malware_detection_script.exe
├── malware_detection_script.spec # PyInstaller spec file
├── malware_detection_tool.py    # Main malware detection tool
├── malware_model.pkl            # Trained model
├── MalwareData.csv              # Dataset
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
└── train_model.py               # Model training script
```
## Installation
Prerequisites
* Python 3.7+
* Required libraries in requirements.txt:
```plaintext
pandas
scikit-learn
joblib
tkinter
```

## Steps
1. Clone Repository:

```bash
git clone https://github.com/yourusername/malware-detection-tool.git
cd malware-detection-tool
```
2. Install Dependencies:

```bash
pip install -r requirements.txt
```
3. Train Model (Optional): If you want to retrain the model, use:

```bash
python train_model.py
```
## Usage
1. Run the Malware Detection Tool:

```bash
python malware_detection_tool.py
```
2. Scan a File:
* Select a file using the GUI to scan it for malware.
* The tool will display whether the file is "legitimate" or "malware".

## Building Executable
To build an executable version of the tool:

1. Run PyInstaller with the .spec file:
```bash
pyinstaller malware_detection_script.spec
```
2. The executable will be generated in the dist folder.

## Future Enhancements
This project is open to enhancements, including:

* Real-Time Scanning: Adding background scanning for real-time malware detection.
* Malware Removal: Incorporating automated quarantine and removal options.
* API Integration: Integrating with VirusTotal API for enhanced malware verification.
* Feature Extraction Improvements: Improving feature extraction to increase model accuracy.
* Alert System: Adding notifications and logs for detected malware.

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch with your feature or bug fix.
3. Make your changes and commit them.
4. Submit a pull request detailing your changes.

