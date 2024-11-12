# Malware Detection Tool

A machine learning-based tool to detect potential malware in files. This project uses a Random Forest Classifier to analyze executable files and identify potential malware through feature extraction and analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Implementation Details](#implementation-details)
- [Security Notes](#security-notes)
- [Contributing](#contributing)

## Overview
This malware detection tool provides antivirus-like functionality with on-demand malware scanning capabilities. Using machine learning, it analyzes executable files by extracting key features and predicting the likelihood of malware presence. The tool is designed for educational purposes and demonstrates the application of machine learning in cybersecurity.

## Features
- **GUI-Based Scanning**: User-friendly interface for file and folder scanning
- **Machine Learning Detection**: Random Forest Classifier trained on executable file features
- **Detailed Analysis**: Extracts and analyzes multiple file characteristics including:
  - Section entropy analysis
  - Import/Export table examination
  - Header characteristic evaluation
  - Resource analysis
- **PDF Report Generation**: Creates detailed scan reports in PDF format
- **Batch Scanning**: Support for scanning entire directories
- **Progress Tracking**: Real-time progress monitoring for scan operations
- **Extensible Architecture**: Modular design for easy feature additions

## Project Structure
```
Cyber Project/
├── build/
│   └── malware_detection_script/
│       └── localpycs/
├── dist/
│   └── malware_detection_script.exe
├── Analysis-00.toc
├── base_library.zip
├── EXE-00.toc
├── feature_importance.png      # Visualization of feature importance
├── feature_info.pkl           # Serialized feature information
├── feature_scaler.pkl         # Trained feature scaler
├── malware_detection_script.exe
├── malware_detection_script.pkg
├── malware_detection_script.spec
├── malware_detection_tool.py   # Main application file
├── malware_model.pkl          # Trained machine learning model
├── MalwareData.csv           # Training dataset
├── model_performance.png      # Model performance visualizations
├── PKG-00.toc
├── PYZ-00.pyz
├── PYZ-00.toc
├── README.md
├── requirements.txt
├── selected_features.pkl      # Selected feature list
├── train_model.py            # Model training script
├── warn-malware_detection_script.txt
└── xref-malware_detection_script.html
```

## Installation

### Prerequisites
- Python 3.7+
- Git (for cloning the repository)
- 4GB RAM minimum
- Windows/Linux/MacOS compatible

### Steps

1. Clone the repository:
```bash
git clone [repository-url]
cd malware-detection-tool
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
python malware_detection_tool.py
```

### Building from Source
To create an executable:
```bash
pyinstaller malware_detection_script.spec
```

## Usage

### GUI Application
1. Launch the application using `python malware_detection_tool.py`
2. Use the interface buttons to:
   - Scan individual files
   - Scan entire folders
   - Generate PDF reports of scan results

### Command Line Usage
```bash
python malware_detection_tool.py [file_path]
```

### Training Custom Models
1. Prepare your dataset in CSV format with the following columns:
   - Machine
   - SizeOfOptionalHeader
   - Characteristics
   - Other PE file features
2. Run the training script:
```bash
python train_model.py
```

## Model Training
The model training process includes:
- Feature extraction from executable files
- Random Forest Classification with optimized parameters
- Cross-validation for accuracy assessment
- Feature importance analysis
- Performance metrics visualization
- Model serialization for deployment

### Feature Selection
The model uses key features including:
- Section entropy metrics
- File header characteristics
- Import/Export table details
- Resource section properties

## Implementation Details

### Main Components
1. **MalwareDetector Class**: Core detection engine
   - Feature extraction
   - Model prediction
   - Result analysis

2. **MalwareDetectorGUI Class**: User interface
   - File/Folder selection
   - Progress tracking
   - Result display

3. **Report Generation**:
   - PDF report creation
   - Scan history tracking
   - Detailed analysis output

### Performance Metrics
- Detection accuracy: Based on cross-validation
- False positive rate: Minimized through feature selection
- Processing speed: Optimized for regular usage

## Security Notes
- This tool is designed for educational and research purposes
- It should not replace professional antivirus software
- Always use updated antivirus software for real protection
- Handle suspicious files with caution
- Run scans in a controlled environment

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Maintain modularity

### Testing
Run tests before submitting pull requests:
```bash
python -m unittest tests/
```

## License
[Add Your License Here]

## Acknowledgments
- List contributors
- Reference papers/resources used
- Credit external libraries

## Disclaimer
This project is for educational purposes only. The authors are not responsible for any misuse or damage caused by this tool.

**IMPORTANT NOTE**: The file `eicar.com` included in this project is a MOCK MALWARE FILE used for testing antivirus software. While it is completely harmless and cannot damage your system, please handle it with caution as it may trigger antivirus warnings. This file follows the EICAR Standard Anti-Virus Test File specifications.