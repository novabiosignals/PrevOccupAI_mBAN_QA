# muscleBAN Quality Assessment (MBQA)

A Python toolkit for assessing and analyzing the quality of muscleBAN (mBAN) accelerometer data, with support for automatic segmentation, quality metrics computation, and comprehensive visualization reports.

## 🔍 Overview

The **muscleBAN Quality Assessment (MBQA)** toolkit provides automated quality analysis for accelerometer data collected from muscleBAN wearable sensors. It identifies data quality issues, performs intelligent segmentation based on missing data gaps, and generates detailed quality reports for research and clinical applications.

### Key Capabilities

- **Automatic Data Loading**: Supports both raw text files and NumPy array formats
- **Quality Assessment**: Comprehensive quality metrics including missing data, spikes, flatlines, and variance analysis
- **Intelligent Segmentation**: Automatically segments files with missing data gaps into valid continuous segments
- **Visualization**: Generates detailed PDF reports with time-series plots and quality metrics
- **Batch Processing**: Process multiple subjects, activities, and sessions efficiently
- **Export Functionality**: Export quality-assessed segments as NumPy arrays for downstream analysis

## ✨ Features

### Data Quality Assessment

- **Missing Data Detection**: Identifies and quantifies missing/invalid data points
- **Spike Detection**: Detects abnormal acceleration spikes that may indicate sensor artifacts
- **Flatline Detection**: Identifies periods of constant values indicating sensor malfunction
- **Variance Analysis**: Computes signal variance to assess data quality and information content
- **Statistical Metrics**: Computes mean, standard deviation, min/max values per axis

### Segmentation

- **Gap-Based Segmentation**: Automatically splits files at missing data gaps
- **Minimum Length Validation**: Ensures segments meet minimum sample requirements (default: 60,000 samples = 1 minute @ 1000Hz)
- **Windowed Analysis**: Analyzes data quality in sliding windows for fine-grained assessment
- **Segment Export**: Saves valid segments as NumPy arrays with metadata

### Visualization & Reporting

- **Time-Series Plots**: Visual representation of acceleration data (X, Y, Z axes)
- **Quality Overlay**: Highlights problematic regions (missing data, spikes, flatlines)
- **Statistical Summaries**: Displays key metrics and quality scores
- **PDF Reports**: Generates publication-ready PDF reports per session/segment
- **Batch Summaries**: CSV exports of quality metrics across all processed files

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/eLbARROS13/New_MB_QA.git
cd New_MB_QA
```

2. **Create and activate virtual environment**:

**On Windows (PowerShell)**:
```powershell
python -m venv New_MBQA_Venv
.\New_MBQA_Venv\Scripts\Activate.ps1
```

**On macOS/Linux**:
```bash
python -m venv New_MBQA_Venv
source New_MBQA_Venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Required Packages

- `numpy` - Array operations and numerical computing
- `pandas` - Data manipulation and analysis
- `matplotlib` - Data visualization
- `scipy` - Signal processing and statistical analysis
- `PyPDF2` - PDF report generation
- `tqdm` - Progress bars

## 📁 Project Structure

```
New_MB_QA/
├── mbqa/                           # Main package directory
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # Main workflow orchestration
│   ├── loader.py                   # Data loading utilities
│   ├── quality_assessment.py      # Quality metrics computation
│   ├── viz.py                      # Visualization and reporting
│   ├── constants.py                # Configuration constants
│   ├── adc_a.py                    # ADC to acceleration conversion
│   └── mBANs_noise.py              # Noise analysis utilities
├── exported_segments_npy/          # Output directory for segment arrays
├── reports/                        # Generated PDF reports
├── Excel/                          # Excel output summaries
├── New_MBQA_Venv/                  # Virtual environment
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📝 License

This project is part of research conducted at LibPhys BioSignals (NOVA FCT). Please contact the repository owner for licensing information.

## 👥 Authors

- **eLbARROS13** - *Initial work* - [GitHub](https://github.com/eLbARROS13)

## 🙏 Acknowledgments

- muscleBAN sensor development team
- Research participants and data collection team
- Contributors to the open-source libraries used in this project

## 📧 Contact

For questions, issues, or collaboration inquiries, please open an issue on GitHub or contact the repository maintainers.

---

**Note**: This toolkit is designed for research purposes. Ensure proper ethical approval and data handling procedures are in place when working with human subject data.
