<div align="center">

# 💬 Facial Expression Recognition: A Machine Learning Approach

### A Comprehensive Study with SVM, Random Forest, KNN, and Decision Tree Using Grid Search Method

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-blue?logo=scikit-learn)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green?logo=opencv)](https://opencv.org/)
[![Conference](https://img.shields.io/badge/NCSP'25-RISP_Workshop-FF6B6B)](https://ncsp.org/)
[![Status](https://img.shields.io/badge/Status-Published-brightgreen)](https://ncsp.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Conference:** NCSP'25 RISP International Workshop on Nonlinear Circuits, Communications and Signal Processing 2025  
**Date:** February 27 - March 2, 2025 | **Location:** Pulau Pinang, Malaysia
</div>

---

## 📑 Table of Contents

- [🎯 Abstract](#abstract)
- [🏆 Key Achievements](#key-achievements)
- [📊 Model Performance](#model-performance)
- [🛠️ Technology Stack](#technology-stack)
- [📁 Project Structure](#project-structure)
- [👥 Authors](#authors)
- [🔬 Methodology](#methodology)
- [📈 Dataset Details](#dataset-details)
- [⭐ Results Highlights](#results-highlights)
- [💻 Installation & Usage](#installation--usage)
- [📝 Citation](#citation)
- [📧 Contact](#contact)

---

## 🎯 Abstract

Facial expression recognition (FER) serves as a vital interface for bridging human emotions and machine understanding, enabling applications across psychology, healthcare, and human-computer interaction. This study explores the performance of machine learning classifiers—SVM, Random Forest, KNN, and Decision Tree—on the CK+ dataset, a benchmark for FER research. Preprocessing techniques, such as grayscale conversion and histogram equalization, were employed to enhance feature clarity. Features extracted via Histogram of Oriented Gradients (HOG) were evaluated using k-fold cross-validation. SVM emerged as the most accurate classifier, achieving a 100% recognition rate with a linear kernel, while Random Forest demonstrated robust but slightly inferior performance. Decision Tree and KNN exhibited lower accuracies, highlighting the tradeoffs between interpretability and performance. These findings underline the potential of SVM for designing reliable and efficient FER systems suitable for practical applications.

---

## 🏆 Key Achievements

| Achievement | Details |
|---|---|
| **Best Accuracy** | 100% (SVM with Linear Kernel) |
| **Highest Performer** | Support Vector Machine (SVM) |
| **Dataset Used** | CK+ (Cohn-Kanade+) |
| **Feature Extraction** | Histogram of Oriented Gradients (HOG) |
| **Validation Method** | k-fold Cross-Validation with Grid Search |
| **Real-world Applicability** | High reliability and efficiency for FER systems |

---

## 📊 Model Performance

### Overall Accuracy Comparison

```
┌──────────────────┬──────────┐
│     Model        │ Accuracy │
├──────────────────┼──────────┤
│ SVM (Linear)     │  100.0%  │
│ Random Forest    │   97.0%  │
│ KNN              │   92.0%  │
│ Decision Tree    │   79.0%  │
└──────────────────┴──────────┘
```

### Table 1: Classification Report for SVM (Linear Kernel)

| Label | Precision (%) | Recall (%) | F1-score (%) | Support |
|-------|---|---|---|---|
| Anger | 100 | 100 | 100 | 27 |
| Contempt | 100 | 100 | 100 | 11 |
| Disgust | 100 | 100 | 100 | 35 |
| Fear | 100 | 100 | 100 | 15 |
| Happy | 100 | 100 | 100 | 42 |
| Sadness | 100 | 100 | 100 | 17 |
| Surprise | 100 | 100 | 100 | 50 |
| **Macro Average** | **100** | **100** | **100** | **197** |
| **Weighted Average** | **100** | **100** | **100** | **197** |
| **Overall Accuracy** | | | | **100% (197)** |

### Table 2: SVM Performance with Different Kernels

| Train-Test Ratio | Linear Kernel (%) | Polynomial Kernel (%) | RBF Kernel (%) |
|---|---|---|---|
| 10% Test Set | 100.00 | 81.81 | 90.90 |
| 20% Test Set | 100.00 | 77.15 | 85.27 |
| 30% Test Set | 98.98 | 72.20 | 87.79 |
| 40% Test Set | 96.18 | 71.24 | 85.24 |
| 50% Test Set | 92.87 | 69.24 | 83.50 |

### Table 3: Classification Report for Decision Tree

| Label | Precision (%) | Recall (%) | F1-score (%) | Support |
|-------|---|---|---|---|
| Anger | 75 | 78 | 76 | 27 |
| Contempt | 86 | 55 | 67 | 11 |
| Disgust | 71 | 86 | 78 | 35 |
| Fear | 91 | 67 | 77 | 15 |
| Happy | 73 | 79 | 76 | 42 |
| Sadness | 83 | 59 | 69 | 17 |
| Surprise | 87 | 90 | 88 | 50 |
| **Macro Average** | **81** | **73** | **76** | **197** |
| **Weighted Average** | **79** | **79** | **78** | **197** |
| **Overall Accuracy** | | | | **79% (197)** |

### Table 4: Classification Report for KNN

| Label | Precision (%) | Recall (%) | F1-score (%) | Support |
|-------|---|---|---|---|
| Anger | 100 | 89 | 94 | 27 |
| Contempt | 92 | 100 | 96 | 11 |
| Disgust | 83 | 86 | 85 | 35 |
| Fear | 100 | 93 | 97 | 15 |
| Happy | 91 | 93 | 92 | 42 |
| Sadness | 94 | 94 | 94 | 17 |
| Surprise | 94 | 96 | 95 | 50 |
| **Macro Average** | **93** | **93** | **93** | **197** |
| **Weighted Average** | **93** | **92** | **92** | **197** |
| **Overall Accuracy** | | | | **92% (197)** |

### Table 5: Classification Report for Random Forest

| Label | Precision (%) | Recall (%) | F1-score (%) | Support |
|-------|---|---|---|---|
| Anger | 100 | 96 | 98 | 27 |
| Contempt | 100 | 100 | 100 | 11 |
| Disgust | 100 | 91 | 96 | 35 |
| Fear | 100 | 100 | 100 | 15 |
| Happy | 93 | 100 | 97 | 42 |
| Sadness | 94 | 94 | 94 | 17 |
| Surprise | 98 | 100 | 99 | 50 |
| **Macro Average** | **98** | **97** | **98** | **197** |
| **Weighted Average** | **98** | **97** | **97** | **197** |
| **Overall Accuracy** | | | | **97% (197)** |

---

## 🛠️ Technology Stack

### Programming Language & Environment
- **Python** 3.8+
- **Jupyter Notebook** for interactive development

### Core Machine Learning Framework
- **scikit-learn** 1.0+ - ML algorithms (SVM, Random Forest, KNN, Decision Tree)
- **scikit-image** - Image processing and HOG feature extraction

### Data Processing & Visualization
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **OpenCV (cv2)** - Image processing and grayscale conversion

### Additional Libraries
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualizations
- **Joblib** - Model serialization and parallel computing

### Domain
**Computer Vision | Image Classification | Facial Expression Recognition**

---

## 📁 Project Structure

```
Facial emotion with SVM/
├── README.md                          # This file
├── Code/
│   ├── svm.ipynb                      # SVM model implementation
│   ├── Randomforest.ipynb             # Random Forest model
│   ├── KNN.ipynb                      # K-Nearest Neighbors model
│   ├── decision_tree.ipynb            # Decision Tree model
│   ├── linear_regression.ipynb        # Baseline regression model
│   ├── logistic_regression.ipynb      # Logistic regression model
│   ├── ridge_regression.ipynb         # Ridge regression model
│   ├── lasso.ipynb                    # Lasso regression model
│   └── loss_table.csv                 # Loss metrics data
├── Dataset/
│   └── CK+48/                         # CK+ Dataset (7 emotions)
│       ├── anger/
│       ├── contempt/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── sadness/
│       └── surprise/
├── Result/
│   ├── accuracy_results.csv           # Accuracy metrics
│   └── accuracy_results_for_research.csv
└── Paper/                             # Research paper documents
```

---

## 👥 Authors

| # | Name | Role | Affiliation |
|---|---|---|---|
| 1 | **Md. Uzzal Mia** | Primary Researcher & Developer | Pabna University of Science and Technology |
| 2 | Md. Rifat Hossen | Co-Researcher | Pabna University of Science and Technology |
| 3 | Rinku Islam | Co-Researcher | Pabna University of Science and Technology |
| 4 | Md. Sarwar Hosain | Co-Researcher | Pabna University of Science and Technology |
| 5 | Mohammad Kamrul Hasan | Supervisor | Pabna University of Science and Technology |
| 6 | Tetsuya Shimamura | International Collaborator | Tokyo Metropolitan University, Japan |

---

## 🔬 Methodology

### Step 1: Data Preprocessing
- **Grayscale Conversion**: Convert RGB images to grayscale for reduced dimensionality
- **Histogram Equalization**: Enhance image contrast to improve feature distinctiveness
- **Normalization**: Normalize pixel values to [0, 1] range

### Step 2: Feature Extraction
- **Histogram of Oriented Gradients (HOG)**: Extract orientation and gradient information
- **HOG Configuration**: 
  - Block size: 16×16 pixels
  - Cell size: 4×4 pixels
  - Number of bins: 9 (orientation bins)
  - Feature dimension: 1,764 features per image

### Step 3: Model Training with Grid Search
- **Grid Search Hyperparameters**:
  - **SVM**: Kernel types (linear, poly, rbf), C values, gamma values
  - **Random Forest**: Number of estimators, max depth, min samples split
  - **KNN**: Number of neighbors (k), distance metrics
  - **Decision Tree**: Max depth, min samples split, criterion

### Step 4: Cross-Validation Strategy
- **k-fold Cross-Validation** (k=5):
  - Ensures robust evaluation across diverse data splits
  - Reduces variance in performance metrics
  - Prevents overfitting assessment

### Step 5: Performance Evaluation
- **Metrics**: Precision, Recall, F1-score, Support
- **Macro & Weighted Averaging**: For multi-class evaluation
- **Overall Accuracy**: Final performance metric

---

## 📈 Dataset Details

### CK+ (Cohn-Kanade+) Dataset

| Property | Details |
|---|---|
| **Dataset Name** | Cohn-Kanade Plus (CK+) |
| **Domain** | Facial Expression Recognition |
| **Total Samples** | 197 test samples |
| **Emotion Classes** | 7 |
| **Class Distribution** | See table below |
| **Image Format** | Grayscale (processed) |
| **Preprocessing** | Grayscale conversion, Histogram equalization |

### Emotion Classes Distribution

| Emotion | Count | Percentage |
|---|---|---|
| 😠 Anger | 27 | 13.7% |
| 😤 Contempt | 11 | 5.6% |
| 😒 Disgust | 35 | 17.8% |
| 😨 Fear | 15 | 7.6% |
| 😊 Happy | 42 | 21.3% |
| 😢 Sadness | 17 | 8.6% |
| 😲 Surprise | 50 | 25.4% |
| **Total** | **197** | **100%** |

---

## ⭐ Results Highlights

### Performance Ranking

🥇 **1st Place: Support Vector Machine (SVM)**
- **Accuracy**: 100%
- **Precision**: 100% (Macro & Weighted)
- **Recall**: 100%
- **F1-Score**: 100%
- **Key Advantage**: Perfect classification across all emotion classes

🥈 **2nd Place: Random Forest**
- **Accuracy**: 97%
- **Precision**: 98% (Macro), 98% (Weighted)
- **Recall**: 97% (Macro), 97% (Weighted)
- **Key Advantage**: Robust and consistent performance with strong generalization

🥉 **3rd Place: K-Nearest Neighbors (KNN)**
- **Accuracy**: 92%
- **Precision**: 93% (Macro), 93% (Weighted)
- **Recall**: 93% (Macro), 92% (Weighted)
- **Key Advantage**: Good balance between interpretability and performance

**4th Place: Decision Tree**
- **Accuracy**: 79%
- **Precision**: 81% (Macro), 79% (Weighted)
- **Recall**: 73% (Macro), 79% (Weighted)
- **Key Advantage**: Highly interpretable but lower accuracy

### Key Findings

✅ **SVM with Linear Kernel** provides the best performance for FER on CK+ dataset  
✅ **Kernel Selection Matters**: Linear kernel outperforms Polynomial and RBF kernels  
✅ **Trade-off Analysis**: Performance decreases with higher train-test ratios  
✅ **Robustness**: Random Forest provides reliable alternative with 97% accuracy  
✅ **Real-world Applicability**: SVM suitable for production FER systems  

---

## 💻 Installation & Usage

### Prerequisites
- Windows 10/11, macOS, or Linux
- Python 3.8 or higher
- pip or conda package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/uzzal2200/Facial-Expression-Recognition.git
cd "Facial emotion with SVM"

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Required Packages

Create a `requirements.txt` file with:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scikit-image>=0.18.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Running the Notebooks

```bash
# Start Jupyter Notebook server
jupyter notebook

# Open desired notebook from the Code/ directory
# Run cells sequentially or all at once

# For SVM model:
# Open Code/svm.ipynb

# For Random Forest model:
# Open Code/Randomforest.ipynb

# For KNN model:
# Open Code/KNN.ipynb

# For Decision Tree model:
# Open Code/decision_tree.ipynb
```

### Quick Start Example

```python
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Load and preprocess image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.equalizeHist(image)

# Extract HOG features
features, _ = hog(image, orientations=9, pixels_per_cell=(4, 4),
                    cells_per_block=(4, 4), visualize=True)

# Train SVM with Grid Search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
svm = GridSearchCV(SVC(), param_grid, cv=5)
svm.fit(X_train, y_train)

# Predict
prediction = svm.predict([features])
print(f"Predicted Emotion: {prediction}")
```

---

## 📝 Citation

Please cite this work if you use it in your research:

### IEEE Format
```
[1] M. Rifat Hossen, M. Uzzal Mia, R. Islam, M. S. Hosain, M. K. Hasan, and T. Shimamura, 
"Facial expression recognition: A machine learning approach with SVM, random forest, KNN, 
and decision tree using grid search method," in Proc. NCSP'25 RISP Int. Workshop Nonlinear 
Circuits, Commun. Signal Process., Pulau Pinang, Malaysia, Feb. 27–Mar. 2, 2025.
```

### BibTeX Format
```bibtex
@article{hossenfacial,
  title={Facial Expression Recognition: A Machine Learning Approach with SVM, Random Forest, 
         KNN, and Decision Tree Using Grid Search Method},
  author={Hossen, Md Rifat and Mia, Md Uzzal and Islam, Rinku and Hosain, Md Sarwar and 
          Hasan, Mohammad Kamrul and Shimamura, Tetsuya},
  conference={NCSP'25 RISP International Workshop on Nonlinear Circuits, Communications and 
              Signal Processing},
  year={2025},
  location={Pulau Pinang, Malaysia},
  month={February 27 -- March 2}
}
```

---

## 📧 Contact

<div align="center">

### 💬 Get in Touch

For questions, collaborations, or research inquiries:

📧 **Email**: [uzzal.220605@s.pust.ac.bd](mailto:uzzal.220605@s.pust.ac.bd)  
🔗 **LinkedIn**: [https://www.linkedin.com/in/md-uzzal-mia-87a3032a1](https://www.linkedin.com/in/md-uzzal-mia-87a3032a1/)  
🐙 **GitHub**: [https://github.com/uzzal2200](https://github.com/uzzal2200)  
🏛️ **Institution**: [Pabna University of Science and Technology](https://pust.ac.bd)

---

### 📱 Social & Academic Networks

[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Profile-4285F4?logo=google-scholar)](https://scholar.google.com/)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Follow-00CCBB?logo=researchgate)](https://www.researchgate.net/)
[![ORCID](https://img.shields.io/badge/ORCID-Connect-green?logo=orcid)](https://orcid.org/0009-0002-4074-2984)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CK+ Dataset**: Thanks to the Cohn-Kanade dataset creators for providing the benchmark dataset
- **scikit-learn Community**: For excellent machine learning tools
- **Tokyo Metropolitan University**: For international collaboration
- **Pabna University of Science and Technology**: For institutional support

---

## 📌 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2025 | Initial publication for NCSP'25 |

---

<div align="center">

### ⭐ If you find this work useful, please consider giving it a star on GitHub!

**Last Updated**: February 2025  
**Status**: ✅ Published and Presented at NCSP'25

</div>
