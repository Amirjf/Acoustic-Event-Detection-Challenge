# Acoustic Event Detection

### Main Assignment for Applied Machine Learning

This repository contains our **Acoustic Event Detection Challenge** project, part of an Applied Machine Learning course. The central theme is analyzing audio data from the **ESC-50** dataset, extracting meaningful features, and applying machine learning algorithms (both supervised and unsupervised) to classify environmental sounds effectively.

---

## Table of Contents

1. [Overview & Assignment Description](#overview--assignment-description)
2. [Main Assignment Steps](#main-assignment-steps)
   - [1. Understanding Your Data](#1-understanding-your-data)
   - [2. Extracting Features (Individually Graded)](#2-extracting-features-individually-graded)
   - [3. Modeling Your Data (Algorithm Development)](#3-modeling-your-data-algorithm-development)
   - [4. Comparing Your Models](#4-comparing-your-models)
3. [Repository Structure](#repository-structure)
4. [Installation & Requirements](#installation--requirements)
5. [Feature Extraction Contributions](#feature-extraction-contributions)
   - [Amir's Feature Extraction](#amirs-feature-extraction)
   - [Mina's Feature Extraction](#minas-feature-extraction)
   - [Soheil's Feature Extraction](#soheils-feature-extraction)
6. [PCA & ICA Notebooks](#pca--ica-notebooks)
   - [Amir's Notebook](#amirs-notebook)
   - [Mina's Notebook](#minas-notebook)
   - [Soheil's Notebook](#soheils-notebook)
7. [Team & Contributions](#team--contributions)

---

## Overview & Assignment Description

### The Acoustic Event Detection Challenge

In many nursing homes, acoustic monitoring is used to detect potential alarms or anomalies. However, simple detection methods based on **sound intensity** and **duration** often result in high false-alarm rates, while some short-lived but important events may be missed.

**Goal**:

- Develop machine learning solutions that **classify** or **detect** specific acoustic events.
- Use the **ESC-50** dataset (environmental sound recordings) to explore data from **10 classes** chosen from **5 different categories**.

The **main research question** guiding this project is:

> **Can we infer the source of a sound recording from extracted features?**

**Subquestions** include:

1. How can we define a suitable set of features?
2. Which algorithms can be used to solve this task?
3. How does performance depend on algorithm choice and their parameters?
4. How does performance depend on feature extraction parameters?
5. How well do supervised algorithms perform against unsupervised algorithms for the task?

---

## Main Assignment Steps

Below is a concise outline of the tasks:

### 1. Understanding Your Data

1. **Read in only the .wav files you want**
   - From **10 classes** (2 classes from each of the 5 categories in ESC-50).
2. **Visualize an instance of each class**
   - For example, plot a **spectrogram** or **spectrum** to get an overview of the data.

- **File**: [Data Visualizations](Data_Visualization.ipynb)

### 2. Extracting Features (Individually Graded)

1. **Extract lower-dimensional features** (e.g., binned histograms from banded spectrograms, MFCCs, etc.).
2. **Create variations of these features** (e.g., different distributions of bin edges, different bandings for spectrograms).
3. **Save out results to NPZ files** so re-generation of features is not required for each run.

### 3. Modeling Your Data (Algorithm Development)

1. **Split your features** into **training** and **test** sets (80:20).
2. **Further split** the **training data** into **train and validation** sets (folds).
3. **Fit your model** on the train data and vary model parameters:
   - **SVM**: Kernel type, C, Gamma
   - **K-means**: Number of clusters
4. **Use the validation data** to assess parameter performance (using ROC, AUC).
5. **Select** a subset of models based on performance (e.g., linear SVM with feature_set1, RBF SVM with feature_set2, K-means with 10 clusters, etc.).

### 4. Comparing Your Models

1. **Use the test set** to calculate metrics (Accuracy, TPR, FPR, etc.).
2. **Plot** these on a coverage plot (or any chosen metric visualization).
3. **Write a commentary** discussing performance (possibly include confusion matrices).

---

## Repository Structure

```plaintext
📂 AED Challenge
│   └── Anomaly_Detection_Assignment-Mina.md

📂 Anomaly Detection
│   └── .gitkeep

📂 ESC-50-master
│   └── ... (audio dataset files and structure)

📂 features
│   ├── .gitkeep
│   ├── extracted_features_multiple_test.npz
│   ├── extracted_features_test.npz
│   └── extracted_features.npz

📂 PCA_ICA
│   ├── __pycache__/
│   ├── dataset/
│   │   ├── .gitkeep
│   │   ├── DejaVuSerif.ttf
│   │   └── fontdemo.py
│   ├── pca_ica.ipynb             (Amir notebook)
│   ├── PCA-ICA_Mina.ipynb        (Mina's notebook)
│   ├── PCA-ICA_Soheil.ipynb      (Soheil's notebook)
│   ├── .gitignore
│   ├── data_utils.py
│   ├── feature_utils.py
│   ├── Main Assignment Steps 2024.md
│   ├── main.ipynb
│   ├── plot_utils.py
│   └── README.md                 (You are here)
```

- **AED Challenge**: Contains anomaly detection assignment files.
- **Anomaly Detection**:
- **ESC-50-master**: Dataset folder with .wav files for 50 classes. We focus on 10 classes.
- **features**: Stores extracted feature files (`.npz`).
- **PCA_ICA**:
  - **Notebooks** for PCA/ICA workflows.
  - **Utility scripts** for data loading, feature engineering, and plotting.

---

## Installation & Requirements

1. **Clone** the repository:
   ```bash
   git clone https://github.com/HanzeUAS-MasterSSE/AML-Soheil-Mina-Amir.git
   ```
2. **Install dependencies** (in a virtual environment recommended):

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, manually install packages mentioned in the notebooks.

3. Ensure you have the correct **ESC-50** .wav files in the **ESC-50-master** folder or link them properly.

---

## Feature Extraction Contributions

Each team member (Amir, Mina, Soheil) explored feature extraction differently. Below is a summary;

### Amir's Feature Extraction

[Your Explanation Here]

### Mina's Feature Extraction

[Your Explanation Here]

### Soheil's Feature Extraction

[Your Explanation Here]

---

## PCA & ICA Notebooks

Detailed dimensionality-reduction approaches and the final extracted components can be found in each member’s notebook:

##### Refer to the following each member's notebook for detailed PCA/ICA analysis:

### Amir's Notebook

- **File**: [PCA-ICA_Amir.ipynb](PCA_ICA/pca_ica.ipynb)
- Words Founded: - `Fink` - ``

### Mina's Notebook

- **File**: [PCA-ICA_Mina.ipynb](PCA_ICA/PCA-ICA_Mina.ipynb)
- Words Founded: - `word here` - `word here`

### Soheil's Notebook

- **File**: [PCA-ICA_Soheil.ipynb](PCA_ICA/PCA-ICA_Soheil.ipynb)
- Words Founded: - `word here` - `word here`

---

## Contributions

- **Mina**
- **Soheil**
- **Amir**

Each member contributed to feature extraction, PCA/ICA analysis, and the overall architecture of the project.

---

<p align="center"> Crafted by Mina, Soheil & Amir </p>
