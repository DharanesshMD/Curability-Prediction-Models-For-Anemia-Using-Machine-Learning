# Curability Prediction Models For Anemia Using Machine Learning

This project demonstrates the use of machine learning to predict the curability (presence) of anemia based on synthetic patient data. It includes scripts for data generation, model training and selection, and a simple GUI application for dataset-wide prediction summaries.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Generate Synthetic Data](#1-generate-synthetic-data)
  - [2. Train and Select the Best Model](#2-train-and-select-the-best-model)
  - [3. Run the Prediction App](#3-run-the-prediction-app)
- [Dataset Description](#dataset-description)
- [Model Details](#model-details)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

This repository provides a full pipeline for predicting anemia using synthetic medical data. It covers data generation, preprocessing, model selection (Logistic Regression, Random Forest, Gradient Boosting), and a Tkinter-based GUI for summarizing predictions on the dataset.

## Features

- **Synthetic Data Generation**: Easily create a dataset with realistic features and missing values.
- **Automated Model Training**: Compares multiple classifiers and selects the best based on accuracy.
- **Preprocessing Pipeline**: Handles encoding, imputation, scaling, and feature selection.
- **GUI Prediction App**: Visualizes dataset-wide anemia predictions and probabilities.

## Project Structure

```
.
├── anemia_best_model.joblib         # Trained model and preprocessors (generated after training)
├── data_generation.py               # Script to generate synthetic data
├── model_training.py                # Script to train and select the best model
├── predict_app.py                   # Tkinter GUI for dataset prediction summary
├── requirements.txt                 # Python dependencies
├── synthetic_anemia_data.csv        # Generated synthetic dataset
```

## Installation

1. **Clone the repository**  
   ```bash
   git clone <repo-url>
   cd Curability-Prediction-Models-For-Anemia-Using-Machine-Learning
   ```

2. **Install dependencies**  
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Synthetic Data

To create a new synthetic dataset:
```bash
python data_generation.py
```
This will generate `synthetic_anemia_data.csv` in the project directory.

### 2. Train and Select the Best Model

To train models and save the best one:
```bash
python model_training.py
```
This will output the best model's accuracy and save the model and preprocessors to `anemia_best_model.joblib`.

### 3. Run the Prediction App

To launch the GUI and view dataset-wide predictions:
```bash
python predict_app.py
```
A window will appear. Click "Show Results" to see a summary of anemia predictions for the dataset.

## Dataset Description

The synthetic dataset contains the following columns:

| Feature         | Description                                 |
|-----------------|---------------------------------------------|
| Age             | Patient age (years)                         |
| Gender          | Male or Female                              |
| Hemoglobin      | Hemoglobin level (g/dL)                     |
| RBC_Count       | Red blood cell count (million cells/mcL)    |
| MCV             | Mean corpuscular volume (fL)                |
| WBC_Count       | White blood cell count (thousand cells/mcL) |
| Platelet_Count  | Platelet count (thousand/mcL)               |
| Anemia          | Target: 1 = Anemia, 0 = No Anemia           |

Some values are missing to simulate real-world data.

## Model Details

- **Preprocessing**: Label encoding for gender, mean imputation for missing values, standard scaling, and selection of the top 5 features.
- **Models Compared**: Logistic Regression, Random Forest, Gradient Boosting.
- **Selection**: The model with the highest accuracy on the test set is saved.
- **Artifacts**: The best model and all preprocessing steps are saved in `anemia_best_model.joblib`.

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- joblib
- matplotlib
- tkinter (usually included with Python)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

---

**Note:**  
- The model and data are synthetic and for demonstration only.
- For any issues or questions, please open an issue or contact the maintainer.

--- 