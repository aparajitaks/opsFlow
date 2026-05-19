# Predictive Maintenance Machine Learning Project

This repository contains Stage 1 through Stage 6 of a Machine Learning workflow for predicting industrial equipment breakdowns using the **AI4I 2020 Predictive Maintenance Dataset**.

---

## 1. Project Folder Structure

A clean, production-ready structure to organize your data, code, models, and results:

```text
predictive-maintenance-ml/
│
├── data/                            # Local copies of dataset files
├── notebooks/                       # Jupyter Notebooks (.ipynb) for EDA/interactive work
├── models/                          # Saved binary model checkpoints (.pkl / .joblib)
├── outputs/                         # Output artifacts (confusion matrices, ROC curves, CSVs)
│   ├── logistic_regression_confusion_matrix.png
│   └── random_forest_confusion_matrix.png
│
├── explore_predictive_maintenance.py # Stage 1: Exploratory Data Analysis (EDA)
├── stage2_preprocessing.py          # Stage 2: Feature Engineering & Cleaning
├── stage4_model_training.py         # Stage 4: Basic Training (LR and RF)
├── stage5_model_evaluation.py       # Stage 5: Detailed Model Metrics & Plots
├── stage6_overfitting_analysis.py   # Stage 6: Train-Test Gap Analysis
│
├── .gitignore                       # System files and local env exclude list
├── requirements.txt                 # Project library dependencies
└── README.md                        # Documentation and Local Run Guide (This File)
```

---

## 2. Local Setup and Installation Guide

Follow these steps to initialize a clean Python environment on your local machine.

### Step 1: Create a Virtual Environment
Avoid conflicts with other Python projects by creating a sandboxed virtual environment:

```bash
# Navigate to project root
cd /Users/galaxy_grid/Desktop/opsFlow

# Create a virtual environment named 'venv'
python3 -m venv venv
```

### Step 2: Activate the Virtual Environment
Activate the environment before executing any commands or scripts:
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
* **Windows:**
  ```cmd
  venv\Scripts\activate
  ```

### Step 3: Install Required Dependencies
With the virtual environment active, install all packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 3. Running and Verifying Scripts

Run each Stage's python file sequentially to verify correct execution:

1. **Stage 1 (EDA):**
   ```bash
   python explore_predictive_maintenance.py
   ```
2. **Stage 2 (Preprocessing):**
   ```bash
   python stage2_preprocessing.py
   ```
3. **Stage 4 (Training):**
   ```bash
   python stage4_model_training.py
   ```
4. **Stage 5 (Evaluation):**
   ```bash
   python stage5_model_evaluation.py
   ```
5. **Stage 6 (Overfitting):**
   ```bash
   python stage6_overfitting_analysis.py
   ```

---

## 4. Common Troubleshooting and Fixes

### Issue 1: `ModuleNotFoundError: No module named 'pandas' / 'sklearn'`
* **Cause:** Your terminal is not running inside the virtual environment where libraries were installed.
* **Fix:** Verify your shell prompt has `(venv)` prefix. If not, run `source venv/bin/activate` or `pip install -r requirements.txt` again.

### Issue 2: Outdated Kaggle API credentials
* **Cause:** `kagglehub` requires internet access to download the dataset.
* **Fix:** Ensure your internet connection is active. No manual Kaggle API tokens are needed for public datasets pulled via `kagglehub.dataset_load()`.
