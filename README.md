# Predictive Maintenance Machine Learning Project

This repository contains Stage 1 through Stage 6 of a Machine Learning workflow for predicting industrial equipment breakdowns using the **AI4I 2020 Predictive Maintenance Dataset**.

---

## 1. Project Folder Structure

A clean, production-ready structure to organize your data, code, models, and results:

```text
opsFlow/
├── task3/
│   ├── v1/                              # Basic working implementation
│   │   ├── stage2_preprocessing.py
│   │   ├── stage4_model_training.py
│   │   ├── stage5_model_evaluation.py
│   │   ├── stage6_overfitting_analysis.py
│   │   └── explore_predictive_maintenance.py
│   ├── v2/                              # Modular split and SMOTE scaling implementation
│   │   ├── src/
│   │   │   ├── config.py
│   │   │   ├── preprocess.py
│   │   │   ├── train.py
│   │   │   └── evaluate.py
│   │   ├── outputs/
│   │   └── main.py
│   ├── v3/                              # Production ready orchestration with Docker & SHAP & MLflow
│   │   ├── src/
│   │   │   ├── config.py
│   │   │   ├── preprocess.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── explainability.py
│   │   ├── outputs/
│   │   │   ├── models/
│   │   │   ├── plots/
│   │   │   └── mlflow/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   └── data/                            # Shared data folder
│       └── ai4i2020.csv
├── README.md                            # Central Documentation (This File)
├── requirements.txt                     # Local environment dependencies
└── venv/                                # Active sandboxed environment
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

---

## 5. Task 3 — Production-Ready v3 Implementation Guide

The **v3** subdirectory is the production-ready version of the Predictive Maintenance pipeline. It adds:
- **MLflow Experiment Tracking:** Local experiment runs logged and viewable in the MLflow UI.
- **Hyperparameter Grid Search:** Optimized C, max_depth, and min_samples_leaf values via 5-Fold Stratified CV.
- **SHAP Explainability:** Local feature impact (Beeswarm) and single-instance prediction driver (Force) plots.
- **Model Serialization & Production Interface:** Saved pickled estimators and scalers loaded through a custom `load_and_predict` dictionary interface.
- **Containerization (Docker):** Standardized runtime environment with custom volume binding.

### Running v3 Locally
```bash
# Navigate to the v3 directory
cd task3/v3

# Activate your virtual environment
source ../../venv/bin/activate

# Execute the orchestrator
python main.py
```

### Viewing MLflow UI Locally
To analyze and compare metrics, runs, parameters, and tags logged by MLflow, run the following command from the `task3/v3/` directory:
```bash
mlflow ui
```
Open your browser and navigate to `http://localhost:5000` to access the MLflow dashboard.

### Building and Running with Docker
Standardize and isolate the environment using Docker:

#### Step 1: Build the Docker Image
```bash
# From within task3/v3/ directory
docker build -t task3-v3 .
```

#### Step 2: Run the Docker Container (with Dataset Volume Mount)
To keep the Docker image small and modular, we exclude the raw dataset from the image and mount it dynamically as a volume at runtime:
```bash
# Using absolute path mount on macOS/Linux
docker run -v $(pwd)/../data:/app/data task3-v3
```
This mounts your local `data/` folder directly to `/app/data/` inside the container, enabling seamless prediction runs without rebuilding the image!

