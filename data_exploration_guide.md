# Stage 1: Data Exploration & Understanding

This guide covers the initial inspection and exploratory data analysis (EDA) of the **AI4I 2020 Predictive Maintenance Dataset** for a machine learning workflow.

---

## 1. Complete Exploratory Python Code

Below is a clean, modular Python script tailored for a Jupyter Notebook or a script environment. It loads the dataset directly from KaggleHub, performs basic sanity checks, and outputs crucial statistics.

```python
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_dataset() -> pd.DataFrame:
    """
    Loads the AI4I 2020 Predictive Maintenance dataset directly from KaggleHub.
    """
    print("Loading dataset from KaggleHub...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
        "ai4i2020.csv",
    )
    return df

def inspect_dataset(df: pd.DataFrame):
    """
    Prints a basic overview of the dataset: shape, column names, 
    data types, head, missing values, and target distribution.
    """
    print("\n--- 1. Dataset Shape ---")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    
    print("\n--- 2. Column Names ---")
    print(df.columns.tolist())
    
    print("\n--- 3. Data Types ---")
    print(df.dtypes)
    
    print("\n--- 4. First 5 Rows ---")
    print(df.head())
    
    print("\n--- 5. Null (Missing) Values ---")
    print(df.isnull().sum())
    
    print("\n--- 6. Target Distribution ('Machine failure') ---")
    counts = df['Machine failure'].value_counts()
    percentage = df['Machine failure'].value_counts(normalize=True) * 100
    for val in counts.index:
        label = "Failure" if val == 1 else "No Failure"
        print(f"  {label} (Class {val}): {counts[val]} rows ({percentage[val]:.2f}%)")

if __name__ == "__main__":
    # Load and inspect
    try:
        df = load_dataset()
        inspect_dataset(df)
    except Exception as e:
        print(f"Error loading dataset: {e}")
```

---

## 2. Explanation of Key Feature Selections

### Why UDI and Product ID are not useful for ML
* **UDI (Unique Identifier):** This is simply an index or row number ranging from 1 to 10,000. It contains no physical information about the machinery or its operating environment. Including it in training might cause the model to overfit on row index numbers.
* **Product ID:** This is a combination of product type (L, M, or H) and a serial number (e.g., `L47181`). While the product type prefix (L/M/H) is valuable, the numeric part is a unique serial number that does not possess predictive value for failure patterns.
* **Best Practice:** Drop `UDI` and `Product ID` before training. Use `Type` (the category prefix) if encoded properly.

### Why "Machine failure" is the target column
* **The Goal:** The primary objective of a binary predictive maintenance classifier is to predict whether a machine is going to fail (`1`) or not fail (`0`). 
* **The Column:** The `Machine failure` column maps exactly to this objective. It acts as our **ground truth label** (the dependent variable $y$), signaling whether a failure event occurred during the specific process run.

### Why TWF, HDF, PWF, OSF, and RNF should be ignored for v1
* **Specific Failure Modes:** These columns represent specific failure modes (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, and Random Failure). 
* **Target Leakage / Circular Logic:** They are sub-causes of the primary failure. If any of these are `1`, then `Machine failure` is almost always guaranteed to be `1`. Passing them as features would make the problem trivial and lead to **target leakage**, as they are not available prior to predicting whether a machine will fail or not.
* **v1 Focus:** For the initial version (v1), we aim to predict *if* a failure will occur, not *why* it occurred. Multi-class or multi-label classification can be introduced later in Stage 2.

---

## 3. Detecting and Understanding Class Imbalance

Executing `df['Machine failure'].value_counts()` returns:
* **No Failure (Class 0):** 9,661 (96.61%)
* **Failure (Class 1):** 339 (3.39%)

### What is Class Imbalance in Predictive Maintenance?
In predictive maintenance, **class imbalance** refers to a scenario where one class (normal operation) significantly outnumbers the other class (machine failure). In this dataset, 96.61% of the observations are non-failures, and only 3.39% are failure events.

### Why Class Imbalance Matters
1. **Misleading Accuracy:** A naive model that simply predicts "No Failure" (0) for every single instance would achieve an accuracy of **96.61%**, while being entirely useless because it fails to capture a single actual breakdown.
2. **Detection Failure:** Machine learning algorithms minimize overall error. Because failures are so rare, the model may treat them as noise or ignore them to optimize overall loss, leading to high false-negative rates.
3. **Business Cost:** In industrial settings, a false negative (failing to predict a machine breakdown) is highly expensive, resulting in unplanned downtime, repair costs, and safety hazards.
