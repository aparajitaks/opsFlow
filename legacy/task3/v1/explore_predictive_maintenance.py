import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_dataset():
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

def inspect_dataset(df):
    """
    Performs basic exploratory inspection of the dataset.
    """
    print("=========================================")
    print("1. Dataset Shape (Rows, Columns)")
    print("=========================================")
    print(df.shape)
    print("\n")

    print("=========================================")
    print("2. Column Names")
    print("=========================================")
    print(df.columns.tolist())
    print("\n")

    print("=========================================")
    print("3. Data Types")
    print("=========================================")
    print(df.dtypes)
    print("\n")

    print("=========================================")
    print("4. First 5 Rows")
    print("=========================================")
    print(df.head())
    print("\n")

    print("=========================================")
    print("5. Null Values Count")
    print("=========================================")
    print(df.isnull().sum())
    print("\n")

    print("=========================================")
    print("6. Target Class Distribution ('Machine failure')")
    print("=========================================")
    class_counts = df['Machine failure'].value_counts()
    class_pct = df['Machine failure'].value_counts(normalize=True) * 100
    for idx in class_counts.index:
        status = "Failure" if idx == 1 else "No Failure"
        print(f"{status} (class {idx}): {class_counts[idx]} rows ({class_pct[idx]:.2f}%)")
    print("=========================================")

if __name__ == "__main__":
    # 1. Load the dataset
    try:
        df = load_dataset()
    except Exception as e:
        print(f"Error loading dataset from KaggleHub: {e}")
        exit(1)
        
    # 2. Inspect the dataset structure
    inspect_dataset(df)
