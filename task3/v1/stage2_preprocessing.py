import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder

def load_dataset() -> pd.DataFrame:
    print("Loading dataset from KaggleHub...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
        "ai4i2020.csv",
    )
    return df

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Starting Stage 2: Feature Engineering & Preprocessing ---")
    processed_df = df.copy()
    
    columns_to_drop = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    print(f"Dropping leaky columns (failure modes): {columns_to_drop}...")
    processed_df = processed_df.drop(columns=columns_to_drop)
    
    print("Encoding categorical column 'Type' using LabelEncoder...")
    le = LabelEncoder()
    processed_df['Type'] = le.fit_transform(processed_df['Type'])
    
    print("\nLabel Encoding Mapping:")
    for idx, category in enumerate(le.classes_):
        print(f"  Category '{category}' -> Encoded as {idx}")
        
    return processed_df

if __name__ == "__main__":
    try:
        raw_df = load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
        
    preprocessed_df = preprocess_features(raw_df)
    
    print("\n" + "="*50)
    print("Preprocessed Dataset Preview:")
    print("="*50)
    print(f"Shape: {preprocessed_df.shape}")
    print("\nColumns remaining after dropping identifiers:")
    print(preprocessed_df.columns.tolist())
    print("\nFirst 5 rows of the preprocessed dataset:")
    print(preprocessed_df.head())
