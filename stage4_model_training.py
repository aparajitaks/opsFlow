import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data() -> pd.DataFrame:
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
        "ai4i2020.csv",
    )
    processed_df = df.copy()
    processed_df = processed_df.drop(columns=['UDI', 'Product ID'])
    le = LabelEncoder()
    processed_df['Type'] = le.fit_transform(processed_df['Type'])
    return processed_df

def prepare_splits(df: pd.DataFrame):
    y = df['Machine failure']
    X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\nTraining Model 1: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
    print(classification_report(y_test, lr_preds, target_names=["No Failure", "Failure"]))
    
    print("\nTraining Model 2: Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
    print(classification_report(y_test, rf_preds, target_names=["No Failure", "Failure"]))

if __name__ == "__main__":
    try:
        data = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = prepare_splits(data)
        train_and_evaluate_models(X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"An error occurred: {e}")
