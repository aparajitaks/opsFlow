import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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

def train_and_analyze_overfitting(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("\n" + "="*60)
    print("STAGE 6 — OVERFITTING ANALYSIS")
    print("="*60)
    
    for name, model, X_tr, X_te in [
        ('Logistic Regression', lr_model, X_train_scaled, X_test_scaled),
        ('Random Forest',       rf_model, X_train,        X_test)
    ]:
        train_score = roc_auc_score(y_train, model.predict_proba(X_tr)[:, 1])
        test_score  = roc_auc_score(y_test,  model.predict_proba(X_te)[:, 1])
        gap = train_score - test_score
        print(f"{name}: train={train_score:.4f}  test={test_score:.4f}  gap={gap:.4f}")

if __name__ == "__main__":
    try:
        data = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = prepare_splits(data)
        train_and_analyze_overfitting(X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"An error occurred: {e}")
