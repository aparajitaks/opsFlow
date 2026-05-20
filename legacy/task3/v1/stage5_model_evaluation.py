import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

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

def train_models(X_train, y_train):
    print("\nTraining models...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return lr_model, rf_model

def evaluate_and_plot(model, X_test, y_test, model_name: str):
    print("\n" + "="*50)
    print(f"EVALUATING MODEL: {model_name}")
    print("="*50)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["No Failure", "Failure"]))
    
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)
    
    auc_score = roc_auc_score(y_test, probs)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Failure", "Failure"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    
    filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot as: {filename}")

if __name__ == "__main__":
    try:
        data = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = prepare_splits(data)
        lr_model, rf_model = train_models(X_train, y_train)
        
        evaluate_and_plot(lr_model, X_test, y_test, "Logistic Regression")
        evaluate_and_plot(rf_model, X_test, y_test, "Random Forest")
    except Exception as e:
        print(f"An error occurred: {e}")
