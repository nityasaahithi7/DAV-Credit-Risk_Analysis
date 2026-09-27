import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

def load_data(file_path: str) -> pd.DataFrame:
    """Loads raw dataset from CSV."""
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame):
    """Executes cleaning, imputation, encoding, splitting, and scaling."""
    df = df.copy()

    # Drop duplicate records
    df = df.drop_duplicates()

    # Handle missing numerical values via median imputation
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    # Remove unrealistic domain outliers
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= 60]

    # One-hot encode categorical features
    df_processed = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    # Separate target variable from feature matrix
    X = df_processed.drop("loan_status", axis=1)
    y = df_processed["loan_status"]

    # Stratified train-test split (80/20 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize numeric features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        df_processed
    )