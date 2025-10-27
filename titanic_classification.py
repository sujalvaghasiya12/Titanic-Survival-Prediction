# titanic_classification.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import warnings

warnings.filterwarnings("ignore")

def get_onehot_encoder():
    """Return an OneHotEncoder instance that works across sklearn versions."""
    try:
        # sklearn >= 1.4
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        # older sklearn versions
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def load_data():
    # prefer seaborn built-in dataset (no need to download)
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
    except Exception:
        df = pd.read_csv('dataset/train.csv')  # fallback if you downloaded Kaggle csv
    return df

def preprocess_and_split(df, random_state=42, test_size=0.2):
    df = df.copy()
    # keep commonly used features (columns as in seaborn titanic)
    columns = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    df = df[[c for c in columns if c in df.columns]]
    df = df[df['survived'].notna()]
    X = df.drop('survived', axis=1)
    y = df['survived'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def build_pipeline():
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', get_onehot_encoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, [c for c in numeric_features]),
        ('cat', categorical_transformer, [c for c in categorical_features])
    ], remainder='drop')

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    return pipeline

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:", round(recall_score(y_test, y_pred), 4))
    print("F1-score:", round(f1_score(y_test, y_pred), 4))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print("ROC AUC:", round(auc, 4))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1], '--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

def main():
    df = load_data()
    print("Dataset sample:\n", df.head())
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("Model trained.")

    evaluate_model(pipeline, X_test, y_test)

    # Save pipeline for later use
    joblib.dump(pipeline, 'titanic_logreg_pipeline.joblib')
    print("Saved pipeline to titanic_logreg_pipeline.joblib")

if __name__ == "__main__":
    main()
