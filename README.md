# Titanic-Survival-Prediction

A simple end-to-end example for training and evaluating a logistic regression model on the Titanic dataset using scikit-learn pipelines. The repository contains a single script `titanic_classification.py` which demonstrates loading data, preprocessing, training, evaluation, and saving a trained pipeline.

## Features

* Loads the Titanic dataset (prefers Seaborn's built-in dataset, falls back to a local Kaggle-style CSV `dataset/train.csv`).
* Preprocessing using `ColumnTransformer` and `Pipeline`:

  * Numeric features: median imputation and standard scaling.
  * Categorical features: most-frequent imputation and one-hot encoding (works across sklearn versions).
* Trains a `LogisticRegression` classifier.
* Evaluates model using accuracy, precision, recall, F1, confusion matrix, classification report and ROC AUC (if available).
* Saves the trained pipeline to `titanic_logreg_pipeline.joblib` for later inference.

---
## Requirements

Create a virtual environment and install the required packages. Example with `pip`:

```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate      # Windows PowerShell
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

> Note: The script contains compatibility logic for `OneHotEncoder` between older and newer scikit-learn versions.

---

## How to run

From the repository root:

```bash
python titanic_classification.py
```

## Quick explanation of the script

* `get_onehot_encoder()`

  * Returns a `OneHotEncoder` instance that is compatible with both older and newer scikit-learn releases (handles `sparse` vs `sparse_output`).

* `load_data()`

  * Tries to load the Titanic dataset from `seaborn.load_dataset('titanic')`. If that fails, it reads `dataset/train.csv` (Kaggle-style CSV).

* `preprocess_and_split(df, random_state=42, test_size=0.2)`

  * Selects a compact set of features (`pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`) when available.
  * Drops rows with missing `survived` values, separates X and y, and performs a stratified train/test split.

* `build_pipeline()`

  * Builds a preprocessing pipeline for numeric and categorical features and appends a `LogisticRegression` classifier.

* `evaluate_model(model, X_test, y_test)`

  * Prints standard classification metrics, confusion matrix, and classification report.
  * If `predict_proba` exists on the model, computes ROC AUC and plots the ROC curve.

* `main()`

  * Orchestrates loading, training, evaluating, and saving the pipeline to disk using `joblib`.

---

