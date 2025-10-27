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

Example run & results

Below is a real example run (captured from a successful script execution). It includes the printed dataset preview, training confirmation, evaluation metrics and the saved pipeline message.

```text
Dataset sample:
   survived  pclass     sex   age  sibsp  ...  adult_male  deck  embark_town alive  alone
0         0       3    male  22.0      1  ...        True   NaN  Southampton    no  False
1         1       1  female  38.0      1  ...       False     C    Cherbourg   yes  False
2         1       3  female  26.0      0  ...       False   NaN  Southampton   yes   True
3         1       1  female  35.0      1  ...       False     C  Southampton   yes  False
4         0       3    male  35.0      0  ...        True   NaN  Southampton    no   True

[5 rows x 15 columns]
Model trained.
Accuracy: 0.8045
Precision: 0.7931
Recall: 0.6667
F1-score: 0.7244

Classification report:
               precision    recall  f1-score   support

            0       0.81      0.89      0.85       110
            1       0.79      0.67      0.72        69

    accuracy                           0.80       179
   macro avg       0.80      0.78      0.79       179
weighted avg       0.80      0.80      0.80       179

Confusion matrix:
 [[98 12]
 [23 46]]
ROC AUC: 0.8427
Saved pipeline to titanic_logreg_pipeline.joblib
```

**Interpretation**

* The model achieves ~80.45% accuracy on the test fold with a balanced precision/recall tradeoff (precision 0.7931, recall 0.6667, F1 0.7244).
* Confusion matrix shows the model predicts the negative class (did not survive) more accurately than the positive class (survived). There are 23 false negatives (actual survived but predicted not), and 12 false positives.
* ROC AUC of 0.8427 indicates good discrimination ability between classes.

You can replace these numbers with your own run results if you retrain the model.

---

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

