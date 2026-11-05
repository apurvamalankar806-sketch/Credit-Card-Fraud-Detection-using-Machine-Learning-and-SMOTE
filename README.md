# Credit Card Fraud Detection

Detecting fraudulent card transactions in a highly imbalanced dataset. The project compares three models on precision, recall and PR-AUC, and tunes the decision threshold to reduce false alarms.

## Dataset
284,807 transactions from European cardholders, of which 492 (0.17%) are fraud. Features V1 to V28 are anonymised PCA components. The remaining columns are `Time`, `Amount` and the label `Class` (1 = fraud).

The data file is about 150 MB, so it is not included in this repository. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the same folder as the notebook.

## Why Accuracy Is Not Used
Because only 0.17% of transactions are fraud, a model that labels every transaction as legitimate would score about 99.8% accuracy while detecting no fraud. Models are therefore compared on precision, recall and PR-AUC for the fraud class.

## Modelling Approach
1. **Logistic Regression (baseline).** Trained first on the original, imbalanced data. It reached very high accuracy, but almost entirely from classifying normal transactions correctly. At the default threshold it missed 35 of the 98 fraud cases (recall 0.64), which shows a bias towards the majority class.
2. **Logistic Regression with SMOTE.** SMOTE generates synthetic fraud examples in the training set only, so the model sees fraud more often. The validation and test sets are left untouched to reflect real conditions. Recall rose to 0.90, but precision fell to 0.06 because many legitimate transactions were flagged.
3. **Random Forest (class-weighted).** A tree-based model that can capture non-linear patterns. Class weights make each fraud case count more during training, which tests whether the bias can be handled without resampling.
4. **Threshold tuning.** Each model's decision threshold was chosen on a separate validation set (highest precision with recall of at least 0.80), so the models can be compared fairly at a similar level of recall.

## Pipeline
1. Stratified train, validation and test split with a fixed random seed.
2. Feature scaling fitted on the training set only.
3. SMOTE applied to the scaled training set only.
4. Final comparison on the held-out test set of 56,962 transactions, 98 of them fraud.

## Results

At the default threshold (0.5):

| Model | Precision | Recall |
|---|---|---|
| Logistic Regression | 0.83 | 0.64 |
| Logistic Regression + SMOTE | 0.06 | 0.90 |
| Random Forest | 0.96 | 0.75 |

After threshold tuning:

| Model | Precision | Recall | False Alarms | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.61 | 0.85 | 54 | 0.745 |
| Logistic Regression + SMOTE | 0.48 | 0.88 | 92 | 0.724 |
| Random Forest | 0.71 | 0.87 | 35 | 0.851 |

## Key Findings
- SMOTE raised recall at the default threshold, but precision fell to 0.06, which means about 1,400 legitimate transactions were flagged as fraud.
- After threshold tuning, all three models reached similar recall (0.85 to 0.88), and SMOTE gave no clear benefit. Choosing the threshold mattered more than resampling.
- The Random Forest had the highest PR-AUC and the best precision at comparable recall, with 35 false alarms among 56,864 legitimate transactions. It is the selected model.

## Limitations
- The test set contains only 98 fraud cases, so recall differences of a few points correspond to a handful of transactions.
- Results come from a single split and were not cross-validated.
- The threshold depends on the recall target. In practice it should be set from the relative cost of a missed fraud versus a false alarm.
- Features are anonymised, so the model's decisions cannot be explained in business terms.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, imbalanced-learn, Matplotlib, Seaborn

## Run Locally
```bash
pip install -r requirements.txt
jupyter notebook Credit_Card__FraudDetection_Project.ipynb
```

## Repository Contents
- `Credit_Card__FraudDetection_Project.ipynb`: full analysis and modelling
- `requirements.txt`: Python dependencies
