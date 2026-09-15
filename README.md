# Fraud Transaction Detection Using Machine Learning & Gradient Boosting

A machine learning-based system for detecting fraudulent financial transactions using transaction details such as transaction type, amount, and sender/receiver account balances. The project compares multiple supervised learning algorithms and provides a Streamlit-based interface for real-time fraud prediction.

## Project Overview

Financial fraud detection is challenging because fraudulent transactions represent only a small fraction of the total transactions. This project addresses the class-imbalance problem by applying appropriate preprocessing, stratified train-test splitting, and imbalance-aware model evaluation.

The project implements and compares four classification models:

- Logistic Regression
- LinearSVC
- LightGBM
- XGBoost

The models are evaluated using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC. Based on the reported experimental results, XGBoost provides the best overall balance between precision and recall and achieves the highest F1-score.

## Features

The system uses the following transaction features:

- **Transaction Type** — PAYMENT, TRANSFER, CASH_OUT, CASH_IN, and DEBIT
- **Amount** — Monetary value of the transaction
- **Old Balance (Sender)** — Sender's balance before the transaction
- **New Balance (Sender)** — Sender's balance after the transaction
- **Old Balance (Receiver)** — Receiver's balance before the transaction
- **New Balance (Receiver)** — Receiver's balance after the transaction

Additional balance-difference and transaction-consistency indicators were also considered during the feature engineering process.

## Machine Learning Workflow

```text
Start
  ↓
Collect Transaction Details
  ↓
Preprocess Input Data
  ↓
Use Trained ML Model
  ↓
Model Analyzes Transaction Patterns
  ↓
Generate Prediction
  ↓
Is Prediction = 1?
  ├── Yes → Fraudulent Transaction
  └── No  → Genuine Transaction
                ↓
       Display Result to User
                ↓
               End
```

## Data Preprocessing

The preprocessing pipeline includes:

1. Selecting relevant transaction features.
2. Separating input features (`X`) and target variable (`y`).
3. Converting the categorical transaction type into numerical form using **one-hot encoding**.
4. Applying a **stratified train-test split** to preserve the fraud/non-fraud class distribution.
5. Integrating preprocessing and model prediction into a machine learning pipeline.
6. Handling the highly imbalanced nature of the fraud dataset during model training and evaluation.

Using a pipeline helps maintain consistent preprocessing during training, testing, and real-time prediction.

## Models Implemented

### 1. Logistic Regression

Used as a baseline binary classification model.

**Reported performance:**

- Accuracy: 0.95
- Precision: 0.02
- Recall: 0.94
- F1-score: 0.04

The model detects most fraudulent transactions but produces a large number of false positives.

### 2. LinearSVC

A linear Support Vector Classifier used to establish a linear decision boundary between the two classes.

**Reported performance:**

- Accuracy: 1.00
- Precision: 0.72
- Recall: 0.50
- F1-score: 0.59

It provides a better balance than Logistic Regression but misses a significant number of fraudulent transactions.

### 3. LightGBM

A gradient boosting framework that builds decision trees sequentially and uses leaf-wise tree growth.

**Reported performance:**

- Accuracy: 0.97
- Precision: 0.04
- Recall: 1.00
- F1-score: 0.07

LightGBM detects the fraudulent class very effectively, but its low precision results in many false-positive predictions.

### 4. XGBoost

An ensemble gradient boosting algorithm capable of learning complex, non-linear relationships between transaction features.

**Reported performance:**

- Accuracy: 1.00
- Precision: 0.95
- Recall: 0.76
- F1-score: 0.85

XGBoost achieved the highest F1-score and the strongest overall balance between precision and recall among the evaluated models.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.95 | 0.02 | 0.94 | 0.04 |
| LinearSVC | 1.00 | 0.72 | 0.50 | 0.59 |
| LightGBM | 0.97 | 0.04 | 1.00 | 0.07 |
| **XGBoost** | **1.00** | **0.95** | **0.76** | **0.85** |

> **Best reported model: XGBoost**, based on the highest F1-score and its stronger precision-recall balance.

## Why F1-Score Matters

The dataset is highly imbalanced, meaning legitimate transactions greatly outnumber fraudulent transactions. Therefore, accuracy alone is not sufficient to judge the model.

- **Precision** measures how many transactions predicted as fraud are actually fraudulent.
- **Recall** measures how many actual fraudulent transactions are successfully detected.
- **F1-score** provides a balance between precision and recall.

For this project, F1-score is particularly useful because both false positives and false negatives are important in fraud detection.

## Deployment

The trained machine learning pipeline is integrated into a **Streamlit** web application.

### Application Flow

1. The user selects the transaction type.
2. The user enters the transaction amount.
3. The user enters sender and receiver balance details.
4. The application preprocesses the input in the same format used during training.
5. The trained model generates a prediction.
6. The result is displayed as:
   - **Fraud**
   - **Not Fraud**

The trained model is saved using **Joblib**, allowing the application to load the model without retraining it.

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **LightGBM**
- **XGBoost**
- **Matplotlib**
- **Seaborn**
- **Joblib**
- **Streamlit**

## Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- ROC-AUC

Confusion matrices help analyze true positives, true negatives, false positives, and false negatives, while ROC curves and AUC evaluate the model's discriminative ability across different classification thresholds.

## Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Financial-Fraud-Detection-Using-ML
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the required packages:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn joblib streamlit
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser and allow you to enter transaction details for prediction.

## Project Structure

```text
Financial-Fraud-Detection-Using-ML/
│
├── app.py
├── requirements.txt
├── fraud_detection_pipeline.pkl
├── notebook/
│   └── fraud_detection.ipynb
├── dataset/
│   └── AIML Dataset.csv
└── README.md
```

> File names and folders may differ depending on the final repository structure.

## Key Challenges Addressed

### Class Imbalance

Fraudulent transactions form only a small portion of the dataset. The project therefore uses stratified splitting and imbalance-aware model evaluation.

### False Positives

A model that flags too many genuine transactions as fraud is not practically useful. Precision and F1-score are therefore considered along with recall.

### Complex Transaction Patterns

Fraudulent behavior can involve non-linear relationships between transaction amount, balance changes, and transaction type. Tree-based boosting models such as XGBoost can capture these interactions effectively.

### Consistent Deployment

The preprocessing and classification steps are combined into a pipeline so that the same transformations are applied during training and prediction.

## Future Improvements

Possible future improvements include:

- Advanced feature engineering
- Cost-sensitive learning
- Handling concept drift in evolving transaction patterns
- Real-time transaction data pipelines
- Scalable cloud deployment
- Further model and hyperparameter optimization

## Conclusion

This project demonstrates the application of supervised machine learning to financial fraud detection. Four classification models were implemented and compared on a highly imbalanced transaction dataset. Among the reported results, XGBoost achieved the strongest overall performance with an F1-score of **0.85**, providing a better balance between precision and recall.

The trained model is integrated into a Streamlit application, enabling users to enter transaction information and receive an immediate fraud/not-fraud prediction.
