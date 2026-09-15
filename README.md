# Financial Fraud Detection Using Machine Learning

A machine learning-based financial fraud detection system that predicts whether a transaction is **Fraudulent** or **Genuine** using transaction details such as transaction type, amount, and sender/receiver balances.

## Project Overview

Financial fraud detection is a binary classification problem where the goal is to identify fraudulent transactions from a large number of legitimate transactions.

In this project, different machine learning algorithms are trained and compared to detect fraudulent transactions. Since the dataset is highly imbalanced, evaluation is performed using **Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC**, rather than relying only on accuracy.

The trained machine learning pipeline is integrated with a **Streamlit web application** that allows users to enter transaction details and receive a fraud prediction.

## Objectives

- Detect fraudulent financial transactions using Machine Learning.
- Perform data preprocessing and feature engineering.
- Handle the highly imbalanced fraud dataset.
- Compare different classification algorithms.
- Evaluate models using appropriate classification metrics.
- Deploy the trained model using Streamlit.
- Provide real-time Fraud/Not Fraud predictions through a web interface.

## Dataset

The project uses a financial transaction dataset containing information about transaction types, transaction amounts, and sender/receiver account balances.

### Features Used

| Feature | Description |
|---|---|
| `type` | Type of transaction |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender's balance before the transaction |
| `newbalanceOrig` | Sender's balance after the transaction |
| `oldbalanceDest` | Receiver's balance before the transaction |
| `newbalanceDest` | Receiver's balance after the transaction |

Additional balance-difference and transaction-consistency features were also considered during feature engineering.

### Target Variable

The target variable represents whether a transaction is fraudulent:

- `1` → Fraud
- `0` → Not Fraud

## Machine Learning Workflow

```text
Start
   ↓
Collect Transaction Details
   ↓
Preprocess Input Data
   ↓
Encode Categorical Features
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
