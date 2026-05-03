import streamlit as st
import pandas as pd
import joblib

saved = joblib.load("xgboost_fraud_pipeline.pkl")

model = saved["model"]
threshold = saved["threshold"]

st.title("Fraud Detection Prediction app")

st.markdown("Please enter the transaction details and use the predict button")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])

amount = st.number_input("Amount", min_value = 0.0, value = 1000.0)

oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value = 0.0, value = 10000.0)

newbalanceOrig = st.number_input("New Balance (Sender)", min_value = 0.0, value = 9000.0)

oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value = 0.0, value = 0.0)

newbalanceDest = st.number_input("New Balance (Reciever)", min_value = 0.0, value = 0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    input_data["balanceDiffOrig"] = input_data["oldbalanceOrg"] - input_data["newbalanceOrig"]
    input_data["balanceDiffDest"] = input_data["newbalanceDest"] - input_data["oldbalanceDest"]

    prob = model.predict_proba(input_data)[:, 1][0]
    prediction = int(prob > threshold)

    prediction = "Possible Fraud" if (prob > threshold) else "Fraud Unlikely"

    st.subheader(f"Prediction: {prediction}")

    if prob > 0.7:
        st.error(f"HIGH RISK 🚨 ({prob:.2f})")
    elif prob > 0.3:
        st.warning(f"MEDIUM RISK ⚠️ ({prob:.2f})")
    else:
        st.success(f"LOW RISK ✅ ({prob:.2f})")