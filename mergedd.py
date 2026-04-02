import os
import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. Train & Save Model (if not exists)
# -------------------------------
MODEL_FILE = "model.pkl"

if not os.path.exists(MODEL_FILE):
    # Create sample dataset
    data = pd.DataFrame({
        'amount': [10, 5000, 20, 8000],
        'oldbalanceOrg': [100, 5000, 200, 100],
        'merchant': [1, 2, 1, 3],
        'isFraud': [0, 1, 0, 1]
    })

    X = data.drop('isFraud', axis=1)
    y = data['isFraud']

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_FILE)

# -------------------------------
# 2. Load Model
# -------------------------------
model = joblib.load(MODEL_FILE)

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="FinFraud Detector", page_icon="💳")

st.title("🛡️ FinFraud Detector")
st.write("Enter transaction details to check for potential fraud.")

# Input Section
st.subheader("Transaction Information")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    old_balance = st.number_input("Sender Old Balance", min_value=0.0, value=500.0)

with col2:
    merchant_id = st.number_input("Merchant ID (Numeric)", min_value=0, value=1234)
    transaction_type = st.selectbox("Transaction Type", ["Transfer", "Payment", "Cash Out", "Debit"])

# Encode transaction type
type_mapping = {
    "Transfer": 0,
    "Payment": 1,
    "Cash Out": 2,
    "Debit": 3
}
transaction_type_encoded = type_mapping[transaction_type]

# -------------------------------
# 4. Prediction
# -------------------------------
if st.button("Analyze Transaction"):
    
    # Create input dataframe (must match training features)
    features = pd.DataFrame(
        [[amount, old_balance, merchant_id, transaction_type_encoded]],
        columns=['amount', 'oldbalanceOrg', 'merchant', 'type']
    )

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        st.error(f"🚨 High Risk! Fraud Probability: {probability:.2%}")
    else:
        st.success(f"✅ Transaction Safe. Fraud Probability: {probability:.2%}")
