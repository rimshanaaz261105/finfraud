import streamlit as st
import joblib
import pandas as pd
import os

# 1. Load model safely using absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(model_path):
    st.error("❌ model.pkl not found! Please place it in the same folder as this script.")
    st.stop()

model = joblib.load(model_path)

# 2. UI Setup
st.set_page_config(page_title="FinFraud Detector", page_icon="💳")
st.title("🛡️ FinFraud Detector")
st.write("Enter transaction details to check for potential fraud.")

# 3. User Inputs
st.subheader("Transaction Information")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    old_balance = st.number_input("Sender Old Balance", min_value=0.0, value=500.0)

with col2:
    merchant_id = st.number_input("Merchant ID (Numeric)", min_value=0, value=1234)
    transaction_type = st.selectbox("Type", ["Transfer", "Payment", "Cash Out", "Debit"])

# 4. Encode transaction type (simple manual encoding)
type_mapping = {
    "Transfer": 0,
    "Payment": 1,
    "Cash Out": 2,
    "Debit": 3
}
transaction_type_encoded = type_mapping[transaction_type]

# 5. Prediction
if st.button("Analyze Transaction"):
    try:
        # IMPORTANT: Must match training features
        features = pd.DataFrame([[
            amount,
            old_balance,
            merchant_id,
            transaction_type_encoded
        ]], columns=[
            'amount',
            'oldbalanceOrg',
            'merchant',
            'type'
        ])

        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        if prediction[0] == 1:
            st.error(f"🚨 High Risk! Fraud Probability: {probability:.2%}")
        else:
            st.success(f"✅ Transaction Safe. Fraud Probability: {probability:.2%}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
