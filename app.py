import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model and scaler from separate files
@st.cache_resource
def load_model_scaler():
    with open("churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_scaler()

st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("📞 Telecom Customer Churn Prediction")
st.subheader("Enter Customer Information")

# Input form
with st.form("input_form"):
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior = st.selectbox("Senior Citizen", ['No', 'Yes'])
    partner = st.selectbox("Partner", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['No', 'Yes'])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ['No', 'Yes'])
    multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])
    payment_method = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=2000.0)

    submitted = st.form_submit_button("Predict Churn")

# When form is submitted
if submitted:
    input_dict = {
        'gender': [gender],
        'SeniorCitizen': [1 if senior == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
    }

    input_df = pd.DataFrame(input_dict)

    # Encode categorical columns using LabelEncoder (must match training)
    cat_cols = input_df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])  # WARNING: should use same encoder from training!

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.success(f"🧾 Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.info(f"🔢 Churn Probability: {proba:.2%}")
