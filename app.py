import streamlit as st
import pandas as pd
import joblib

# Load model
model, feature_columns = joblib.load("model.pkl")

st.title("📊 Telco Customer Churn Prediction")
st.write("Prediksi apakah pelanggan akan churn atau tidak")

# Input user
tenure = st.number_input("Tenure (bulan)", 0, 72)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

# Buat dataframe input
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

# Dummy encoding manual
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

if contract == "Month-to-month":
    input_data["Contract_Month-to-month"] = 1
elif contract == "One year":
    input_data["Contract_One year"] = 1

if internet == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1
elif internet == "No":
    input_data["InternetService_No"] = 1

if payment == "Electronic check":
    input_data["PaymentMethod_Electronic check"] = 1

input_df = pd.DataFrame([input_data])[feature_columns]

# Prediksi
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ Pelanggan kemungkinan CHURN")
    else:
        st.success("✅ Pelanggan kemungkinan TETAP")

