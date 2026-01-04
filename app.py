import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model, feature_names = joblib.load("model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("📊 Customer Churn Prediction App")
st.write("Aplikasi prediksi apakah pelanggan akan **Churn** atau **Tidak Churn** menggunakan **Decision Tree**.")

st.divider()

# =========================
# INPUT FORM
# =========================
st.subheader("Masukkan Data Pelanggan")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
credit_card = st.selectbox("Has Credit Card?", [0, 1])
active_member = st.selectbox("Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# =========================
# ENCODING MANUAL (HARUS SAMA DENGAN TRAINING)
# =========================
country_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Female": 0, "Male": 1}

input_data = pd.DataFrame([[
    credit_score,
    country_map[country],
    gender_map[gender],
    age,
    tenure,
    balance,
    products_number,
    credit_card,
    active_member,
    estimated_salary
]], columns=feature_names)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Pelanggan **BERPOTENSI CHURN**")
    else:
        st.success("✅ Pelanggan **TIDAK CHURN**")

    st.write("### Probabilitas")
    st.write(f"Tidak Churn: **{probability[0]*100:.2f}%**")
    st.write(f"Churn: **{probability[1]*100:.2f}%**")

st.divider()
st.caption("Model: Decision Tree Classifier | Dataset: Bank Customer Churn")
