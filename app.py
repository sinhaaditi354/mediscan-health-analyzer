# app.py

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Page configuration
st.set_page_config(page_title="MediScan AI – Diabetes Prediction", page_icon="🩺")

# Load the trained model
try:
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("🚫 Model file not found! Make sure 'diabetes_model.pkl' is in the same folder.")
    st.stop()

# App Title and Description
st.title("🩺 MediScan AI – Diabetes Prediction App")
st.markdown("""
Welcome to **MediScan AI** – your personal health assistant!

### 🌟 Features:
- 🧪 Predict diabetes risk from health metrics
- 📊 Visual health analysis
- 💡 Instant feedback & suggestions

---

👉 **Enter your health details below to get started:**
""")

# Input fields
st.header("🔍 Enter Your Health Information")

preg = st.number_input("Number of Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, key="glucose")
bp = st.number_input("Blood Pressure (mm Hg)", min_value=0)
skin = st.number_input("Skin Thickness (mm)", min_value=0)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

# Optional: Additional health metrics
st.markdown("### 🧍‍♂️ Additional Health Parameters (optional)")
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=60, max_value=140, value=80)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=72)
sugar = glucose  # For clarity

# Predict button
if st.button("🔮 Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(input_data)

    st.header("🧾 Prediction Result")
    if result[0] == 1:
        st.error("⚠️ High risk of Diabetes detected. Please consult a doctor.")
    else:
        st.success("✅ Low risk of Diabetes. You seem to be healthy!")

    # Prediction confidence
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
    else:
        probabilities = [1 - result[0], result[0]]

    st.subheader("📊 Prediction Confidence")
    labels = ['Low Risk', 'High Risk']
    fig, ax = plt.subplots()
    ax.bar(labels, probabilities, color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Diabetes Risk Confidence")
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)

    # --------------------------
    # 📊 Bar Chart – Health Metrics
    st.subheader("📋 Health Parameters Overview")
    param_labels = ['Age', 'Height(cm)', 'Weight(kg)', 'Systolic BP', 'Diastolic BP', 'Heart Rate', 'Sugar']
    param_values = [age, height, weight, systolic, diastolic, heart_rate, sugar]

    fig2, ax2 = plt.subplots()
    ax2.bar(param_labels, param_values, color='skyblue')
    plt.xticks(rotation=45)
    ax2.set_ylabel('Values')
    ax2.set_title('Your Body Vitals')
    st.pyplot(fig2)

    # --------------------------
    # 🧮 BMI Health Category
    bmi_status = ""
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_status = "Normal"
    elif 25 <= bmi < 29.9:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"

    # Sugar Status
    if sugar < 90:
        sugar_status = "Low"
    elif 90 <= sugar <= 140:
        sugar_status = "Normal"
    else:
        sugar_status = "High"

    # Risk Summary
    if bmi_status == "Obese" or sugar_status == "High":
        risk_level = "⚠️ High Risk"
    elif bmi_status == "Overweight" or sugar_status == "Low":
        risk_level = "⚠️ Moderate Risk"
    else:
        risk_level = "✅ Low Risk"

    st.subheader("🧠 Risk Interpretation")
    st.write(f"**BMI Category:** {bmi_status}")
    st.write(f"**Sugar Status:** {sugar_status}")
    st.markdown(f"### 🚨 **Overall Risk Summary:** {risk_level}")

    # --------------------------
    # 📈 BMI Trend (Simulated)
    bmi_history = [22.5, 23.0, 24.2, 25.6, 27.1, bmi]
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(bmi_history))
    df_bmi = pd.DataFrame({'Date': dates, 'BMI': bmi_history})
    st.subheader("📈 BMI Trend Over Time")
    st.line_chart(df_bmi.set_index("Date"))

    # --------------------------
    # 💯 Health Score
    st.subheader("💯 Your Health Score")

    health_score = 100
    if bmi < 18.5 or bmi > 24.9:
        health_score -= 20
    else:
        health_score -= 5

    if sugar < 70 or sugar > 140:
        health_score -= 20
    else:
        health_score -= 5

    health_score = max(0, min(health_score, 100))
    st.metric(label="🧬 Health Score", value=f"{health_score}/100")

    st.markdown("### 🧠 Interpretation:")
    if health_score >= 90:
        st.success("Excellent! Your health is in top shape.")
    elif health_score >= 70:
        st.info("Good! Maintain your habits and stay active.")
    elif health_score >= 50:
        st.warning("Moderate. Consider improving your diet and activity.")
    else:
        st.error("Risky health level. Please consult a healthcare provider.")
