import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# -------------------------------
# Load model & encoder
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, 'rf_model.pkl'), 'rb'))
le = pickle.load(open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb'))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Bankruptcy Predictor", layout="wide")

# -------------------------------
# Custom CSS (Modern UI 🔥)
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 style='text-align: center;'>🏦 Bankruptcy Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict company financial risk using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Layout (2 columns 🔥)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
    management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
    financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    credibility = st.selectbox("Credibility", [0, 0.5, 1])
    competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
    operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Predict Risk"):

    input_data = np.array([[industrial_risk, management_risk,
                            financial_flexibility, credibility,
                            competitiveness, operating_risk]])

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    # Verified mapping:
    # 0 = Bankruptcy
    # 1 = Non-bankruptcy

    if prediction[0] == 0:
        result = "bankruptcy"
    else:
        result = "non-bankruptcy"

    # Get confidence for the predicted class
    class_index = list(model.classes_).index(prediction[0])
    confidence = prob[0][class_index] * 100

    st.markdown("---")

    # -------------------------------
    # Result Card
    # -------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if result == "bankruptcy":
        st.error("⚠️ High Risk: Company may go Bankrupt")
    else:
        st.success("✅ Low Risk: Company is Safe")

    st.write(f"### 📊 Prediction Confidence: {confidence:.0f}%")

    st.markdown("</div>", unsafe_allow_html=True)
