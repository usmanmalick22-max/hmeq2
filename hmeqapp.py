# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # Required for pickle loading
import requests  # To download pickle from URL
from io import BytesIO

# Permalink to the trained model (RAW GitHub link)
MODEL_URL = "https://raw.githubusercontent.com/usmanmalick22-max/hmeq2/main/hmeq_model.pkl"

# Load the model directly from GitHub
@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    if response.status_code != 200:
        st.error("Failed to download model file from GitHub URL.")
        return None
    return pickle.load(BytesIO(response.content))

model = load_model()

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Home Equity Loan Approval</b></h1>",
    unsafe_allow_html=True
)

# Show model link
st.markdown(f"[Model File Permalink]({MODEL_URL})")

# Numeric inputs
st.header("Enter Loan Applicant's Details")

loan = st.slider("Loan Amount (LOAN)", min_value=1000, max_value=500000, step=1000)
mortdue = st.slider("Mortgage Due (MORTDUE)", min_value=0.0, max_value=1000000.0, step=1000.0)
value = st.slider("Property Value (VALUE)", min_value=0.0, max_value=1000000.0, step=1000.0)
yoj = st.selectbox("Years at Job (YOJ)", options=list(range(1, 41)))
derog = st.number_input("Derogatory Reports (DEROG)", min_value=0, max_value=15, step=1)
delinq = st.selectbox("Delinquent Reports (DELINQ)", options=list(range(0, 15)))
clage = st.slider("Age of Oldest Trade Line in Months (CLAGE)", min_value=0.0, max_value=100.0, step=1.0)
ninq = st.slider("Number of Recent Credit Inquiries (NINQ)", min_value=0.0, max_value=15.0, step=1.0)
clno = st.slider("Number of Credit Lines (CLNO)", min_value=0.0, max_value=50.0, step=1.0)
debtinc = st.slider("Debt-to-Income Ratio (DEBTINC)", min_value=0.0, max_value=200.0, step=0.1)

reason = st.selectbox("Reason for Loan (REASON)", ["HomeImp", "DebtCon"])
job = st.selectbox("Job Category (JOB)", ["ProfExe", "Other", "Mgr", "Office", "Sales"])

# Create a DataFrame
input_data = pd.DataFrame({
    "LOAN": [loan],
    "MORTDUE": [mortdue],
    "VALUE": [value],
    "YOJ": [yoj],
    "DEROG": [derog],
    "DELINQ": [delinq],
    "CLAGE": [clage],
    "NINQ": [ninq],
    "CLNO": [clno],
    "DEBTINC": [debtinc],
    "REASON": [reason],
    "JOB": [job]
})

# Prediction button
if st.button("Evaluate Loan"):
    if model is None:
        st.error("Model could not be loaded, cannot perform prediction.")
    else:
        # Use NumPy array so there are no feature names (fixes the ValueError)
        X = input_data.values
        prediction = model.predict(X)[0]

        if prediction == 1:
            st.write("The prediction is: **Bad Loan** 🚫")
        else:
            st.write("The prediction is: **Good Loan** 💲")
