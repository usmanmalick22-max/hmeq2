
-*- coding: utf-8 -*-
GitHub Permalink for hmeq_model.pkl: https://github.com/usmanmalick22-max/hmeq2/blob/4a6d0ba84f35c9d83cd131e94972008940ad7e9a/hmeq_model.pkl
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model from the same folder as this script
with open("hmeq_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
# st.title("Home Equity Loan Approval")
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Home Equity Loan Approval</b></h1>",
    unsafe_allow_html=True
)


# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numeric values
loan = st.slider("Loan Amount (LOAN)", min_value=1000, max_value=500000, step=1000)
mortdue = st.slider("Mortgage Due (MORTDUE)", min_value=0.0, max_value=1000000.0, step=1000.0)
value = st.slider("Property Value (VALUE)", min_value=0.0, max_value=1000000.0, step=1000.0)
yoj = st.selectbox("Years at Job (YOJ)", options=list(range(1, 41)))  # Options from 1 to 40
derog = st.number_input("Derogatory Reports (DEROG)", min_value=0, max_value=15, step=1)
delinq = st.selectbox("Delinquent Reports (DELINQ)", options=list(range(0, 15)))  # Options from 0 to 10
clage = st.slider("Age of Oldest Trade Line in Months (CLAGE)", min_value=0.0, max_value=100.0, step=1.0)
ninq = st.slider("Number of Recent Credit Inquiries (NINQ)", min_value=0.0, max_value=15.0, step=1.0)
clno = st.slider("Number of Credit Lines (CLNO)", min_value=0.0, max_value=50.0, step=1.0)
debtinc = st.slider("Debt-to-Income Ratio (DEBTINC)", min_value=0.0, max_value=200.0, step=0.1)

# Categorical inputs with options
reason = st.selectbox("Reason for Loan (REASON)", ["HomeImp", "DebtCon"])
job = st.selectbox("Job Category (JOB)", ["ProfExe", "Other", "Mgr", "Office", "Sales"])

# Create the input data as a DataFrame
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

# -----------------------
#  NO ONE-HOT ENCODING
#  PASS RAW DATA TO PIPELINE MODEL
# -----------------------

# Predict button
if st.button("Evaluate Loan"):
    # Predict using the loaded model
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.write("The prediction is: **Bad Loan** 🚫")
    else:
        st.write("The prediction is: **Good Loan** 💲")
