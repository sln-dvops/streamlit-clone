import streamlit as st
import pandas as pd
import joblib
 
# Load CART model
cart_model = joblib.load('model_cart.pkl')
 
# Feature list in the exact order
all_features = [
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
    'previous_loan_defaults_on_file',
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]
 
# Streamlit app
st.title("Loan Approval Prediction App")
 
# Introduction section
st.header("Loan Approval System")
st.write(
    "This app predicts whether a loan application is likely to be approved based on user-provided data. "
    "It uses a Decision Tree Classifier (CART model) trained on historical loan data."
)
 
# Input form for user data
st.header("User Input Form")
 
# Collecting user inputs
person_age = st.number_input("Age", min_value=18, max_value=100, step=1)
person_income = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0)
person_emp_exp = st.number_input("Years of Employment", min_value=0, max_value=50, step=1)
loan_amnt = st.number_input("Loan Amount ($)", min_value=0.0, step=1000.0)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=50.0, step=0.1)
loan_percent_income = st.number_input("Loan as % of Income", min_value=0, max_value=100, step=1)
cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, step=1)
credit_score = st.number_input("Credit Score (0-1000)", min_value=300, max_value=1000, step=1)
previous_loan_defaults_on_file = st.radio(
    "Previous Loan Defaults on File", options=["Yes", "No"]
)
 
# Home ownership
person_home_ownership = st.selectbox(
    "Home Ownership",
    options=["OWN", "RENT", "MORTGAGE"]
)
 
# Loan intent
loan_intent = st.selectbox(
    "Loan Intent",
    options=[
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
        "MEDICAL", "PERSONAL", "VENTURE"
    ]
)
 
# Prepare the input data
user_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_exp': [person_emp_exp],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
    'credit_score': [credit_score],
    'previous_loan_defaults_on_file': [1 if previous_loan_defaults_on_file == "Yes" else 0],
    'person_home_ownership_MORTGAGE': [1 if person_home_ownership == "MORTGAGE" else 0],
    'person_home_ownership_OWN': [1 if person_home_ownership == "OWN" else 0],
    'person_home_ownership_RENT': [1 if person_home_ownership == "RENT" else 0],
    'loan_intent_DEBTCONSOLIDATION': [1 if loan_intent == "DEBTCONSOLIDATION" else 0],
    'loan_intent_EDUCATION': [1 if loan_intent == "EDUCATION" else 0],
    'loan_intent_HOMEIMPROVEMENT': [1 if loan_intent == "HOMEIMPROVEMENT" else 0],
    'loan_intent_MEDICAL': [1 if loan_intent == "MEDICAL" else 0],
    'loan_intent_PERSONAL': [1 if loan_intent == "PERSONAL" else 0],
    'loan_intent_VENTURE': [1 if loan_intent == "VENTURE" else 0],
})
 
# Add missing features to match the model's expected input
for col in all_features:
    if col not in user_data.columns:
        user_data[col] = 0
 
# Ensure feature order matches the model's expected order
user_data = user_data[all_features]
 
# Prediction section
st.subheader("Prediction")
 
if st.button("Predict"):
    try:
        prediction = cart_model.predict(user_data)[0]

        if prediction == 1:
            st.image("approved.png")
            st.write("loan approved")
            st.write(prediction)
        else:
            st.image("rejected.png")
            st.write("loan rejected")
            st.write(prediction)
    except Exception as e:
        st.error(f"Error during prediction: {e}")