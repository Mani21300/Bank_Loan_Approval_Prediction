import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the trained model
with open("xgb_model.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_loan_approval(Gender, Married, Dependents, Education, Self_Employed,
                          LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Income):
    # Ensure input values are in the correct format
    input_data = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                            LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Income]])
    prediction = classifier.predict(input_data)
    return prediction

def main():
    st.title("Loan Approval Prediction")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Loan Approval ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Create input fields
    Gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    Married = st.selectbox("Married", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    Dependents = st.selectbox("Dependents", options=[0, 1, 2, 3], format_func=lambda x: str(x))
    Education = st.selectbox("Education", options=[0, 1], format_func=lambda x: "Not Graduate" if x == 0 else "Graduate")
    Self_Employed = st.selectbox("Self_Employed", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    LoanAmount = st.number_input("LoanAmount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan_Amount_Term", min_value=0)
    Credit_History = st.selectbox("Credit_History", options=[0, 1], format_func=lambda x: "Bad" if x == 0 else "Good")
    Property_Area = st.selectbox("Property_Area", options=[0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x])
    Income = st.number_input("Income", min_value=0)
    
    # Make prediction
    if st.button("Predict"):
        result = predict_loan_approval(Gender, Married, Dependents, Education, Self_Employed,
                                       LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Income)
        st.success(f'The prediction is: {"Approved" if result[0] == 1 else "Not Approved"}')
    
    if st.button("About"):
        st.text("Let's Learn")
        st.text("This app predicts loan approval status based on various inputs.")

if __name__ == '__main__':
    main()
