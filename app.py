import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# Load the trained XGBoost model
model_filename = "xgboost_model.pkl"  # Updated model filename
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found! Ensure 'xgboost_model.pkl' exists in the directory.")
    st.stop()

# Load selected features
feature_filename = "selected_features.pkl"
try:
    with open(feature_filename, 'rb') as file:
        selected_features = pickle.load(file)
except FileNotFoundError:
    st.error("Feature selection file not found! Ensure 'selected_features.pkl' exists.")
    st.stop()

# Load the scaler
scaler_filename = "scaler.pkl"
try:
    with open(scaler_filename, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Scaler file not found! Ensure 'scaler.pkl' exists.")
    st.stop()

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease risk.")

# User input fields
def user_input():
    user_data = {}
    for feature in selected_features:
        user_data[feature] = st.number_input(f"Enter {feature}", min_value=0)
    return user_data

user_values = user_input()

# Predict button
if st.button("Predict"): 
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_values], columns=selected_features)
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_proba = model.predict_proba(input_scaled)[0]
    confidence_score = max(prediction_proba)
    prediction = np.argmax(prediction_proba)
    
    # Display result
    result_text = "Presence of Heart Disease: YES,Consult a Doctor" if prediction == 1 else "Presence of Heart Disease: NO,Maintain a Healthy LifeStyle"
    st.subheader(result_text)
    #st.write(f"Confidence Score: {confidence_score:.2f}")
