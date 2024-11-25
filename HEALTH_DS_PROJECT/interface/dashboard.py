import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load("models/heart_model.pkl")
label_encoders = joblib.load("models/heart_label_encoders.pkl")
feature_names = joblib.load("models/heart_feature_names.pkl")

# Streamlit UI
st.title("Heart Disease Prediction")
st.header("Input Features")

# Create input fields
input_data = {
    "bmi": st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0),
    "smoking": st.selectbox("Smoking", ["No", "Yes"]),
    "alcoholdrinking": st.selectbox("Alcohol Drinking", ["No", "Yes"]),
    "stroke": st.selectbox("Stroke", ["No", "Yes"]),
    "diffwalking": st.selectbox("Difficulty Walking", ["No", "Yes"]),
    "physicalactivity": st.selectbox("Physical Activity", ["No", "Yes"]),
    "sleeptime": st.slider("Sleep Time (hours)", min_value=0.0, max_value=24.0, value=7.0),
    "agecategory": st.selectbox("Age Category", ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                                                 "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]),
    "race": st.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other"]),
    "sex": st.selectbox("Sex", ["Male", "Female"]),
    "diabetic": st.selectbox("Diabetic", ["No", "Yes"]),
    "asthma": st.selectbox("Asthma", ["No", "Yes"]),
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical variables
for feature, encoder in label_encoders.items():
    if feature in input_df.columns:
        input_df[feature] = encoder.transform(input_df[feature])

# Align with training feature names
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Make predictions
try:
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]
    st.subheader("Heart Disease Prediction")
    st.write(f"Prediction: {'Heart Disease Likely' if prediction == 1 else 'No Heart Disease'}")
    st.write(f"Confidence: {prediction_proba:.2f}")
except Exception as e:
    st.error(f"Error during prediction: {e}")
