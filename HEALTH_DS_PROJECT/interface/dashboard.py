import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessing objects
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/heart_scaler.pkl")
feature_names = joblib.load("models/heart_feature_names.pkl")
label_encoders = joblib.load("models/heart_label_encoders.pkl")

# Page title
st.title("Heart Disease Prediction")

# Input section for user features
st.sidebar.header("Input Features")

input_features = {}

input_features["bmi"] = st.sidebar.slider("BMI", min_value=10.0, max_value=60.0, step=0.1, value=25.0)
input_features["smoking"] = st.sidebar.selectbox("Smoking", ["No", "Yes"])
input_features["alcoholdrinking"] = st.sidebar.selectbox("Alcohol Drinking", ["No", "Yes"])
input_features["stroke"] = st.sidebar.selectbox("Stroke", ["No", "Yes"])
input_features["diffwalking"] = st.sidebar.selectbox("Difficulty Walking", ["No", "Yes"])
input_features["physicalactivity"] = st.sidebar.slider("Physical Activity Level", min_value=0.0, max_value=100.0, step=1.0, value=30.0)
input_features["generalhealth"] = st.sidebar.selectbox("General Health", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
input_features["asthma"] = st.sidebar.selectbox("Asthma", ["No", "Yes"])
input_features["kidneydisease"] = st.sidebar.selectbox("Kidney Disease", ["No", "Yes"])
input_features["skincancer"] = st.sidebar.selectbox("Skin Cancer", ["No", "Yes"])
input_features["agecategory"] = st.sidebar.selectbox("Age Category", ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
input_features["race"] = st.sidebar.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other"])
input_features["sex"] = st.sidebar.selectbox("Sex", ["Female", "Male"])
input_features["diabetic"] = st.sidebar.selectbox("Diabetic", ["No", "Yes"])
input_features["mentalhealth"] = st.sidebar.slider("Mental Health (days)", min_value=0, max_value=30, step=1, value=0)
input_features["physicalhealth"] = st.sidebar.slider("Physical Health (days)", min_value=0, max_value=30, step=1, value=0)
input_features["sleeptime"] = st.sidebar.slider("Sleep Time (hours)", min_value=0.0, max_value=24.0, step=0.1, value=7.0)

# Convert input features into a DataFrame
input_df = pd.DataFrame([input_features])

# Encode categorical features
error_message = ""
for feature, encoder in label_encoders.items():
    if feature in input_df.columns:
        try:
            input_df[feature] = encoder.transform(input_df[feature])
        except ValueError as e:
            error_message += f"Invalid input for {feature}. Resetting to default.\n"
            input_df[feature] = encoder.transform([encoder.classes_[0]])  # Default to first class

# Ensure input features align with model requirements
for feature in feature_names:
    if feature not in input_df.columns:
        input_df[feature] = 0  # Add missing feature with default value
input_df = input_df[feature_names]  # Ensure order matches model input

# Scale input features
scaled_input = scaler.transform(input_df)

# Make prediction
prediction = model.predict(scaled_input)
confidence = model.predict_proba(scaled_input).max()

# Display the prediction
st.subheader("Heart Disease Prediction")
if error_message:
    st.warning(error_message.strip())  # Display warnings if any invalid input was handled
st.write(f"Prediction: {'Heart Disease Likely' if prediction[0] == 1 else 'Heart Disease Unlikely'}")
st.write(f"Confidence: {confidence:.2f}")
