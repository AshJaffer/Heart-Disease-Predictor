import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, encoders, and feature names
model = joblib.load("models/heart_model_v5.pkl")
scaler = joblib.load("models/scaler_v5.pkl")
encoders = joblib.load("models/label_encoders_v5.pkl")
feature_names = joblib.load("models/heart_feature_names.pkl")

st.title("Heart Disease Prediction")

# Input Features
st.subheader("Input Features")
bmi = st.slider("BMI", 10.0, 60.0, step=0.1)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol_drinking = st.selectbox("Alcohol Drinking", ["No", "Yes"])
stroke = st.selectbox("Stroke", ["No", "Yes"])
diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"])
physical_activity = st.selectbox("Physical Activity", ["No", "Yes"])
sleep_time = st.slider("Sleep Time (hours)", 0.0, 24.0, step=0.1)
age_category = st.selectbox("Age Category", encoders["AgeCategory"].classes_)
race = st.selectbox("Race", encoders["Race"].classes_)
sex = st.selectbox("Sex", encoders["Sex"].classes_)
diabetic = st.selectbox("Diabetic", ["No", "Yes"])
asthma = st.selectbox("Asthma", ["No", "Yes"])
bmi_category = st.selectbox("BMI Category", encoders["bmi_category"].classes_)
sleep_disorder = st.selectbox("Sleep Disorder", encoders["sleep_disorder"].classes_)

# Default placeholders for uncollected features
default_values = {
    "PhysicalHealth": 0,
    "MentalHealth": 0,
    "person_id": 0,
    "age": 40,
    "sleep_duration": 7.0,
    "quality_of_sleep": 5,
    "physical_activity_level": 30,
    "stress_level": 5,
    "heart_rate": 70,
    "daily_steps": 7000,
    "blood_pressure_upper": 120,
    "blood_pressure_lower": 80,
}

# Create input data
input_data = pd.DataFrame([{**default_values, **{
    "BMI": bmi,
    "Smoking": smoking,
    "AlcoholDrinking": alcohol_drinking,
    "Stroke": stroke,
    "DiffWalking": diff_walking,
    "Sex": sex,
    "AgeCategory": age_category,
    "Race": race,
    "Diabetic": diabetic,
    "Asthma": asthma,
    "PhysicalActivity": physical_activity,
    "SleepTime": sleep_time,
    "bmi_category": bmi_category,
    "sleep_disorder": sleep_disorder,
}}])

try:
    # Encode categorical features
    categorical_columns = [
        "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
        "Sex", "AgeCategory", "Race", "Diabetic", "Asthma",
        "PhysicalActivity", "bmi_category", "sleep_disorder"
    ]

    for col in categorical_columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    # Align input with feature names
    aligned_data = input_data[feature_names]

    # Scale input data
    scaled_input = scaler.transform(aligned_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0][prediction]

    st.subheader("Heart Disease Prediction")
    st.write(f"Prediction: {'Heart Disease Likely' if prediction else 'Heart Disease Unlikely'}")
    st.write(f"Confidence: {confidence:.2f}")

except ValueError as e:
    st.error(f"Error during prediction: {e}")
    st.write("Debug: Ensure all input features are correctly processed.")
    st.write("Current data types:", input_data.dtypes.to_dict())
    st.write("Current columns:", input_data.columns.tolist())
except KeyError as e:
    st.error(f"Error in feature alignment: {e}")
    st.write("Debug: Ensure feature names match the model requirements.")
    st.write("Expected feature names:", feature_names)
    st.write("Current columns:", input_data.columns.tolist())
