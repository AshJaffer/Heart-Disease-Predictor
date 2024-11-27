import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model, scaler, encoders, and feature names
model = joblib.load("models/heart_model_v7.pkl")
scaler = joblib.load("models/scaler_v7.pkl")
encoders = joblib.load("models/label_encoders_v7.pkl")
feature_names = joblib.load("models/heart_feature_names.pkl")

st.title("Heart Disease Prediction")

# Input Features
st.subheader("Input Features")
sex = st.selectbox("Sex", encoders["Sex"].classes_)
age_category = st.selectbox("Age Category", encoders["AgeCategory"].classes_)
race = st.selectbox("Race", encoders["Race"].classes_)
smoking = st.selectbox("Do you habitually smoke?", ["No", "Yes"])
alcohol_drinking = st.selectbox("Do you habitually consume alcohol?", ["No", "Yes"])
diff_walking = st.selectbox("Do you have difficulty walking?", ["No", "Yes"])
physical_activity = st.selectbox("Do you engage in weekly physical activity?", ["No", "Yes"])
stroke = st.selectbox("Do you have a history of stroke?", ["No", "Yes"])
diabetic = st.selectbox("Are you diabetic?", ["No", "Yes"])
asthma = st.selectbox("Do you have asthma?", ["No", "Yes"])

sleep_disorder_map = {
    "No sleep disorder": 0,
    "Insomnia": 1,
    "Sleep Apnea": 2
}
sleep_disorder_choice = st.selectbox("Do you have a sleep disorder? Select below:", list(sleep_disorder_map.keys()))
sleep_disorder = sleep_disorder_map[sleep_disorder_choice]

sleep_time = st.slider("List the average time of sleep per night (hours)", 0.0, 24.0, step=0.1)
bmi = st.slider("Select your BMI (10 <- Underweight / 60 <- Overweight)", 10.0, 60.0, step=0.1)

# Initialize numeric values
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

try:
    # Create input data
    input_data = pd.DataFrame([{**default_values, **{
        "BMI": float(bmi),
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
        "SleepTime": float(sleep_time),
        "sleep_disorder": sleep_disorder,
    }}])

    # Encode categorical features
    categorical_columns = [
        "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
        "Sex", "AgeCategory", "Race", "Diabetic", "Asthma",
        "PhysicalActivity"
    ]

    for col in categorical_columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    # Ensure numeric columns are float type
    numeric_columns = [col for col in input_data.columns if col not in categorical_columns]
    for col in numeric_columns:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

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

    # Display Feature Importance
    st.subheader("Feature Importance Analysis")
    try:
        feature_importance = pd.read_csv('models/feature_importance.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Top 10 Most Important Features:")
            top_features = feature_importance.nlargest(10, 'importance')
            for idx, row in top_features.iterrows():
                st.write(f"{row['feature']}: {row['importance']:.4f}")
        
        with col2:
            st.write("Top 10 Features (Permutation Importance):")
            top_perm_features = feature_importance.nlargest(10, 'perm_importance')
            for idx, row in top_perm_features.iterrows():
                st.write(f"{row['feature']}: {row['perm_importance']:.4f}")
        
        st.image('models/feature_importance.png')
        
    except FileNotFoundError:
        st.write("Feature importance analysis not available.")

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