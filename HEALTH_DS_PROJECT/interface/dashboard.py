import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, encoders, and feature names
model = joblib.load("models/heart_model_v7.pkl")
scaler = joblib.load("models/scaler_v7.pkl")
encoders = joblib.load("models/label_encoders_v7.pkl")
feature_names = joblib.load("models/heart_feature_names.pkl")

st.markdown("""
    <div style="display: flex; justify-content: center;">
        <h1>Heart Disease Prediction</h1>
    </div>
""", unsafe_allow_html=True)

# Create a form
with st.form("prediction_form"):
    st.subheader("Input Features")
    
    # Input Features
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
    sleep_disorder_choice = st.selectbox("Do you have a sleep disorder? Select below:", 
                                       list(sleep_disorder_map.keys()))
    sleep_disorder = sleep_disorder_map[sleep_disorder_choice]

    sleep_time = st.slider("List the average time of sleep per night (hours)", 
                          0.0, 24.0, step=0.1)
    bmi = st.slider("Select your BMI (10 ← Underweight / 60 ← Overweight)", 
                    10.0, 60.0, step=0.1)

    # Submit button
    submitted = st.form_submit_button("PREDICT", use_container_width=True)

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

# Only show prediction and analysis if form is submitted
if submitted:
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

        # Show results in an expander
        with st.expander("View Prediction Results"):
            if prediction:
                st.markdown("""
                    <div style='background-color: #e72b1d; padding: 20px; border-radius: 5px;'>
                        <h3 style='color: #FFFFFF; margin: 0;'>Heart Disease Likely</h3>
                        <p style='color: #FFFFFF; margin: 10px 0 0 0;'>Confidence: {:.2%}</p>
                    </div>
                """.format(confidence), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background-color: #28a745; padding: 20px; border-radius: 5px;'>
                        <h3 style='color: #FFFFFF; margin: 0;'>Heart Disease Unlikely</h3>
                        <p style='color: #FFFFFF; margin: 10px 0 0 0;'>Confidence: {:.2%}</p>
                    </div>
                """.format(confidence), unsafe_allow_html=True)

            # Feature Importance Analysis
            st.subheader("Importance Of Each Variable")
            try:
                feature_importance = pd.read_csv('models/feature_importance.csv')
                
                # Create a container with centered content
                left, middle, right = st.columns([1,2,1])
                with middle:
                    st.markdown("""
                        <div style='text-align: center;'>
                            <h4>Most Important Features:</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    top_features = feature_importance.nlargest(5, 'importance')
                    for idx, row in top_features.iterrows():
                        st.markdown(f"""
                            <div style='text-align: center;'>
                                {row['feature']}: {row['importance']:.4f}
                            </div>
                        """, unsafe_allow_html=True)

            except FileNotFoundError:
                st.write("Feature importance analysis not available.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")