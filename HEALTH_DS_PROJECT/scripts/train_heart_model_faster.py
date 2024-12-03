import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Debug: Print model features directly
print("Loading model and checking features...")
model_data = joblib.load("models/heart_model_final.pkl")
print("\nModel features from PKL file:")
print(model_data['feature_names'])

def load_model():
    model_data = joblib.load("models/heart_model_final.pkl")
    return model_data['model'], model_data['threshold'], model_data['scaler'], model_data['feature_names']

def create_feature_vector(age_numeric, bmi, physical_health, mental_health, sleep_time,
                         smoking, stroke, diff_walking, sex_male, diabetic, physical_activity,
                         feature_names):
    """Create feature vector ensuring all model features are included."""
    print("\nCreating feature vector...")
    
    # Base features
    features = {
        'bmi': bmi,
        'physical_health': physical_health,
        'mental_health': mental_health,
        'sleep_time': sleep_time,
        'heart_rate': 75,
        'stress_level': 5,
        'physical_activity_level': 30,
        'sleep_quality': 7,
        'daily_steps': 7000,
        'bp_systolic': 120,
        'bp_diastolic': 80,
        'age_numeric': age_numeric,
        'bmi_age': age_numeric * bmi,
        'activity_stress': 150,
        'smoking_Yes': int(smoking),
        'alcohol_drinking_Yes': 0,
        'stroke_Yes': int(stroke),
        'diff_walking_Yes': int(diff_walking),
        'sex_Male': int(sex_male),
        'diabetic_Yes': int(diabetic),
        'physical_activity_Yes': int(physical_activity),
        'race_Asian': 0,
        'race_Black': 0,
        'race_Hispanic': 0,
        'race_Other': 0,
        'race_White': 1,
        'diabetic_No, borderline diabetes': 0,
        'diabetic_Yes (during pregnancy)': 0,
        'asthma_Yes': 0,
        'age_bmi_risk': age_numeric * bmi / 200,
        'health_risk_score': int(stroke) + int(diff_walking) + int(diabetic),
        'high_age_risk': int(age_numeric >= 60)
    }
    
    print("\nDebug - Initial features created:", list(features.keys()))
    
    # Create initial DataFrame
    df = pd.DataFrame([features])
    print("\nDebug - Initial DataFrame columns:", df.columns.tolist())
    
    # Add missing features from model's feature_names
    for feature in feature_names:
        if feature not in df.columns:
            print(f"\nDebug - Adding missing feature: {feature}")
            df[feature] = 0
    
    # Set age category based on age_numeric
    if 25 <= age_numeric < 80:
        for cat in ['25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                   '55-59', '60-64', '65-69', '70-74', '75-79']:
            start, end = map(int, cat.split('-'))
            if start <= age_numeric < end + 1:
                print(f"\nDebug - Setting age category: age_category_{cat}")
                df[f'age_category_{cat}'] = 1
                break
    elif age_numeric >= 80:
        df['age_category_80 or older'] = 1

    print("\nDebug - Final DataFrame columns:", df.columns.tolist())
    print("\nDebug - Expected features:", feature_names)
    
    # Reorder columns to match feature_names exactly
    return df[feature_names]

def main():
    st.set_page_config(page_title="Heart Disease Risk Assessment", layout="wide")
    
    st.markdown("""
        <div style="text-align: center;">
            <h1>Heart Disease Risk Assessment</h1>
            <p style="font-size: 1.2em;">AI-Powered Health Risk Prediction Tool</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        print("\nLoading model components...")
        model, threshold, scaler, feature_names = load_model()
        print("Model loaded successfully")
        print("Feature names:", feature_names)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Information")
            age_category = st.selectbox("Age Category", 
                ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
                 '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])
            sex = st.selectbox("Sex", ["Male", "Female"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            
        with col2:
            st.subheader("Health Conditions")
            smoking = st.checkbox("Current Smoker")
            stroke = st.checkbox("History of Stroke")
            diabetes = st.checkbox("Diabetic")
            diff_walking = st.checkbox("Difficulty Walking")
            
        with col3:
            st.subheader("Lifestyle & Health Metrics")
            physical_health = st.slider("Physical Health Issues (days/month)", 0, 30, 0)
            mental_health = st.slider("Mental Health Issues (days/month)", 0, 30, 0)
            sleep_time = st.slider("Sleep Time (hours/day)", 0, 24, 7)
            physical_activity = st.checkbox("Regular Physical Activity")

        submitted = st.form_submit_button("Analyze Risk")

    if submitted:
        print("\nForm submitted, processing input...")
        age_mapping = {
            '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
            '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
            '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
            '80 or older': 82
        }
        age_numeric = age_mapping[age_category]
        print(f"Age numeric value: {age_numeric}")

        try:
            input_df = create_feature_vector(
                age_numeric=age_numeric,
                bmi=bmi,
                physical_health=physical_health,
                mental_health=mental_health,
                sleep_time=sleep_time,
                smoking=smoking,
                stroke=stroke,
                diff_walking=diff_walking,
                sex_male=(sex == "Male"),
                diabetic=diabetes,
                physical_activity=physical_activity,
                feature_names=feature_names
            )
            
            print("\nFeature vector created successfully")
            print("Input DataFrame shape:", input_df.shape)
            
            # Scale features
            scaled_input = scaler.transform(input_df)
            print("Features scaled successfully")
            
            # Make prediction
            prediction_prob = model.predict_proba(scaled_input)[0, 1]
            prediction = 1 if prediction_prob >= threshold else 0
            print(f"Prediction made: {prediction} with probability {prediction_prob}")

            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown(f"""
                        <div style='background-color: #ff4b4b; padding: 20px; border-radius: 10px;'>
                            <h2 style='color: white; margin: 0;'>High Risk Detected</h2>
                            <p style='color: white; margin: 10px 0 0 0;'>Confidence: {prediction_prob:.1%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='background-color: #00cc66; padding: 20px; border-radius: 10px;'>
                            <h2 style='color: white; margin: 0;'>Low to Moderate Risk</h2>
                            <p style='color: white; margin: 10px 0 0 0;'>Confidence: {1-prediction_prob:.1%}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Risk Factors")
                risk_factors = []
                if bmi >= 30:
                    risk_factors.append(("High BMI", "BMI of 30 or higher indicates obesity"))
                if smoking:
                    risk_factors.append(("Current Smoker", "Smoking increases heart disease risk"))
                if stroke:
                    risk_factors.append(("History of Stroke", "Previous stroke indicates higher risk"))
                if diabetes:
                    risk_factors.append(("Diabetes", "Diabetes increases heart disease risk"))
                if diff_walking:
                    risk_factors.append(("Difficulty Walking", "Mobility issues may indicate health concerns"))
                if physical_health > 14:
                    risk_factors.append(("Poor Physical Health", "Frequent health issues reported"))
                if sleep_time < 6:
                    risk_factors.append(("Insufficient Sleep", "Less than 6 hours of sleep can impact heart health"))
                if age_numeric >= 60:
                    risk_factors.append(("Advanced Age", "Age is a significant risk factor"))
                
                if risk_factors:
                    for factor, explanation in risk_factors:
                        st.markdown(f"""
                            <div style='margin-bottom: 10px;'>
                                <strong style='color: #ff4b4b;'>• {factor}</strong><br>
                                <small>{explanation}</small>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No major risk factors identified.")
        
        except Exception as e:
            print(f"\nError during prediction: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center;'>
                <p><strong>Disclaimer:</strong> This tool provides an estimate based on the information provided. 
                It should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()