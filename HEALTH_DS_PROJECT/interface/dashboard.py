import streamlit as st
import pandas as pd
import numpy as np
import joblib

FEATURE_ORDER = [
    'bmi', 'physical_health', 'mental_health', 'sleep_time', 'heart_rate',
    'stress_level', 'physical_activity_level', 'sleep_quality', 'daily_steps',
    'bp_systolic', 'bp_diastolic', 'age_numeric', 'bmi_age', 'activity_stress',
    'smoking_Yes', 'alcohol_drinking_Yes', 'stroke_Yes', 'diff_walking_Yes',
    'sex_Male', 'age_category_25-29', 'age_category_30-34', 'age_category_35-39',
    'age_category_40-44', 'age_category_45-49', 'age_category_50-54',
    'age_category_55-59', 'age_category_60-64', 'age_category_65-69',
    'age_category_70-74', 'age_category_75-79', 'age_category_80 or older',
    'race_Asian', 'race_Black', 'race_Hispanic', 'race_Other', 'race_White',
    'diabetic_No, borderline diabetes', 'diabetic_Yes', 
    'diabetic_Yes (during pregnancy)', 'physical_activity_Yes', 'asthma_Yes',
    'age_bmi_risk', 'health_risk_score', 'high_age_risk'
]

def load_model():
    model_data = joblib.load("models/heart_model_final.pkl")
    return model_data['model'], model_data['threshold'], model_data['scaler']

def create_feature_vector(age_numeric, bmi, physical_health, mental_health, sleep_time,
                         smoking, stroke, diff_walking, sex_male, diabetic, physical_activity):
    """Create feature vector with exact ordering."""
    features = {feature: 0 for feature in FEATURE_ORDER}
    
    # Update with actual values
    features.update({
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
        'race_White': 1,
        'diabetic_Yes': int(diabetic),
        'physical_activity_Yes': int(physical_activity),
        'age_bmi_risk': age_numeric * bmi / 200,
        'health_risk_score': int(stroke) + int(diff_walking) + int(diabetic),
        'high_age_risk': int(age_numeric >= 60)
    })
    
    # Set appropriate age category
    if age_numeric >= 80:
        features['age_category_80 or older'] = 1
    elif 25 <= age_numeric < 80:
        category = None
        if 25 <= age_numeric <= 29: category = '25-29'
        elif 30 <= age_numeric <= 34: category = '30-34'
        elif 35 <= age_numeric <= 39: category = '35-39'
        elif 40 <= age_numeric <= 44: category = '40-44'
        elif 45 <= age_numeric <= 49: category = '45-49'
        elif 50 <= age_numeric <= 54: category = '50-54'
        elif 55 <= age_numeric <= 59: category = '55-59'
        elif 60 <= age_numeric <= 64: category = '60-64'
        elif 65 <= age_numeric <= 69: category = '65-69'
        elif 70 <= age_numeric <= 74: category = '70-74'
        elif 75 <= age_numeric <= 79: category = '75-79'
        
        if category:
            features[f'age_category_{category}'] = 1
    
    return pd.DataFrame([features])[FEATURE_ORDER]

def main():
    st.set_page_config(page_title="Heart Disease Risk Assessment", layout="wide")
    
    st.markdown("""
        <div style="text-align: center;">
            <h1>Heart Disease Risk Assessment</h1>
            <p style="font-size: 1.2em;">AI-Powered Health Risk Prediction Tool</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        model, threshold, scaler = load_model()
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
            physical_health = st.slider("Physical Health Issues (days/month)", 0, 30, 0,
                                      help="Days in past month with physical health issues")
            mental_health = st.slider("Mental Health Issues (days/month)", 0, 30, 0,
                                    help="Days in past month with mental health issues")
            sleep_time = st.slider("Sleep Time (hours/day)", 0, 24, 7,
                                 help="Average hours of sleep per day")
            physical_activity = st.checkbox("Regular Physical Activity",
                                         help="Do you engage in regular physical activity?")

        submitted = st.form_submit_button("Analyze Risk")

    if submitted:
        try:
            age_mapping = {
                '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
                '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
                '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
                '80 or older': 82
            }
            age_numeric = age_mapping[age_category]

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
                physical_activity=physical_activity
            )
            
            # Make prediction
            prediction_prob = model.predict_proba(input_df)[0, 1]
            prediction = 1 if prediction_prob >= threshold else 0
            
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
                if not physical_activity:
                    risk_factors.append(("Limited Physical Activity", "Regular exercise helps reduce risk"))
                
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
            st.error(f"An error occurred during prediction: {str(e)}")

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center;'>
                <p><strong>Disclaimer:</strong> This tool provides an estimate based on the information provided. 
                It should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()