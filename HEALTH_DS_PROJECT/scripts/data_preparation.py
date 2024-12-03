import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_data(health_data_path, sleep_data_path):
    """
    Improved data preparation pipeline with proper merging and feature engineering.
    """
    # Load datasets
    health_data = pd.read_csv(health_data_path, low_memory=False)
    sleep_data = pd.read_csv(sleep_data_path)
    
    # Clean column names
    health_data.columns = health_data.columns.str.strip().str.lower()
    sleep_data.columns = sleep_data.columns.str.strip().str.lower()
    
    # Store target variable before preprocessing
    y = (health_data['heartdisease'] == 'Yes').astype(int)
    
    # Select and rename columns we want to keep (excluding target)
    columns_mapping = {
        'bmi': 'bmi',
        'smoking': 'smoking',
        'alcoholdrinking': 'alcohol_drinking',
        'stroke': 'stroke',
        'physicalhealth': 'physical_health',
        'mentalhealth': 'mental_health',
        'diffwalking': 'diff_walking',
        'sex': 'sex',
        'agecategory': 'age_category',
        'race': 'race',
        'diabetic': 'diabetic',
        'physicalactivity': 'physical_activity',
        'sleeptime': 'sleep_time',
        'asthma': 'asthma',
        'heart rate': 'heart_rate',
        'stress level': 'stress_level',
        'blood pressure': 'blood_pressure',
        'physical activity level': 'physical_activity_level',
        'quality of sleep': 'sleep_quality',
        'daily steps': 'daily_steps'
    }
    
    # Keep only columns that exist in the dataset
    available_columns = [col for col in columns_mapping.keys() if col in health_data.columns]
    health_data = health_data[available_columns]
    
    # Rename columns
    health_data = health_data.rename(columns=columns_mapping)
    
    print("\nSelected features after renaming:")
    print(health_data.columns.tolist())
    
    # Process blood pressure column if available
    if 'blood_pressure' in health_data.columns:
        try:
            health_data[['bp_systolic', 'bp_diastolic']] = health_data['blood_pressure'].str.split('/', expand=True).astype(float)
            health_data.drop('blood_pressure', axis=1, inplace=True)
        except:
            health_data.drop('blood_pressure', axis=1, inplace=True)
            print("Could not process blood pressure column, dropping it.")
    
    # Create age numeric feature if age_category is available
    if 'age_category' in health_data.columns:
        age_mapping = {
            '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
            '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
            '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
            '80 or older': 82
        }
        health_data['age_numeric'] = health_data['age_category'].map(age_mapping)
    
    # Handle missing values
    numeric_cols = health_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        health_data[col] = health_data[col].fillna(health_data[col].median())
    
    # Create interaction features if possible
    if 'bmi' in health_data.columns and 'age_numeric' in health_data.columns:
        health_data['bmi_age'] = health_data['bmi'] * health_data['age_numeric']
    
    if 'physical_activity_level' in health_data.columns and 'stress_level' in health_data.columns:
        health_data['activity_stress'] = health_data['physical_activity_level'] * health_data['stress_level']
    
    # Convert boolean columns to int
    bool_columns = health_data.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        health_data[col] = health_data[col].astype(int)
    
    # Encode categorical variables
    categorical_cols = health_data.select_dtypes(include=['object']).columns
    health_data = pd.get_dummies(health_data, columns=categorical_cols, drop_first=True)
    
    print("\nFinal features before splitting:")
    print(health_data.columns.tolist())
    
    # Split data BEFORE applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(health_data, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale only numeric features
    scaler = StandardScaler()
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # Apply SMOTE only to training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test, scaler

if __name__ == "__main__":
    health_data_path = "data/processed/merged_data.csv"
    sleep_data_path = "data/raw/sleep_health_and_lifestyle_dataset.csv"
    
    try:
        X_train, X_test, y_train, y_test, scaler = prepare_data(health_data_path, sleep_data_path)
        
        print("\nTraining set shape:", X_train.shape)
        print("Testing set shape:", X_test.shape)
        print("Class distribution in training set:", np.bincount(y_train))
        
        # Create the processed directory if it doesn't exist
        os.makedirs("data/processed", exist_ok=True)
        
        # Save the prepared data
        joblib.dump(X_train, "data/processed/X_train.pkl")
        joblib.dump(X_test, "data/processed/X_test.pkl")
        joblib.dump(y_train, "data/processed/y_train.pkl")
        joblib.dump(y_test, "data/processed/y_test.pkl")
        joblib.dump(scaler, "data/processed/scaler.pkl")
        
        print("\nData preparation completed successfully!")
        
    except Exception as e:
        print(f"\nError during data preparation: {str(e)}")
        print("\nPlease check the column names in your dataset and adjust the script accordingly.")