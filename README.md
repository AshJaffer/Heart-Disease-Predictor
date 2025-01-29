# AI-Powered Heart Disease Predictor

<img width="1296" alt="Screenshot 2024-12-04 at 4 51 01 PM" src="https://github.com/user-attachments/assets/7685b57f-dcc2-465a-835e-3608a14567f3">

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Optimizations](#optimizations)
- [Data Sources](#data-sources)
- [Contact Information](#contact-information)

## Project Overview
This AI-powered heart disease risk prediction model assesses a user's risk of heart disease based on various lifestyle factors and health history. The project combines data from multiple healthcare datasets to provide comprehensive risk assessments with detailed explanations.

## Features
- Real-time risk assessments
- Color-coded risk and confidence levels
- Interactive health metrics analysis:
  - Age-adjusted risk calculations
  - BMI impact analysis
  - Sleep pattern evaluation
  - Mental Health Assessment
  - Physical Activity Tracking

## Technologies Used
- Python 3.7+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- SMOTE
- Seaborn
- Matplotlib

## Setup Instructions

1. Clone the Repository:
```bash
git clone https://github.com/AshJaffer/Health_DS_Project.git
cd Health_DS_Project
```

2. Create Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Create Directory Structure:
```bash
mkdir -p data/raw data/processed models
```

4. Place Datasets in data/raw:
- heart_2020_cleaned.csv
- sleep_health_and_lifestyle_dataset.csv

5. Run Processing Scripts:
```bash
python scripts/merge_datasets.py
python scripts/data_preparation.py
python scripts/eda.py
python scripts/train_heart_model_faster.py
```

6. Launch Dashboard:
```bash
streamlit run interface/dashboard.py
```

## Project Structure
```
Health_DS_Project/
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Cleaned and merged data
├── models/                   # Trained model files
├── scripts/
│   ├── merge_datasets.py
│   ├── data_preparation.py
│   ├── eda.py
│   └── train_heart_model_faster.py
├── interface/
│   └── dashboard.py         # Streamlit interface
└── README.md
```

## Model Details
The machine learning model:
- Analyzes combined data from Heart Dataset and Sleep Dataset
- Uses SMOTE for handling class imbalance
- Implements feature engineering for variable interactions
- Utilizes Random Forest Classifier for pattern recognition
- Provides risk assessments with explanations

## Optimizations
Several challenges were addressed during development:
- Created a feature mapping system for standardizing variables across datasets
- Implemented variable selection based on risk assessment significance
- Adjusted risk thresholds to correct prediction bias
- Enhanced feature engineering to improve result distribution

## Data Sources

### Heart Disease Dataset (2020)
- Source: Centers for Disease Control and Prevention (CDC)
- Title: Heart Disease Health Indicators Dataset
- Publisher: Kaggle
- URL: [Heart Disease Dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
- Description: 400k+ adult survey responses from CDC's BRFSS 2020

### Sleep Health Dataset
- Title: Sleep Health and Lifestyle Dataset
- Publisher: Kaggle
- URL: [Sleep Health Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
- Description: Sleep patterns and health metrics data from 400 individuals

## Contact Information
- Creator: Ashhad Jaffer
- Email: ajaffer@umich.edu
