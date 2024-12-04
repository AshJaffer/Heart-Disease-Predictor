AI-Powered Heart Disease Predictor Model

    The project is an AI-powered heart disease risk prediction model that assesses a user’s risk of heart disease based on various lifestyle factors and 
    health history. 

Demo Screenshots


Setup (clone repo)


Create Virtual Environment

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Needed Tools/Dependancies

    -Python 3.7+
    -Streamlit
    -Pandas
    -Numpy
    -Scikit-learn
    -Joblib
    -SMOTE

Data Setup

    Creating the necessary directories:

    mkdir -p data/raw data/processed models 

    Place datasets in the data/raw folder:
        heart_2020_cleaned.csv
        sleep_health_and_lifestyle_dataset.csv

Data Processing/Training

    Run the following commands:
    
        python scripts/merge_datasets.py

        python scripts/data_preparation.py

        python scripts/eda.py

        python scripts/train_heart_model_faster.py

    Starting the dashboard:

        streamlit interface/dashboard.py


How it works and how its made:
    Tech used: Python, Streamlit, Scikit-Learn, Pandas, NumPy, Joblib, SMOTE, Random           Forest, Seaborn, Maatplotlib

    Machine Learning Model: It analyzed two datasets (Heart Dataset and Sleep Dataset)         from studies of health variables linked to heart disease and analyzed the user’s input     to provide risk assessments with explanations.
        
    Made a data processing system that merges and cleanses data from distinct relevant         healthcare datasets.

    Implemented a ML model using Scikit-Learn that used:
        -SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance                in the medical data
        -Feature Engineering for variable interactions
        -Calculates health metrics based on multiple variables

    Frontend Interface: Used Streamlit to create a dashboard that has the following:

        -Real-time risk assessments
        -Color-coded risk and confidence levels
        -Insights on user variable input 

    Health Features Tested:

            Age-adjusted risk calculations
            BMI impact analysis
            Sleep pattern evaluation
            Mental Health Assessment
            Physical Activity Tracking

Optimizations
    I ran into a quite some problems, starting from the actual merging of the sleep and heart datasets. To mitigate this, I developed a feature mapping system with a custom preprocessing pipeline to standardize variables across the two datasets.

    I wanted to then drop certain variables as some were insignificant on risk assessment measurements and weren't fit for user input. I struggled with retaining the same trained model with those dropped variables so I utilized Random Forest Classifier to focus on the data patterns rather than specific relationships between variables.

    Towards the end of the project, the accuracy of the model was a concern as it kept showing a high bias towards positive risk predictions, regardless of the variables chosen. I corrected by adjusting the risk thresholds for the variables to and enhnaced feature engineering to highlight specific variables that hold heavy weight towards the assessment to level out the results.

Lessons Learned

    This was my first machine learning project, with a topic I'm passionate about it being healthcare. The biggest lesson I've learned from it to balance the model's complexity and accuracy with interpretability. While more complex models sometimes provide slightly better accuracy, the ability to explain the model's output to users and create benefit is paramount.
