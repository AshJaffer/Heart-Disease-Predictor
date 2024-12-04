AI Powered Heart Disease Predictor Model

    The project is an AI-powered heart disease risk prediction model that assesses a user’s risk of heart disease based on various lifestyle factors and health history. 

How it works and how its made:
    Tech used: Python, Streamlit, Scikit-Learn, Pandas, NumPy, Joblib, SMOTE, Random Forest, Seaborn, Maatplotlib

    Machine Learning Model: analyzed two datasets (Heart Dataset and Sleep Dataset) from studies of health variables linked to heart disease, and it analyzes the user’s input to provide risk assessments with explanations.
        
        Made a data processsing system that merges and cleanses data from distinct relevant healthcare datasets.

        Implmented a ML model using Scikit-Learn that used:
            SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance in the medical data
            Feature Engineering for variable interactions
            Calculates health metrics based on multiple variables

    -Frontend Interface: Used Streamlit to create a dashboard that has the following:

        Real time risk assessments
        Color coded risk and confidence levels
        Insights on user variable input 

    -Health Features Tested:

            Age adjusted risk calculations
            BMI impact analysis
            Sleep pattern evaluation
            Mental Health Assessment
            Physical Activity Tracking

Optimizations
    I ran into a quite a number of problems, starting from the actual merging of the sleep and heart datasets. To mitigate, I developed a feature mapping system with a custome preprocessing pipeline to standardize variables across the two datasets.

    I wanted to then drop certain variables as some were insignificant on risk assessment measurements and weren't fit for user input. I struggled with retaining the same trained model with those dropeed variables so I uitilized Random Forest Classifier to focus on the data patterns rather than specific relationships between variables.

    Towards the end of the project, the accuracy of the model was a concern as it kept showing a high bias towards positive risk predictions, regardless of the variables chosen. I corrected by adjusting the risk thresholds for the variables to and enhnaced feature engineering to highlight specific variables that hold heavy weight towards the assessment to level out the results.

Lessons Learned

    This was my first machine learning project, with a topic I'm passionate about it being healthcare. The biggest lesson I've learned from it to balance the model complexity and accuracy with interpretability. While more complex models sometimes provided slightly better accuracy, the ability to explain the model's output to users and create benefit is paramount.