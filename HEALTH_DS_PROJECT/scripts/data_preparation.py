import pandas as pd
import numpy as np  # Use NumPy directly

# Load datasets
heart_data = pd.read_csv("data/processed/heart_cleaned.csv")
sleep_data = pd.read_csv("data/processed/sleep_cleaned.csv")

# Preview columns for verification
print("Heart Data Columns:", heart_data.columns)
print("Sleep Data Columns:", sleep_data.columns)

# Engineer health status for heart data
heart_data['health_status'] = np.where(
    (heart_data['physicalactivity'] == 'Yes') & (heart_data['genhealth'] == 'Very good'),
    'healthy',
    np.where(
        (heart_data['physicalactivity'] == 'No') | (heart_data['genhealth'].isin(['Fair', 'Poor'])),
        'unhealthy',
        'moderate'
    )
)

# Engineer health status for sleep data
sleep_data['health_status'] = np.where(
    (sleep_data['physical activity level'] > 60) & (sleep_data['sleep duration'] >= 7),
    'healthy',
    np.where(
        (sleep_data['physical activity level'] < 30) | (sleep_data['sleep duration'] < 5),
        'unhealthy',
        'moderate'
    )
)

# Save processed datasets
heart_data.to_csv("data/processed/heart_prepared.csv", index=False)
sleep_data.to_csv("data/processed/sleep_prepared.csv", index=False)

print("Heart data and sleep data have been processed and saved.")
