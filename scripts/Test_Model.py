import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
from typing import Union
import warnings
from Model import cap_outliers, feature_extraction ,cap_target_outliers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)
df = pd.read_csv(Path(r"dataset\Test.csv"))
x_test = df.drop(columns=['trip_duration'])
x_test = feature_extraction(x_test)


y_test = df['trip_duration']

# Define boolean and numeric columns
boolean_cols = ['pickup_year', 'is_morning_rush', 'is_evening_rush', 'is_lunch_hour',
                'is_late_night', 'is_early_morning', 'is_night', 'is_monday', 
                'is_friday', 'is_sunday', 'weekend_rush_hour', 'weekday_night', 'weekend_night', 
                'is_pickup_near_JFK', 'is_dropoff_near_JFK', 'airport_JFK_trip', 'is_pickup_near_LGA',
                'is_dropoff_near_LGA', 'airport_LGA_trip', 'is_pickup_near_EWR', 'is_dropoff_near_EWR', 
                'airport_EWR_trip', 'pickup_is_manhattan', 'dropoff_is_manhattan', 'manhattan_trip', 
                'to_manhattan', 'from_manhattan', 'is_couple_trip', 'is_group_trip', 
                'is_large_group', 'distance_night', 'night_speed_bonus', 'pickup_year','store_and_fwd_flag']

    
# Select numeric columns excluding boolean columns
numeric_cols = x_test.select_dtypes(include=['float64', 'int64']).columns.difference(boolean_cols)
x_test = cap_outliers(x_test, numeric_cols, boolean_cols)
y_test = cap_target_outliers(y_test)
print(' Shape of x_test and y_test:')
print(x_test.shape, y_test.shape)

# Load the trained model
model_path = Path(r"Outputs\models\ridge_model.pkl")

model = joblib.load(model_path)

y_pred = model.predict(x_test)

# Calculate and print evaluation metrics
metrics = {
    'r2_score': r2_score(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'mae': mean_absolute_error(y_test, y_pred),
}
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

"""  Shape of x_test and y_test:
(1459, 64) (1459,)
r2_score: 0.6853
rmse: 293.8478
mae: 210.7181 """

# Save the predictions to a CSV file
output_df = pd.DataFrame({'id': df['id'], 'trip_duration': y_pred})
output_df.to_csv(Path(r"Outputs\predictions\submission.csv"), index=False)
print("Predictions saved to Outputs\predictions\submission.csv")