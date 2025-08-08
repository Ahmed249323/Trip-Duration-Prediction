import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
from typing import Union
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def haversine_distance(lat1, lon1, lat2, lon2):
    """    Calculate the Haversine distance between two points on the Earth specified in decimal degrees.

    Parameters:
    - lat1, lon1: Latitude and longitude of the first point.
    - lat2, lon2: Latitude and longitude of the second point.
    Returns:
    - Distance in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def feature_extraction(df):

    """Extract features from the dataset.

    Parameters:
    - df: Pandas DataFrame containing the raw data.

    Returns:
    - A new DataFrame with extracted features.
    """
    df = df.copy()
    for col in ['Unnamed: 0','id','dropoff_datetime','vendor_id'] :
        if col in df.columns:
            df.drop(columns=[col] , inplace=True)

    # ==============================
    # TIME FEATURES
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_is_weekend'] = (df['pickup_dayofweek'] >= 5).astype(int)

    df['is_morning_rush'] = df['pickup_hour'].between(7, 9).astype(int)
    df['is_evening_rush'] = df['pickup_hour'].between(16, 19).astype(int)
    df['is_lunch_hour'] = df['pickup_hour'].between(11, 14).astype(int)
    df['is_late_night'] = df['pickup_hour'].between(22, 23).astype(int)
    df['is_early_morning'] = df['pickup_hour'].between(0, 6).astype(int)
    df['is_night'] = df['pickup_hour'].apply(lambda x: 1 if (x >= 22 or x <= 5) else 0)
    df['is_monday'] = (df['pickup_dayofweek'] == 0).astype(int)
    df['is_friday'] = (df['pickup_dayofweek'] == 4).astype(int)
    df['is_sunday'] = (df['pickup_dayofweek'] == 6).astype(int)
    df['weekend_rush_hour'] = ((df['pickup_is_weekend']) & (df['is_morning_rush'] | df['is_evening_rush'])).astype(int)
    df['weekday_night'] = ((df['pickup_is_weekend'] == 0) & (df['is_night'] == 1)).astype(int)
    df['weekend_night'] = ((df['pickup_is_weekend'] == 1) & (df['is_night'] == 1)).astype(int)

    # ==============================
    # DISTANCES
    df['delta_longitude'] = df['dropoff_longitude'] - df['pickup_longitude']
    df['delta_latitude'] = df['dropoff_latitude'] - df['pickup_latitude']
    df['abs_delta_longitude'] = df['delta_longitude'].abs()
    df['abs_delta_latitude'] = df['delta_latitude'].abs()

    df['euclidean_distance'] = np.sqrt(df['delta_longitude']**2 + df['delta_latitude']**2)
    df['manhattan_distance'] = df['abs_delta_longitude'] + df['abs_delta_latitude']

    df['haversine_distance'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['log_haversine_distance'] = np.log1p(df['haversine_distance'])
    df['sqrt_haversine_distance'] = np.sqrt(df['haversine_distance'])
    df['haversine_distance_sq'] = df['haversine_distance'] ** 2
    df['haversine_distance_cube'] = df['haversine_distance'] ** 3
    df['inv_haversine_distance'] = 1 / (df['haversine_distance'] + 1e-6)

    # SPEED ESTIMATES
    df['estimated_speed_30'] = df['haversine_distance'] / (30 / 3600)
    df['estimated_speed_40'] = df['haversine_distance'] / (40 / 3600)
    df['estimated_speed_50'] = df['haversine_distance'] / (50 / 3600)

    # DISTANCE-TIME INTERACTIONS
    df['distance_hour'] = df['haversine_distance'] * df['pickup_hour']
    df['distance_dayofweek'] = df['haversine_distance'] * df['pickup_dayofweek']
    df['distance_rush'] = df['haversine_distance'] * ((df['is_morning_rush'] + df['is_evening_rush']) > 0).astype(int)
    df['distance_night'] = df['haversine_distance'] * df['is_night']
    df['rush_hour_speed_penalty'] = df['haversine_distance'] * ((df['is_morning_rush'] + df['is_evening_rush']) > 0).astype(int) * 1.5
    df['night_speed_bonus'] = df['haversine_distance'] * df['is_night'] * 0.7

    # PASSENGER FEATURES
    df['passenger_count'] = df['passenger_count'].clip(1, 6)
    df['distance_passenger'] = df['haversine_distance'] * df['passenger_count']
    df['distance_per_passenger'] = df['haversine_distance'] / df['passenger_count']
    df['is_couple_trip'] = (df['passenger_count'] == 2).astype(int)
    df['is_group_trip'] = (df['passenger_count'] > 2).astype(int)
    df['is_large_group'] = (df['passenger_count'] > 4).astype(int)

    # STORE AND FWD
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1})

    # CENTER DISTANCE
    NYC_CENTER = {'lat': 40.7580, 'lon': -73.9855}
    df['pickup_center_distance'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], NYC_CENTER['lat'], NYC_CENTER['lon'])
    df['dropoff_center_distance'] = haversine_distance(df['dropoff_latitude'], df['dropoff_longitude'], NYC_CENTER['lat'], NYC_CENTER['lon'])
    df['total_center_distance'] = df['pickup_center_distance'] + df['dropoff_center_distance']
    df['center_dist_hour'] = df['pickup_center_distance'] * df['pickup_hour']

    # LANDMARKS
    LANDMARKS = {
        'central_park': {'lat': 40.7812, 'lon': -73.9665},
        'times_square': {'lat': 40.7580, 'lon': -73.9855},
        'empire_state': {'lat': 40.7484, 'lon': -73.9857}
    }
    for name, loc in LANDMARKS.items():
        df[f'pickup_dist_{name}'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], loc['lat'], loc['lon'])
        df[f'dropoff_dist_{name}'] = haversine_distance(df['dropoff_latitude'], df['dropoff_longitude'], loc['lat'], loc['lon'])

    # AIRPORTS
    AIRPORTS = {
        'JFK': {'lat': 40.6413, 'lon': -73.7781},
        'LGA': {'lat': 40.7769, 'lon': -73.8740},
        'EWR': {'lat': 40.6895, 'lon': -74.1745}
    }
    for code, loc in AIRPORTS.items():
        df[f'dist_pickup_{code}'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], loc['lat'], loc['lon'])
        df[f'dist_dropoff_{code}'] = haversine_distance(df['dropoff_latitude'], df['dropoff_longitude'], loc['lat'], loc['lon'])
        df[f'is_pickup_near_{code}'] = (df[f'dist_pickup_{code}'] < 2).astype(int)
        df[f'is_dropoff_near_{code}'] = (df[f'dist_dropoff_{code}'] < 2).astype(int)
        df[f'airport_{code}_trip'] = (df[f'is_pickup_near_{code}'] | df[f'is_dropoff_near_{code}']).astype(int)

    # MANHATTAN
    df['pickup_is_manhattan'] = df['pickup_latitude'].between(40.700, 40.800) & df['pickup_longitude'].between(-74.020, -73.930)
    df['dropoff_is_manhattan'] = df['dropoff_latitude'].between(40.700, 40.800) & df['dropoff_longitude'].between(-74.020, -73.930)
    df['manhattan_trip'] = (df['pickup_is_manhattan'] & df['dropoff_is_manhattan']).astype(int)
    df['to_manhattan'] = ((~df['pickup_is_manhattan']) & df['dropoff_is_manhattan']).astype(int)
    df['from_manhattan'] = (df['pickup_is_manhattan'] & (~df['dropoff_is_manhattan'])).astype(int)

    # FINAL SELECTION

    selected_features = [
        'abs_delta_latitude', 'abs_delta_longitude', 'center_dist_hour',
        'dist_pickup_JFK', 'distance_dayofweek', 'distance_hour',
        'distance_passenger', 'distance_per_passenger', 'distance_rush',
        'dropoff_center_distance', 'dropoff_dist_central_park',
        'dropoff_dist_empire_state', 'dropoff_dist_times_square',
        'estimated_speed_30', 'estimated_speed_40', 'estimated_speed_50',
        'euclidean_distance', 'haversine_distance', 'haversine_distance_cube',
        'haversine_distance_sq', 'inv_haversine_distance',
        'log_haversine_distance', 'manhattan_distance',
        'pickup_center_distance', 'pickup_dist_central_park',
        'pickup_dist_empire_state', 'pickup_dist_times_square',
        'rush_hour_speed_penalty', 'sqrt_haversine_distance',
        'total_center_distance', 'pickup_year',
        'is_morning_rush', 'is_evening_rush', 'is_lunch_hour', 'is_late_night',
        'is_early_morning', 'is_night', 'is_monday', 'is_friday', 'is_sunday',
        'weekend_rush_hour', 'weekday_night', 'weekend_night',
        'is_pickup_near_JFK', 'is_dropoff_near_JFK', 'airport_JFK_trip',
        'is_pickup_near_LGA', 'is_dropoff_near_LGA', 'airport_LGA_trip',
        'is_pickup_near_EWR', 'is_dropoff_near_EWR', 'airport_EWR_trip',
        'pickup_is_manhattan', 'dropoff_is_manhattan', 'manhattan_trip',
        'to_manhattan', 'from_manhattan', 'is_couple_trip', 'is_group_trip',
        'is_large_group', 'distance_night', 'night_speed_bonus',
        'store_and_fwd_flag'
    ]

    return df[selected_features]

def  cap_outliers(df, numeric_cols, boolean_cols):
    """Cap outliers in numeric columns using IQR method."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)
    return df[numeric_cols.tolist() + boolean_cols]

def cap_target_outliers(y):
    """
    Cap outliers in the target variable using the IQR method.

    Parameters:
    - y: Pandas Series representing the target variable.

    Returns:
    - A new Series with outliers capped.
    """
    y = y.copy()
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return y.clip(lower, upper)

def train_ridge_model(x_train, y_train, alpha=1.0, random_state=42):
    """
    Trains a Ridge regression model using MinMax scaling.

    Parameters:
    - x_train: Training features (DataFrame or ndarray)
    - y_train: Training target (Series or ndarray)
    - alpha: Regularization strength for Ridge (default=1.0)
    - random_state: Random seed for reproducibility (default=42)

    Returns:
    - Trained Pipeline model
    """
    pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('model', Ridge(alpha=alpha, random_state=random_state))
    ])
    
    model = pipeline.fit(x_train, y_train)
    return model

def save_pipeline(pipeline, model_path: Union[str, Path] = 'model.pkl'):
    """
    Save the trained pipeline to disk using joblib.

    Parameters:
    - pipeline: Trained pipeline object (e.g., sklearn Pipeline)
    - model_path: File path to save the model (default: 'model.pkl')
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Create folder if not exists

    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to: {model_path}")

def main():
    """Main function to train Ridge model"""
        # Load dataset
    df = pd.read_csv(r'dataset\Train.csv')
    # Feature extraction
    x = feature_extraction(df)
    y = df['trip_duration']
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
    numeric_cols = x.select_dtypes(include=['float64', 'int64']).columns.difference(boolean_cols)

        # Cap outliers in numeric columns
    x = cap_outliers(x, numeric_cols, boolean_cols)

    # Cap outliers in target variable
    y = cap_target_outliers(y)

    # Train Ridge model
    model = train_ridge_model(x, y)

    # Save the trained model
    save_pipeline(model, r'Outputs\models\ridge_model.pkl')


if __name__ == "__main__":
    main()