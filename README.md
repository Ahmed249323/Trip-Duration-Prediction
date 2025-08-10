
# Trip Duration Prediction

A comprehensive machine learning project that predicts taxi trip duration using advanced feature engineering and Ridge regression. The model incorporates temporal patterns, geographical features, distance calculations, and location-based insights to achieve accurate predictions.

## 🚀 Project Overview

This project implements a sophisticated trip duration prediction system that processes raw taxi trip data and transforms it into 64 engineered features. The model uses Ridge regression with MinMax scaling to predict trip duration based on pickup/dropoff coordinates, temporal patterns, passenger information, and geographical landmarks.

### Key Features
- **Advanced Feature Engineering**: 64 carefully crafted features including temporal, geographical, and interaction features
- **Geospatial Analysis**: Distance calculations using Haversine formula, Manhattan distance, and Euclidean distance
- **Location Intelligence**: Proximity detection for NYC landmarks, airports, and Manhattan boundaries
- **Temporal Pattern Recognition**: Rush hour detection, weekend patterns, and time-based interactions
- **Robust Data Processing**: Outlier capping using IQR method for both features and target variable
- **Production-Ready Pipeline**: Complete preprocessing and modeling pipeline with joblib serialization

## 📊 Dataset

This project uses the **NYC Taxi Trip Duration** dataset from Kaggle:

**📥 Dataset Link**: [https://www.kaggle.com/competitions/nyc-taxi-trip-duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)

### Dataset Information
- **Source**: Kaggle Competition - NYC Taxi Trip Duration
- **Size**: ~1.5M taxi trips from NYC
- **Time Period**: January to June 2016
- **File Format**: CSV files (Train.csv, Test.csv)

### Download Instructions
1. Visit the [Kaggle competition page](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)
2. Accept the competition rules
3. Download the dataset files:
   - `train.csv` - Training data with trip duration labels
   - `test.csv` - Test data for predictions
4. Create a `dataset/` folder in your project directory
5. Place the downloaded files as:
   - `dataset/Train.csv`
   - `dataset/Test.csv`

**Note**: You'll need a Kaggle account to download the dataset. Registration is free at [kaggle.com](https://www.kaggle.com).

## 🛠 Technical Specifications

### Python Environment
- **Python Version**: 3.13.5

### Library Dependencies
```
numpy           : 2.2.0
pandas          : 2.2.3
joblib          : 1.4.2
scikit-learn    : 1.7.0
plotly          : 5.24.1
seaborn         : 0.13.2
matplotlib      : 3.9.2
scipy           : 1.14.1
```

## 📦 Installation & Setup

### Prerequisites
- Python 3.13.5
- pip package manager
- Kaggle account (for dataset download)

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Ahmed249323/Trip-Duration-Prediction.git
cd Trip-Duration-Prediction
```

2. **Create dataset directory:**
```bash
mkdir dataset
```

3. **Download dataset from Kaggle:**
   - Visit [https://www.kaggle.com/competitions/nyc-taxi-trip-duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)
   - Download `train.csv` and `test.csv`
   - Rename them to `Train.csv` and `Test.csv`
   - Place them in the `dataset/` folder

4. **Create virtual environment:**
```bash
python -m venv trip_duration_env
# On Windows:
trip_duration_env\Scripts\activate
# On macOS/Linux:
source trip_duration_env/bin/activate
```

5. **Install dependencies:**

Create a `requirements.txt` file:
```txt
numpy==2.2.0
pandas==2.2.3
joblib==1.4.2
scikit-learn==1.7.0
plotly==5.24.1
seaborn==0.13.2
matplotlib==3.9.2
scipy==1.14.1
```

Then install:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Trip-Duration-Prediction/
├── dataset/                        # Dataset files (download from Kaggle)
│   ├── Train.csv                   # Training data with trip duration labels
│   └── Test.csv                    # Test data for predictions
├── Outputs/
│   ├── models/
│   │   └── ridge_model.pkl         # Trained Ridge regression model
│   ├── predictions/
│   │   └── results.csv             # Model predictions output
│   └── reports/
│       └── Report.pdf              # Project analysis report
├── notebooks/
│   ├── figures/                    # Generated visualizations and plots
│   └── EDA.ipynb                   # Exploratory Data Analysis notebook
├── scripts/
│   ├── Model.py                    # Main training script with core functions
│   └── Test_Model.py               # Model evaluation and testing script
├── Python&Libraries version.txt   # Technical specifications document
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## 📈 Model Performance

### Evaluation Results

The Ridge regression model achieved the following performance on the test dataset:

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | **0.6853** | Coefficient of determination (68.53% variance explained) |
| **RMSE** | **293.8478** | Root Mean Square Error (seconds) |
| **MAE** | **210.7181** | Mean Absolute Error (seconds) |

### Performance Analysis

- **R² Score (0.6853)**: The model explains approximately 68.53% of the variance in trip duration, indicating good predictive performance
- **RMSE (293.8478 seconds)**: On average, predictions deviate by ~4.9 minutes from actual trip duration
- **MAE (210.7181 seconds)**: The median absolute error is ~3.5 minutes, showing robust central tendency

### Test Dataset Statistics
- **Test samples**: 1,459 trips
- **Feature dimensions**: 64 features after preprocessing
- **Target shape**: 1,459 trip duration values

## 🔧 Usage

### Training the Model

Run the complete training pipeline:
```bash
python scripts/Model.py
```

This will:
1. Load the training data from `dataset/Train.csv`
2. Extract 53 engineered features
3. Apply outlier capping using IQR method
4. Train a Ridge regression model with MinMax scaling
5. Save the trained pipeline to `Outputs/models/ridge_model.pkl`

### Testing and Evaluation

Evaluate the trained model on test data:
```bash
python scripts/Test_Model.py
```

This will:
1. Load test data from `dataset/Test.csv`
2. Apply the same feature engineering pipeline
3. Load the trained model and make predictions
4. Calculate evaluation metrics (R², RMSE, MAE)
5. Save predictions to `Outputs/predictions/results.csv`

### Exploratory Data Analysis

Explore the data using the provided Jupyter notebook:
```bash
jupyter notebook notebooks/EDA.ipynb
```

The EDA notebook includes:
- Data distribution analysis
- Feature correlation studies
- Temporal and geographical pattern visualization
- Generated figures saved in `notebooks/figures/`

### Model Evaluation Script

The `scripts/Test_Model.py` script provides comprehensive model evaluation:

```python
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scripts.Model import cap_outliers, feature_extraction, cap_target_outliers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load test data and preprocess
df = pd.read_csv("dataset/Test.csv")
x_test = df.drop(columns=['trip_duration'])
x_test = feature_extraction(x_test)
y_test = df['trip_duration']

# Apply same preprocessing as training
boolean_cols = [...]  # Same boolean columns as training
numeric_cols = x_test.select_dtypes(include=['float64', 'int64']).columns.difference(boolean_cols)
x_test = cap_outliers(x_test, numeric_cols, boolean_cols)
y_test = cap_target_outliers(y_test)

# Load model and predict
model = joblib.load("Outputs/models/ridge_model.pkl")
y_pred = model.predict(x_test)

# Calculate metrics
metrics = {
    'r2_score': r2_score(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'mae': mean_absolute_error(y_test, y_pred)
}

# Save predictions
output_df = pd.DataFrame({'id': df['id'], 'trip_duration': y_pred})
output_df.to_csv("Outputs/predictions/results.csv", index=False)
```

### Making Predictions on New Data

```python
import pandas as pd
import joblib
from pathlib import Path
from scripts.Model import feature_extraction, cap_outliers

# Load the trained model
model = joblib.load('Outputs/models/ridge_model.pkl')

# Prepare new data (example)
new_data = pd.DataFrame({
    'pickup_datetime': ['2024-08-10 14:30:00'],
    'pickup_longitude': [-73.9857],
    'pickup_latitude': [40.7484],
    'dropoff_longitude': [-73.9665],
    'dropoff_latitude': [40.7812],
    'passenger_count': [2],
    'store_and_fwd_flag': ['N']
})

# Apply feature engineering
features = feature_extraction(new_data)

# Apply outlier capping (same as training)
boolean_cols = ['pickup_year', 'is_morning_rush', 'is_evening_rush', 'is_lunch_hour',
                'is_late_night', 'is_early_morning', 'is_night', 'is_monday', 
                'is_friday', 'is_sunday', 'weekend_rush_hour', 'weekday_night', 'weekend_night', 
                'is_pickup_near_JFK', 'is_dropoff_near_JFK', 'airport_JFK_trip', 'is_pickup_near_LGA',
                'is_dropoff_near_LGA', 'airport_LGA_trip', 'is_pickup_near_EWR', 'is_dropoff_near_EWR', 
                'airport_EWR_trip', 'pickup_is_manhattan', 'dropoff_is_manhattan', 'manhattan_trip', 
                'to_manhattan', 'from_manhattan', 'is_couple_trip', 'is_group_trip', 
                'is_large_group', 'distance_night', 'night_speed_bonus', 'pickup_year','store_and_fwd_flag']

numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns.difference(boolean_cols)
features = cap_outliers(features, numeric_cols, boolean_cols)

# Make prediction
prediction = model.predict(features)
print(f"Predicted trip duration: {prediction[0]:.2f} seconds")
```

## 🧠 Feature Engineering

The project implements comprehensive feature engineering with **64 features** across multiple categories:

### 1. Temporal Features (12 features)
- **Time Components**: Year, hour, day of week
- **Time Periods**: Morning rush (7-9), evening rush (16-19), lunch hour (11-14)
- **Special Times**: Late night, early morning, weekend patterns
- **Day Indicators**: Monday, Friday, Sunday flags

### 2. Distance Features (15 features)
- **Basic Distances**: 
  - Haversine distance (great-circle distance)
  - Manhattan distance (L1 norm)
  - Euclidean distance (L2 norm)
- **Distance Transformations**:
  - Logarithmic, square root, squared, and cubic transformations
  - Inverse distance for proximity weighting
- **Speed Estimates**: Based on 30, 40, 50 km/h average speeds

### 3. Geographical Features (18 features)
- **NYC Center Distance**: Distance from pickup/dropoff to NYC center
- **Landmark Proximity**: Distance to Central Park, Times Square, Empire State Building
- **Airport Detection**: Proximity to JFK, LGA, EWR airports (within 2km)
- **Manhattan Boundaries**: Detection of trips within/to/from Manhattan

### 4. Interaction Features (8 features)
- **Distance-Time**: Distance × hour, Distance × day of week
- **Traffic Patterns**: Rush hour speed penalties, night speed bonuses
- **Passenger Interactions**: Distance per passenger, group trip indicators

### Key Algorithms Used

#### Haversine Distance Calculation
```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points on Earth"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return 6371 * c  # Earth radius in km
```

#### Outlier Capping (IQR Method)
```python
def cap_outliers(df, numeric_cols, boolean_cols):
    """Cap outliers using Interquartile Range method"""
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df
```

## 🤖 Machine Learning Pipeline

### Model Architecture
- **Algorithm**: Ridge Regression with L2 regularization
- **Preprocessing**: MinMax scaling (0-1 normalization)
- **Pipeline**: Scikit-learn Pipeline for end-to-end processing

### Model Training
```python
def train_ridge_model(x_train, y_train, alpha=1.0, random_state=42):
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', Ridge(alpha=alpha, random_state=random_state))
    ])
    return pipeline.fit(x_train, y_train)
```

### Data Preprocessing Steps
1. **Data Cleaning**: Remove unnecessary columns (`id`, `vendor_id`, `dropoff_datetime`)
2. **Feature Extraction**: Generate 53 engineered features
3. **Outlier Treatment**: Apply IQR-based capping to numeric features and target
4. **Scaling**: MinMax normalization for consistent feature ranges
5. **Model Training**: Ridge regression with regularization

## 📊 Data Requirements

### Input Data Format
The model expects a CSV file with the following columns:
```
- pickup_datetime: Timestamp of trip start
- pickup_longitude: Pickup longitude coordinate
- pickup_latitude: Pickup latitude coordinate
- dropoff_longitude: Dropoff longitude coordinate
- dropoff_latitude: Dropoff latitude coordinate
- passenger_count: Number of passengers
- store_and_fwd_flag: Y/N flag for store and forward
- trip_duration: Target variable (seconds) - for training only
```

### Data Validation
- **Passenger count**: Automatically clipped to range [1, 6]
- **Coordinates**: Should be valid NYC area coordinates
- **DateTime**: Must be in parseable datetime format
- **Store flag**: Mapped to binary (N=0, Y=1)

## 🌍 Geographical Intelligence

### NYC Landmarks and Airports
The model incorporates knowledge of key NYC locations:

**Landmarks:**
- Central Park: (40.7812, -73.9665)
- Times Square: (40.7580, -73.9855)
- Empire State Building: (40.7484, -73.9857)

**Airports:**
- JFK: (40.6413, -73.7781)
- LaGuardia (LGA): (40.7769, -73.8740)
- Newark (EWR): (40.6895, -74.1745)

**Manhattan Boundaries:**
- Latitude: 40.700 - 40.800
- Longitude: -74.020 - -73.930

## 🚀 Performance Features

### Optimization Techniques
- **Vectorized Operations**: NumPy-based calculations for speed
- **Memory Efficiency**: Selective feature retention
- **Pipeline Architecture**: Streamlined preprocessing and prediction
- **Joblib Serialization**: Fast model loading/saving

### Scalability Considerations
- **Batch Processing**: Can handle large datasets efficiently
- **Feature Selection**: Only 53 most relevant features retained
- **Outlier Robustness**: IQR-based capping prevents extreme values

## 🔧 Configuration

### Model Hyperparameters
- **Ridge Alpha**: 1.0 (L2 regularization strength)
- **Random State**: 42 (for reproducibility)
- **Scaler**: MinMaxScaler (0-1 normalization)

### File Paths
- **Training Data**: `dataset/Train.csv`
- **Test Data**: `dataset/Test.csv`
- **Model Output**: `Outputs/models/ridge_model.pkl`

## 📈 Model Interpretation

### Feature Importance Categories
1. **Distance Features**: Primary predictors based on trip length
2. **Temporal Features**: Rush hour and time-of-day effects
3. **Location Features**: Manhattan, airport, and landmark proximity
4. **Interaction Features**: Combined effects of distance and time

### Business Insights
- **Rush Hour Impact**: Morning and evening rush hours increase trip duration
- **Distance Relationship**: Non-linear relationship captured through transformations
- **Location Effects**: Manhattan trips vs. airport trips have different patterns
- **Passenger Impact**: Group trips may have different duration characteristics

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Ahmed** - [Ahmed249323](https://github.com/Ahmed249323)

## 🔧 System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.13.5
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: At least 1GB free space for dependencies and models
- **Internet**: Required for dataset download from Kaggle

## 📚 References

- [NYC Taxi Trip Duration Dataset](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) - Kaggle Competition
- [Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula) for great-circle distance calculations
- [Scikit-learn Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

---

⭐ If you found this project helpful, please consider giving it a star!
