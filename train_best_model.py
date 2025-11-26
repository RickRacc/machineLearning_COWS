"""
Train the best model: 9 engineered features + native CatBoost encoding
This is our best performer: 4.1092 validation RMSE, 4.16148 Kaggle RMSE
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import joblib

print("="*80)
print("TRAINING BEST MODEL (9 Engineered Features)")
print("="*80)

# Load and clean data
print("\n[1/7] Loading and preprocessing data...")
data = pd.read_csv("cattle_data_train.csv")

features_to_remove = [
    'Feed_Quantity_lb', 'Cattle_ID', 'Rumination_Time_hrs',
    'HS_Vaccine', 'BQ_Vaccine', 'BVD_Vaccine', 'Brucellosis_Vaccine',
    'FMD_Vaccine', 'Resting_Hours', 'Housing_Score', 'Feeding_Frequency',
    'Walking_Distance_km', 'Body_Condition_Score', 'Humidity_percent',
    'Grazing_Duration_hrs', 'Milking_Interval_hrs'
]

data_cleaned = data.drop(columns=features_to_remove)

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])
data_cleaned['Month'] = data_cleaned['Date'].dt.month
data_cleaned['Season'] = data_cleaned['Month'].apply(get_season)

data_cleaned['Breed'] = data_cleaned['Breed'].str.strip()
data_cleaned['Breed'] = data_cleaned['Breed'].replace({'Holstien': 'Holstein'})

data_cleaned = data_cleaned[data_cleaned['Milk_Yield_L'] >= 0].copy()

data_cleaned['Feed_Quantity_kg'] = data_cleaned.groupby('Feed_Type')['Feed_Quantity_kg'].transform(
    lambda x: x.fillna(x.median())
)

print(f"  After cleaning: {data_cleaned.shape}")

# Engineering features
print("\n[2/7] Engineering 9 features...")
data_original = pd.read_csv("cattle_data_train.csv")
data_original["Date"] = pd.to_datetime(data_original["Date"])
data_original["Month"] = data_original["Date"].dt.month
data_original = data_original[data_original["Milk_Yield_L"] >= 0].copy()

data_cleaned["Feed_Efficiency"] = data_cleaned["Feed_Quantity_kg"] / data_cleaned["Weight_kg"]
data_cleaned["Water_Feed_Ratio"] = data_cleaned["Water_Intake_L"] / data_cleaned["Feed_Quantity_kg"]
data_cleaned["Peak_Lactation"] = ((data_cleaned["Days_in_Milk"] >= 60) &
                                   (data_cleaned["Days_in_Milk"] <= 120)).astype(int)
data_cleaned["Heat_Stress"] = ((data_cleaned["Ambient_Temperature_C"] - 25) *
                                (data_cleaned["Season"] == "Summer").astype(int)).clip(lower=0)
data_cleaned["Yield_Momentum"] = data_cleaned["Previous_Week_Avg_Yield"] / (data_cleaned["Days_in_Milk"] + 1)
data_cleaned["Age_Parity_Ratio"] = data_cleaned["Age_Months"] / (data_cleaned["Parity"] + 1)
data_cleaned["Month_Sin"] = np.sin(2 * np.pi * data_original["Month"] / 12)
data_cleaned["Month_Cos"] = np.cos(2 * np.pi * data_original["Month"] / 12)
data_cleaned["Weight_Age_Ratio"] = data_cleaned["Weight_kg"] / data_cleaned["Age_Months"]

data_cleaned = data_cleaned.drop(columns=['Date', 'Month'])

print(f"  After engineering: {data_cleaned.shape}")
print(f"  Added 9 new features")

# Extract features and target
X = data_cleaned.drop(columns=['Milk_Yield_L'])
y = data_cleaned['Milk_Yield_L']

# Train/test split
print("\n[3/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {X_train.shape[0]:,} samples")
print(f"  Test:  {X_test.shape[0]:,} samples")

# Prepare for CatBoost
cat_features = ['Farm_ID', 'Breed', 'Climate_Zone', 'Management_System',
                'Lactation_Stage', 'Feed_Type', 'Season']

for col in cat_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

numeric_features = [col for col in X_train.columns if col not in cat_features]

# Scale numeric features
print("\n[4/7] Scaling numeric features...")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print(f"  Categorical features: {len(cat_features)}")
print(f"  Numeric features: {len(numeric_features)}")
print(f"  Total features: {len(X_train_scaled.columns)}")

# Train CatBoost
print("\n[5/7] Training CatBoost with native categorical encoding...")
print("  Hyperparameters:")
print("    - depth: 6")
print("    - learning_rate: 0.05")
print("    - iterations: 500")
print("    - l2_leaf_reg: 10")
print("    - cat_features: 7 categorical columns (native encoding)")

model = CatBoostRegressor(
    cat_features=cat_features,
    depth=6,
    learning_rate=0.05,
    iterations=500,
    l2_leaf_reg=10,
    border_count=128,
    random_state=42,
    verbose=100
)

model.fit(X_train_scaled, y_train)

# Evaluate
print("\n[6/7] Evaluating model...")
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Training RMSE:   {train_rmse:.4f}")
print(f"Validation RMSE: {test_rmse:.4f}")
print(f"Gap (Test-Train): {test_rmse - train_rmse:.4f}")

print("\n" + "="*80)
print("COMPARISON TO BASELINE")
print("="*80)
print("Baseline (no engineered features): 4.1179")
print(f"With engineered features:          {test_rmse:.4f}")
print(f"Improvement:                       {4.1179 - test_rmse:.4f} ({(4.1179 - test_rmse)/4.1179 * 100:.2f}%)")

print("\nSTATUS: This is our BEST model! (Kaggle: 4.16148)")

# Feature importance
print("\n" + "="*80)
print("TOP 20 FEATURE IMPORTANCES")
print("="*80)
importances = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(20).to_string(index=False))

# Save model and preprocessing
print("\n[7/7] Saving model info for Kaggle submission...")

joblib.dump(scaler, 'scaler_features.pkl')
joblib.dump(cat_features, 'cat_features.pkl')
joblib.dump(list(X_train_scaled.columns), 'feature_names.pkl')
joblib.dump(model, 'catboost_model_features.pkl')

print("\n" + "="*80)
print("SAVED FILES")
print("="*80)
print("  - scaler_features.pkl")
print("  - cat_features.pkl")
print("  - feature_names.pkl")
print("  - catboost_model_features.pkl")
print("\nReady to generate Kaggle predictions!")
