"""
Train CatBoost with engineered features + native categorical encoding
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import time

print("="*80)
print("CATBOOST TRAINING WITH ENGINEERED FEATURES")
print("="*80)

# Load and clean data (same as before)
print("\n[1/7] Loading and cleaning data...")
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

# Fix Breed
data_cleaned['Breed'] = data_cleaned['Breed'].str.strip()
data_cleaned['Breed'] = data_cleaned['Breed'].replace({'Holstien': 'Holstein'})

# Remove negative yields
data_cleaned = data_cleaned[data_cleaned['Milk_Yield_L'] >= 0].copy()

# Impute Feed_Quantity_kg
data_cleaned['Feed_Quantity_kg'] = data_cleaned.groupby('Feed_Type')['Feed_Quantity_kg'].transform(
    lambda x: x.fillna(x.median())
)

print(f"  After cleaning: {data_cleaned.shape}")

# FEATURE ENGINEERING
print("\n[2/7] Engineering features...")

# Reload original for Month
data_original = pd.read_csv("cattle_data_train.csv")
data_original["Date"] = pd.to_datetime(data_original["Date"])
data_original["Month"] = data_original["Date"].dt.month
data_original = data_original[data_original["Milk_Yield_L"] >= 0].copy()

# Create 9 features (drop Milking_Interval_hrs - too low correlation)
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

# Drop Date and Month (we have Season and Month_Sin/Cos)
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

# NATIVE CATBOOST ENCODING (no manual target encoding!)
print("\n[4/7] Preparing for native CatBoost encoding...")

# Categorical columns (keep as strings/objects)
cat_features = ['Farm_ID', 'Breed', 'Climate_Zone', 'Management_System',
                'Lactation_Stage', 'Feed_Type', 'Season']

# Ensure categorical columns are strings
for col in cat_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Get numeric columns for scaling
numeric_features = [col for col in X_train.columns if col not in cat_features]

print(f"  Categorical features: {len(cat_features)}")
print(f"  Numeric features: {len(numeric_features)}")
print(f"  Total features: {len(X_train.columns)}")

# Scale numeric features only
print("\n[5/7] Scaling numeric features...")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Train CatBoost
print("\n[6/7] Training CatBoost with native categorical encoding...")
print("  Hyperparameters:")
print("    - depth: 6")
print("    - learning_rate: 0.05")
print("    - iterations: 500")
print("    - l2_leaf_reg: 10")
print("    - cat_features: 7 categorical columns (native encoding)")

cat_model = CatBoostRegressor(
    cat_features=cat_features,  # Native CatBoost encoding!
    depth=6,
    learning_rate=0.05,
    iterations=500,
    l2_leaf_reg=10,
    border_count=128,
    random_state=42,
    verbose=100  # Show progress every 100 iterations
)

start_time = time.time()
cat_model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

# Evaluate
print("\n[7/7] Evaluating model...")
y_pred_train = cat_model.predict(X_train_scaled)
y_pred_test = cat_model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Training RMSE:   {train_rmse:.4f}")
print(f"Validation RMSE: {test_rmse:.4f}")
print(f"Gap (Test-Train): {test_rmse - train_rmse:.4f}")
print(f"Training time:   {train_time:.1f} seconds ({train_time/60:.1f} minutes)")

# Compare to baseline
baseline_test_rmse = 4.1179  # Original CatBoost without features
improvement = baseline_test_rmse - test_rmse
improvement_pct = (improvement / baseline_test_rmse) * 100

print("\n" + "="*80)
print("COMPARISON TO BASELINE")
print("="*80)
print(f"Baseline (no engineered features): {baseline_test_rmse:.4f}")
print(f"With engineered features:          {test_rmse:.4f}")
print(f"Improvement:                       {improvement:.4f} ({improvement_pct:+.2f}%)")

if test_rmse < baseline_test_rmse:
    print("\nSTATUS: IMPROVEMENT! Engineered features helped!")
else:
    print("\nWARNING: No improvement. May need different features.")

# Feature importance
print("\n" + "="*80)
print("TOP 20 FEATURE IMPORTANCES")
print("="*80)

importances = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(20).to_string(index=False))

# Highlight new features
print("\nNew engineered features in top 20:")
new_feat_names = ["Feed_Efficiency", "Water_Feed_Ratio", "Peak_Lactation", "Heat_Stress",
                  "Yield_Momentum", "Age_Parity_Ratio", "Month_Sin", "Month_Cos", "Weight_Age_Ratio"]
new_in_top20 = importances.head(20)['feature'].isin(new_feat_names).sum()
print(f"  {new_in_top20}/9 new features in top 20")

# Save model info for Kaggle prediction
print("\n" + "="*80)
print("SAVING MODEL INFO FOR KAGGLE SUBMISSION")
print("="*80)

# Save scaler, cat_features list, and feature names
import joblib
joblib.dump(scaler, 'scaler_features.pkl')
joblib.dump(cat_features, 'cat_features.pkl')
joblib.dump(list(X_train_scaled.columns), 'feature_names.pkl')
joblib.dump(cat_model, 'catboost_model_features.pkl')

print("Saved:")
print("  - scaler_features.pkl")
print("  - cat_features.pkl")
print("  - feature_names.pkl")
print("  - catboost_model_features.pkl")

print("\nReady to generate Kaggle predictions!")
