"""
Generate Kaggle submission with engineered features + native CatBoost encoding
"""
import pandas as pd
import numpy as np
import joblib

print("="*80)
print("GENERATING KAGGLE SUBMISSION WITH ENGINEERED FEATURES")
print("="*80)

# Load model and preprocessing objects
print("\n[1/6] Loading model and preprocessing objects...")
scaler = joblib.load('scaler_features.pkl')
cat_features = joblib.load('cat_features.pkl')
feature_names = joblib.load('feature_names.pkl')
cat_model = joblib.load('catboost_model_features.pkl')

print(f"  Loaded CatBoost model")
print(f"  Expected features: {len(feature_names)}")
print(f"  Categorical features: {len(cat_features)}")

# Load Kaggle test set
print("\n[2/6] Loading Kaggle test set...")
kaggle_test = pd.read_csv("cattle_data_test.csv")
cattle_ids = kaggle_test['Cattle_ID'].copy()
print(f"  Test samples: {len(kaggle_test):,}")

# SAME PREPROCESSING AS TRAINING
print("\n[3/6] Applying preprocessing...")

# Remove same features
features_to_remove = [
    'Feed_Quantity_lb', 'Cattle_ID', 'Rumination_Time_hrs',
    'HS_Vaccine', 'BQ_Vaccine', 'BVD_Vaccine', 'Brucellosis_Vaccine',
    'FMD_Vaccine', 'Resting_Hours', 'Housing_Score', 'Feeding_Frequency',
    'Walking_Distance_km', 'Body_Condition_Score', 'Humidity_percent',
    'Grazing_Duration_hrs', 'Milking_Interval_hrs'
]

kaggle_test_cleaned = kaggle_test.drop(columns=features_to_remove)

# Extract Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

kaggle_test_cleaned['Date'] = pd.to_datetime(kaggle_test_cleaned['Date'])
kaggle_test_cleaned['Month'] = kaggle_test_cleaned['Date'].dt.month
kaggle_test_cleaned['Season'] = kaggle_test_cleaned['Month'].apply(get_season)

# Fix Breed
kaggle_test_cleaned['Breed'] = kaggle_test_cleaned['Breed'].str.strip()
kaggle_test_cleaned['Breed'] = kaggle_test_cleaned['Breed'].replace({'Holstien': 'Holstein'})

# Impute Feed_Quantity_kg using TRAINING medians
train_data = pd.read_csv("cattle_data_train.csv")
train_data = train_data[train_data['Milk_Yield_L'] >= 0]  # Remove negative yields like in training
train_feed_medians = train_data.groupby('Feed_Type')['Feed_Quantity_kg'].median()

kaggle_test_cleaned['Feed_Quantity_kg'] = kaggle_test_cleaned.apply(
    lambda row: train_feed_medians[row['Feed_Type']] if pd.isna(row['Feed_Quantity_kg']) else row['Feed_Quantity_kg'],
    axis=1
)

print("  Preprocessing complete")

# FEATURE ENGINEERING (SAME AS TRAINING)
print("\n[4/6] Engineering features...")

# Reload original Kaggle test for Month
kaggle_original = pd.read_csv("cattle_data_test.csv")
kaggle_original["Date"] = pd.to_datetime(kaggle_original["Date"])
kaggle_original["Month"] = kaggle_original["Date"].dt.month

# Create 9 engineered features
kaggle_test_cleaned["Feed_Efficiency"] = kaggle_test_cleaned["Feed_Quantity_kg"] / kaggle_test_cleaned["Weight_kg"]
kaggle_test_cleaned["Water_Feed_Ratio"] = kaggle_test_cleaned["Water_Intake_L"] / kaggle_test_cleaned["Feed_Quantity_kg"]
kaggle_test_cleaned["Peak_Lactation"] = ((kaggle_test_cleaned["Days_in_Milk"] >= 60) &
                                          (kaggle_test_cleaned["Days_in_Milk"] <= 120)).astype(int)
kaggle_test_cleaned["Heat_Stress"] = ((kaggle_test_cleaned["Ambient_Temperature_C"] - 25) *
                                       (kaggle_test_cleaned["Season"] == "Summer").astype(int)).clip(lower=0)
kaggle_test_cleaned["Yield_Momentum"] = kaggle_test_cleaned["Previous_Week_Avg_Yield"] / (kaggle_test_cleaned["Days_in_Milk"] + 1)
kaggle_test_cleaned["Age_Parity_Ratio"] = kaggle_test_cleaned["Age_Months"] / (kaggle_test_cleaned["Parity"] + 1)
kaggle_test_cleaned["Month_Sin"] = np.sin(2 * np.pi * kaggle_original["Month"] / 12)
kaggle_test_cleaned["Month_Cos"] = np.cos(2 * np.pi * kaggle_original["Month"] / 12)
kaggle_test_cleaned["Weight_Age_Ratio"] = kaggle_test_cleaned["Weight_kg"] / kaggle_test_cleaned["Age_Months"]

# Drop Date and Month
kaggle_test_cleaned = kaggle_test_cleaned.drop(columns=['Date', 'Month'])

print(f"  Features after engineering: {kaggle_test_cleaned.shape[1]}")

# Ensure categorical columns are strings
for col in cat_features:
    kaggle_test_cleaned[col] = kaggle_test_cleaned[col].astype(str)

# Get numeric features
numeric_features = [col for col in kaggle_test_cleaned.columns if col not in cat_features]

# Scale numeric features using TRAINING scaler
kaggle_test_scaled = kaggle_test_cleaned.copy()
kaggle_test_scaled[numeric_features] = scaler.transform(kaggle_test_cleaned[numeric_features])

# Reorder columns to match training
kaggle_test_final = kaggle_test_scaled[feature_names]

print(f"  Final shape: {kaggle_test_final.shape}")
print(f"  Matches training features: {list(kaggle_test_final.columns) == feature_names}")

# Generate predictions
print("\n[5/6] Generating predictions...")
predictions = cat_model.predict(kaggle_test_final)

print(f"  Prediction statistics:")
print(f"    Mean:   {predictions.mean():.3f} L")
print(f"    Std:    {predictions.std():.3f} L")
print(f"    Min:    {predictions.min():.3f} L")
print(f"    Max:    {predictions.max():.3f} L")
print(f"    Median: {np.median(predictions):.3f} L")

# Create submission
submission = pd.DataFrame({
    'Cattle_ID': cattle_ids,
    'Milk_Yield_L': predictions
})

# Save
submission_filename = 'kaggle_submission_features.csv'
submission.to_csv(submission_filename, index=False)

print("\n[6/6] Submission created!")
print("="*80)
print(f"FILE: {submission_filename}")
print("="*80)
print(f"  Rows: {len(submission):,}")
print(f"  Columns: {list(submission.columns)}")
print(f"\nFirst 5 predictions:")
print(submission.head().to_string(index=False))
print(f"\nLast 5 predictions:")
print(submission.tail().to_string(index=False))

print("\n" + "="*80)
print("READY TO SUBMIT TO KAGGLE!")
print("="*80)
print(f"Submit: {submission_filename}")
