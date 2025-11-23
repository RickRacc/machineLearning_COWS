"""
Test feature engineering and correlations
"""
import pandas as pd
import numpy as np

# Load data
print("Loading data...")
data = pd.read_csv("cattle_data_train.csv")
print(f"Original shape: {data.shape}")

# Feature removal (same as model.ipynb)
features_to_remove = [
    'Feed_Quantity_lb', 'Cattle_ID', 'Rumination_Time_hrs',
    'HS_Vaccine', 'BQ_Vaccine', 'BVD_Vaccine', 'Brucellosis_Vaccine',
    'FMD_Vaccine', 'Resting_Hours', 'Housing_Score', 'Feeding_Frequency',
    'Walking_Distance_km', 'Body_Condition_Score', 'Humidity_percent',
    'Grazing_Duration_hrs', 'Milking_Interval_hrs'
]

data_cleaned = data.drop(columns=features_to_remove)

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

data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])
data_cleaned['Month'] = data_cleaned['Date'].dt.month
data_cleaned['Season'] = data_cleaned['Month'].apply(get_season)
data_cleaned = data_cleaned.drop(columns=['Date', 'Month'])

# Fix Breed
data_cleaned['Breed'] = data_cleaned['Breed'].str.strip()
data_cleaned['Breed'] = data_cleaned['Breed'].replace({'Holstien': 'Holstein'})

# Remove negative yields
data_cleaned = data_cleaned[data_cleaned['Milk_Yield_L'] >= 0].copy()

# Impute Feed_Quantity_kg
data_cleaned['Feed_Quantity_kg'] = data_cleaned.groupby('Feed_Type')['Feed_Quantity_kg'].transform(
    lambda x: x.fillna(x.median())
)

print(f"After cleaning: {data_cleaned.shape}")

# FEATURE ENGINEERING
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Tier 1: High-confidence features
print("\nCreating Tier 1 features...")
data_cleaned["Feed_Efficiency"] = data_cleaned["Feed_Quantity_kg"] / data_cleaned["Weight_kg"]
data_cleaned["Water_Feed_Ratio"] = data_cleaned["Water_Intake_L"] / data_cleaned["Feed_Quantity_kg"]
data_cleaned["Peak_Lactation"] = ((data_cleaned["Days_in_Milk"] >= 60) &
                                   (data_cleaned["Days_in_Milk"] <= 120)).astype(int)

# Heat stress (only positive values matter)
data_cleaned["Heat_Stress"] = ((data_cleaned["Ambient_Temperature_C"] - 25) *
                                (data_cleaned["Season"] == "Summer").astype(int)).clip(lower=0)

data_cleaned["Yield_Momentum"] = data_cleaned["Previous_Week_Avg_Yield"] / (data_cleaned["Days_in_Milk"] + 1)

# Tier 2: Test correlation
print("Creating Tier 2 features...")
data_cleaned["Age_Parity_Ratio"] = data_cleaned["Age_Months"] / (data_cleaned["Parity"] + 1)

# Reload original data for Month and Milking_Interval_hrs
data_original = pd.read_csv("cattle_data_train.csv")
data_original["Date"] = pd.to_datetime(data_original["Date"])
data_original["Month"] = data_original["Date"].dt.month

# Remove negative yield rows from original (to match data_cleaned indices)
data_original = data_original[data_original["Milk_Yield_L"] >= 0].copy()

# Cyclic month encoding
data_cleaned["Month_Sin"] = np.sin(2 * np.pi * data_original["Month"] / 12)
data_cleaned["Month_Cos"] = np.cos(2 * np.pi * data_original["Month"] / 12)
data_cleaned["Weight_Age_Ratio"] = data_cleaned["Weight_kg"] / data_cleaned["Age_Months"]

# Add back Milking_Interval_hrs
data_cleaned["Milking_Interval_hrs"] = data_original["Milking_Interval_hrs"].values

print(f"\nFinal shape: {data_cleaned.shape}")
print(f"Added {data_cleaned.shape[1] - 20} new features")

# Test correlations
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

new_features = ["Feed_Efficiency", "Water_Feed_Ratio", "Peak_Lactation", "Heat_Stress",
                "Yield_Momentum", "Age_Parity_Ratio", "Month_Sin", "Month_Cos",
                "Weight_Age_Ratio", "Milking_Interval_hrs"]

print("\nNew feature correlations with Milk_Yield_L:")
print(f"{'Feature':<30} {'Correlation':>12} {'Decision':>10}")
print("-" * 55)

keep_features = []
for feat in new_features:
    corr = data_cleaned[feat].corr(data_cleaned["Milk_Yield_L"])
    decision = "KEEP" if abs(corr) > 0.05 else "REVIEW"
    if abs(corr) > 0.02:  # Keep if > 0.02 for CatBoost to decide
        keep_features.append(feat)
        decision = "KEEP" if abs(corr) > 0.05 else "KEEP (low)"
    print(f"{feat:<30} {corr:>12.4f} {decision:>10}")

print(f"\nFeatures to keep: {len(keep_features)}/{len(new_features)}")
print(f"Features: {keep_features}")

# Compare to existing strong features
print("\n" + "="*80)
print("COMPARISON TO EXISTING FEATURES")
print("="*80)

existing_strong = ["Age_Months", "Weight_kg", "Parity", "Feed_Quantity_kg",
                   "Water_Intake_L", "Mastitis", "Previous_Week_Avg_Yield"]

print(f"\n{'Feature':<30} {'Correlation':>12}")
print("-" * 45)
for feat in existing_strong:
    corr = data_cleaned[feat].corr(data_cleaned["Milk_Yield_L"])
    print(f"{feat:<30} {corr:>12.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Starting features: 20 (after initial cleaning)")
print(f"New features created: {len(new_features)}")
print(f"New features to keep: {len(keep_features)}")
print(f"Total features: {20 + len(keep_features)}")
print("\nReady to train CatBoost with engineered features!")
