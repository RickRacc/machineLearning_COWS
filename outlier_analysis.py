import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load cleaned data (after feature removal and data quality fixes)
print("Loading data...")
data = pd.read_csv("cattle_data_train.csv")

# Apply same cleaning steps as in notebook
features_to_remove = [
    'Feed_Quantity_lb', 'Cattle_ID', 'Rumination_Time_hrs',
    'HS_Vaccine', 'BQ_Vaccine', 'BVD_Vaccine', 'Brucellosis_Vaccine', 'FMD_Vaccine',
    'Resting_Hours', 'Housing_Score', 'Feeding_Frequency', 'Walking_Distance_km',
    'Body_Condition_Score', 'Humidity_percent', 'Grazing_Duration_hrs', 'Milking_Interval_hrs'
]
data_cleaned = data.drop(columns=features_to_remove)

# Date -> Season
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])
data_cleaned['Month'] = data_cleaned['Date'].dt.month
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'
data_cleaned['Season'] = data_cleaned['Month'].apply(get_season)
data_cleaned = data_cleaned.drop(columns=['Date', 'Month'])

# Breed cleaning
data_cleaned['Breed'] = data_cleaned['Breed'].str.strip()
data_cleaned['Breed'] = data_cleaned['Breed'].replace({'Holstien': 'Holstein'})

# Remove negative yields
data_cleaned = data_cleaned[data_cleaned['Milk_Yield_L'] >= 0].copy()

# Impute Feed_Quantity_kg
data_cleaned['Feed_Quantity_kg'] = data_cleaned.groupby('Feed_Type')['Feed_Quantity_kg'].transform(
    lambda x: x.fillna(x.median())
)

# Get numeric columns only (excluding target)
numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Milk_Yield_L')  # Analyze target separately

print(f"\nAnalyzing {len(numeric_cols)} numeric features for outliers...")
print("="*80)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    """Detect outliers using Z-score method (|z| > threshold)"""
    mean = data[column].mean()
    std = data[column].std()
    z_scores = np.abs((data[column] - mean) / std)
    outliers = data[z_scores > threshold]
    return outliers

# Analyze each numeric feature
outlier_summary = []

for col in numeric_cols:
    print(f"\n{col}")
    print("-" * 60)

    # Basic statistics
    print(f"  Count: {data_cleaned[col].count():,}")
    print(f"  Mean:  {data_cleaned[col].mean():.2f}")
    print(f"  Std:   {data_cleaned[col].std():.2f}")
    print(f"  Min:   {data_cleaned[col].min():.2f}")
    print(f"  Q1:    {data_cleaned[col].quantile(0.25):.2f}")
    print(f"  Median:{data_cleaned[col].median():.2f}")
    print(f"  Q3:    {data_cleaned[col].quantile(0.75):.2f}")
    print(f"  Max:   {data_cleaned[col].max():.2f}")

    # IQR outliers
    iqr_outliers, lower, upper = detect_outliers_iqr(data_cleaned, col)
    iqr_pct = len(iqr_outliers) / len(data_cleaned) * 100

    print(f"\n  IQR Method (1.5×IQR):")
    print(f"    Bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"    Outliers: {len(iqr_outliers):,} ({iqr_pct:.2f}%)")

    # Z-score outliers
    zscore_outliers = detect_outliers_zscore(data_cleaned, col, threshold=3)
    zscore_pct = len(zscore_outliers) / len(data_cleaned) * 100

    print(f"  Z-Score Method (|z| > 3):")
    print(f"    Outliers: {len(zscore_outliers):,} ({zscore_pct:.2f}%)")

    outlier_summary.append({
        'Feature': col,
        'IQR_Outliers': len(iqr_outliers),
        'IQR_Pct': iqr_pct,
        'ZScore_Outliers': len(zscore_outliers),
        'ZScore_Pct': zscore_pct,
        'Range': f"[{data_cleaned[col].min():.2f}, {data_cleaned[col].max():.2f}]"
    })

# Analyze target variable separately
print("\n" + "="*80)
print("TARGET VARIABLE: Milk_Yield_L")
print("="*80)
target_iqr_outliers, target_lower, target_upper = detect_outliers_iqr(data_cleaned, 'Milk_Yield_L')
target_zscore_outliers = detect_outliers_zscore(data_cleaned, 'Milk_Yield_L', threshold=3)

print(f"  Mean:  {data_cleaned['Milk_Yield_L'].mean():.2f} L")
print(f"  Std:   {data_cleaned['Milk_Yield_L'].std():.2f} L")
print(f"  Range: [{data_cleaned['Milk_Yield_L'].min():.2f}, {data_cleaned['Milk_Yield_L'].max():.2f}] L")
print(f"\n  IQR Outliers: {len(target_iqr_outliers):,} ({len(target_iqr_outliers)/len(data_cleaned)*100:.2f}%)")
print(f"  Z-Score Outliers: {len(target_zscore_outliers):,} ({len(target_zscore_outliers)/len(data_cleaned)*100:.2f}%)")

# Summary table
print("\n" + "="*80)
print("OUTLIER SUMMARY (sorted by IQR percentage)")
print("="*80)
summary_df = pd.DataFrame(outlier_summary).sort_values('IQR_Pct', ascending=False)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nFeatures with >5% outliers (IQR method) might need attention:")
high_outlier_features = summary_df[summary_df['IQR_Pct'] > 5]
if len(high_outlier_features) > 0:
    for _, row in high_outlier_features.iterrows():
        print(f"  - {row['Feature']}: {row['IQR_Pct']:.2f}% outliers")
else:
    print("  No features have >5% outliers!")

print("\nOptions for handling outliers:")
print("  1. KEEP THEM - If they're legitimate values (recommended for tree-based models)")
print("  2. CAP/WINSORIZE - Replace extreme values with percentile bounds (e.g., 1st/99th)")
print("  3. REMOVE - Delete outlier rows (risky - lose data)")
print("  4. LOG TRANSFORM - For right-skewed features (e.g., if max >> median)")
print("\nFor cattle data, outliers are often REAL (e.g., high-producing cow, old cow)")
print("Recommendation: KEEP outliers - they contain valuable information!")
