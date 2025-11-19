"""
Feature Analysis for Cattle Milk Yield Prediction
==================================================
This script analyzes correlations between features and the target variable (Milk_Yield_L)
to help identify which features are most predictive and which can be removed.

Author: Machine Learning Team
Purpose: Feature selection and data preprocessing analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
print("Loading cattle_data_train.csv...")
data = pd.read_csv('cattle_data_train.csv')
print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns\n")

# ============================================================================
# SECTION 1: CORRELATION ANALYSIS
# ============================================================================
print("=" * 80)
print("SECTION 1: CORRELATION WITH TARGET (Milk_Yield_L)")
print("=" * 80)

# Select only numeric columns for correlation analysis
numeric_data = data.select_dtypes(include=[np.number])
correlations = numeric_data.corr()['Milk_Yield_L'].abs().sort_values(ascending=False)

print("\nSTRONG PREDICTORS (correlation > 0.15):")
print("-" * 80)
strong = correlations[correlations > 0.15]
for feat, corr in strong.items():
    if feat != 'Milk_Yield_L':
        print(f"   {feat:35s} {corr:.6f}")

print("\nMODERATE PREDICTORS (0.05 < correlation <= 0.15):")
print("-" * 80)
moderate = correlations[(correlations > 0.05) & (correlations <= 0.15)]
for feat, corr in moderate.items():
    print(f"   {feat:35s} {corr:.6f}")

print("\nWEAK PREDICTORS (correlation <= 0.05):")
print("-" * 80)
weak = correlations[correlations <= 0.05]
for feat, corr in weak.items():
    if feat != 'Milk_Yield_L':
        print(f"   {feat:35s} {corr:.6f}")

# ============================================================================
# SECTION 2: REAL DATA EXAMPLES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: REAL EXAMPLES FROM THE DATA")
print("=" * 80)

# Example 1: Strong correlation - Age
print("\nExample 1: Age_Months (Strong Predictor, correlation = {:.4f})".format(correlations['Age_Months']))
print("-" * 80)
age_bins = [0, 50, 100, 150]
age_labels = ['Young (24-50)', 'Middle (51-100)', 'Old (101-143)']
data['Age_Group'] = pd.cut(data['Age_Months'], bins=age_bins, labels=age_labels)
avg_yield_by_age = data.groupby('Age_Group', observed=True)['Milk_Yield_L'].agg(['mean', 'std', 'count'])
print(avg_yield_by_age)
print("\nInterpretation: Age has a clear relationship with milk yield!")

# Example 2: Strong correlation - Weight
print("\nExample 2: Weight_kg (Strong Predictor, correlation = {:.4f})".format(correlations['Weight_kg']))
print("-" * 80)
weight_bins = [0, 400, 550, 800]
weight_labels = ['Light (<400kg)', 'Medium (400-550kg)', 'Heavy (>550kg)']
data['Weight_Group'] = pd.cut(data['Weight_kg'], bins=weight_bins, labels=weight_labels)
avg_yield_by_weight = data.groupby('Weight_Group', observed=True)['Milk_Yield_L'].agg(['mean', 'std', 'count'])
print(avg_yield_by_weight)
print("\nInterpretation: Heavier cows tend to produce more milk!")

# Example 3: Low correlation - Body Condition Score
print("\nExample 3: Body_Condition_Score (Weak Predictor, correlation = {:.4f})".format(correlations['Body_Condition_Score']))
print("-" * 80)
bcs_groups = data.groupby('Body_Condition_Score')['Milk_Yield_L'].agg(['mean', 'std', 'count'])
print(bcs_groups)
print("\nInterpretation: All body condition scores give similar milk yields!")
print("                This feature doesn't help predict milk yield.")

# Example 4: Near-zero correlation - HS_Vaccine
print("\nExample 4: HS_Vaccine (Near-Zero Predictor, correlation = {:.6f})".format(correlations['HS_Vaccine']))
print("-" * 80)
hs_groups = data.groupby('HS_Vaccine')['Milk_Yield_L'].agg(['mean', 'std', 'count'])
print(hs_groups)
diff = abs(hs_groups.loc[1, 'mean'] - hs_groups.loc[0, 'mean'])
print(f"\nDifference in means: {diff:.6f} L (essentially zero!)")
print("Interpretation: HS vaccine status has NO impact on milk yield prediction.")

# ============================================================================
# SECTION 3: DATA QUALITY ISSUES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: DATA QUALITY ISSUES")
print("=" * 80)

# Check for missing values
print("\nMissing Values:")
print("-" * 80)
missing = data.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    for col, count in missing.items():
        pct = (count / len(data)) * 100
        print(f"   {col:35s} {count:6d} ({pct:.2f}%)")
else:
    print("   No missing values found!")

# Check Rumination_Time_hrs for negative values
print("\nRumination_Time_hrs Analysis:")
print("-" * 80)
rumination = data['Rumination_Time_hrs']
neg_count = (rumination < 0).sum()
neg_pct = (neg_count / len(data)) * 100
print(f"   Total values:     {len(data):,}")
print(f"   Negative values:  {neg_count:,} ({neg_pct:.1f}%)")
print(f"   Min value:        {rumination.min():.2f} hours")
print(f"   Max value:        {rumination.max():.2f} hours")
print(f"   Mean value:       {rumination.mean():.2f} hours")
print(f"\n   WARNING: {neg_pct:.1f}% of rumination time values are NEGATIVE!")
print("   This is physically impossible - rumination time cannot be negative.")
print("   RECOMMENDATION: Remove this feature due to data quality issues.")

# Check Feed_Quantity_kg vs Feed_Quantity_lb
print("\nFeed_Quantity_kg vs Feed_Quantity_lb Analysis:")
print("-" * 80)
valid_feed = data[['Feed_Quantity_kg', 'Feed_Quantity_lb']].dropna()
feed_corr = valid_feed.corr().iloc[0, 1]
print(f"   Correlation: {feed_corr:.6f}")
valid_feed['ratio'] = valid_feed['Feed_Quantity_lb'] / valid_feed['Feed_Quantity_kg']
print(f"   Mean ratio (lb/kg): {valid_feed['ratio'].mean():.4f}")
print(f"   Expected ratio:     2.2046 (conversion factor)")
print(f"\n   These columns are essentially duplicates!")
print("   RECOMMENDATION: Keep Feed_Quantity_kg, remove Feed_Quantity_lb")

# ============================================================================
# SECTION 4: VACCINE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: VACCINE FEATURE ANALYSIS")
print("=" * 80)

vaccine_cols = ['FMD_Vaccine', 'Brucellosis_Vaccine', 'HS_Vaccine', 'BQ_Vaccine',
                'Anthrax_Vaccine', 'IBR_Vaccine', 'BVD_Vaccine', 'Rabies_Vaccine']

print("\nVaccine Distribution:")
print("-" * 80)
for vaccine in vaccine_cols:
    count = data[vaccine].sum()
    pct = (count / len(data)) * 100
    corr = correlations[vaccine] if vaccine in correlations.index else 0
    print(f"   {vaccine:25s} {count:6d} ({pct:5.1f}%)  |  Correlation: {corr:.6f}")

print("\nVaccine Correlation Summary:")
print("-" * 80)
vaccine_corrs = correlations[vaccine_cols].sort_values(ascending=False)
for vaccine, corr in vaccine_corrs.items():
    status = "KEEP" if corr > 0.05 else "REMOVE"
    print(f"   {vaccine:25s} {corr:.6f}  [{status}]")

# ============================================================================
# SECTION 5: RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: FEATURE REMOVAL RECOMMENDATIONS")
print("=" * 80)

features_to_remove = [
    ('Feed_Quantity_lb', 'Duplicate of Feed_Quantity_kg (99.99% correlation)'),
    ('Cattle_ID', 'Unique identifier, no predictive value'),
    ('Date', 'Raw temporal data, adds noise without feature engineering'),
    ('Rumination_Time_hrs', 'Data quality issue: 55% negative values'),
    ('HS_Vaccine', 'Near-zero correlation: 0.000034'),
    ('BQ_Vaccine', 'Near-zero correlation: 0.000466'),
    ('BVD_Vaccine', 'Near-zero correlation: 0.000491'),
    ('Brucellosis_Vaccine', 'Very low correlation: 0.002089'),
    ('FMD_Vaccine', 'Very low correlation: 0.002477'),
    ('Resting_Hours', 'Near-zero correlation: 0.001653'),
    ('Housing_Score', 'Low correlation: 0.004008 + 3% missing values'),
    ('Feeding_Frequency', 'Near-zero correlation: 0.000380'),
    ('Walking_Distance_km', 'Near-zero correlation: 0.001538'),
    ('Body_Condition_Score', 'Near-zero correlation: 0.001647'),
    ('Humidity_percent', 'Very low correlation: 0.002153'),
]

print("\nFeatures to REMOVE (15 total):")
print("-" * 80)
for i, (feature, reason) in enumerate(features_to_remove, 1):
    print(f"{i:2d}. {feature:30s} - {reason}")

print("\n\nFeatures to KEEP (High/Moderate Predictive Value):")
print("-" * 80)
keep_features = []
for feat, corr in correlations.items():
    if feat != 'Milk_Yield_L' and feat not in [f[0] for f in features_to_remove]:
        keep_features.append((feat, corr))

keep_features.sort(key=lambda x: x[1], reverse=True)
for feat, corr in keep_features:
    print(f"   {feat:35s} (correlation: {corr:.6f})")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Original features:     {data.shape[1] - 1} (excluding target)")
print(f"Features to remove:    {len(features_to_remove)}")
print(f"Features to keep:      {len(keep_features)}")
print(f"Reduction:             {len(features_to_remove) / (data.shape[1] - 1) * 100:.1f}%")
print("=" * 80)

# ============================================================================
# SECTION 6: CREATE VISUALIZATION
# ============================================================================
print("\nGenerating correlation heatmap...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Analysis for Cattle Milk Yield Prediction', fontsize=16, fontweight='bold')

# Plot 1: Correlation bar chart
ax1 = axes[0, 0]
top_features = correlations.drop('Milk_Yield_L').head(15)
colors = ['green' if x > 0.15 else 'orange' if x > 0.05 else 'red' for x in top_features.values]
top_features.plot(kind='barh', ax=ax1, color=colors)
ax1.set_xlabel('Absolute Correlation with Milk_Yield_L')
ax1.set_title('Top 15 Features by Correlation')
ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Weak threshold')
ax1.axvline(x=0.15, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong threshold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Age vs Milk Yield
ax2 = axes[0, 1]
age_scatter = data.sample(min(5000, len(data)), random_state=42)
ax2.scatter(age_scatter['Age_Months'], age_scatter['Milk_Yield_L'], alpha=0.3, s=10)
ax2.set_xlabel('Age (Months)')
ax2.set_ylabel('Milk Yield (L)')
ax2.set_title(f'Age vs Milk Yield (Correlation: {correlations["Age_Months"]:.3f})')
ax2.grid(alpha=0.3)

# Plot 3: Weight vs Milk Yield
ax3 = axes[1, 0]
weight_scatter = data.sample(min(5000, len(data)), random_state=42)
ax3.scatter(weight_scatter['Weight_kg'], weight_scatter['Milk_Yield_L'], alpha=0.3, s=10, color='green')
ax3.set_xlabel('Weight (kg)')
ax3.set_ylabel('Milk Yield (L)')
ax3.set_title(f'Weight vs Milk Yield (Correlation: {correlations["Weight_kg"]:.3f})')
ax3.grid(alpha=0.3)

# Plot 4: Body Condition Score vs Milk Yield (showing no relationship)
ax4 = axes[1, 1]
bcs_data = data.groupby('Body_Condition_Score')['Milk_Yield_L'].mean()
ax4.bar(bcs_data.index, bcs_data.values, color='red', alpha=0.6)
ax4.set_xlabel('Body Condition Score')
ax4.set_ylabel('Average Milk Yield (L)')
ax4.set_title(f'Body Condition Score vs Milk Yield (Correlation: {correlations["Body_Condition_Score"]:.6f})')
ax4.axhline(y=data['Milk_Yield_L'].mean(), color='black', linestyle='--',
            linewidth=2, label='Overall Mean', alpha=0.7)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'feature_correlation_analysis.png'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nNext Steps:")
print("1. Review the recommendations above")
print("2. Check the generated visualization: feature_correlation_analysis.png")
print("3. Implement feature removal in your model pipeline")
print("4. Compare model performance before and after feature removal")
print("=" * 80)
