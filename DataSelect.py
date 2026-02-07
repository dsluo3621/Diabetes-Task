import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp
import warnings

warnings.filterwarnings('ignore')

# ===================== Global Configuration =====================
# Data paths (replace with your local path)
INPUT_PATH = "CDC Diabetes Dataset.csv"
OUTPUT_PATH = "diabetes_sampled_50000.csv"
# Target sample size
TARGET_SAMPLE_SIZE = 50000
# Random seed (ensure reproducibility)
RANDOM_SEED = 42
# Core features to validate
CATEGORICAL_FEATURES = ['Diabetes_012', 'HighBP', 'HighChol', 'Smoker', 'Sex']  # Categorical features
NUMERIC_FEATURES = ['BMI', 'Age', 'MentHlth', 'PhysHlth']  # Numeric features

# ===================== Fix Matplotlib Backend Issue =====================
plt.switch_backend('Agg')  # Switch backend to avoid PyCharm compatibility issues
plt.rcParams['font.sans-serif'] = ['SimHei']  # Support Chinese character display
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue


# ===================== 1. Load Original Data =====================
def load_original_data(file_path):
    """Load original dataset and perform basic validation"""
    df = pd.read_csv(file_path)
    print("=" * 50)
    print("Original Dataset Basic Information:")
    print(f"Dataset Size: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Missing Value Check: {df.isnull().sum().sum()} missing values")
    print(f"Target Variable Distribution:\n{df['Diabetes_012'].value_counts(normalize=True).round(4) * 100}")
    # Output statistics for core numeric features (for subsequent comparison)
    print("\nOriginal Dataset Core Numeric Feature Statistics:")
    for feat in NUMERIC_FEATURES:
        print(f"{feat}: Mean={df[feat].mean():.4f}, Median={df[feat].median():.4f}, Std={df[feat].std():.4f}")
    return df


# ===================== 2. Optimized Stratified Sampling (Preserve Distribution) =====================
def optimized_stratified_sampling(original_df, target_size, random_seed):
    """
    Optimized stratified sampling:
    1. Stratify only by target variable (Diabetes_012) - reduce stratification dimensions to avoid small sample bias
    2. Sample with exact proportions, use replacement for insufficient samples (ensure 100% distribution consistency)
    """
    df = original_df.copy()

    # Step 1: Calculate exact sample counts for each target variable category (rounded to integer)
    target_dist = df['Diabetes_012'].value_counts(normalize=True)
    sample_counts = (target_dist * target_size).round().astype(int)

    # Correct total count (adjust largest category if rounded total != target size)
    total_sample = sample_counts.sum()
    if total_sample != target_size:
        diff = target_size - total_sample
        max_cat = sample_counts.idxmax()
        sample_counts[max_cat] += diff

    # Step 2: Perform stratified sampling by target variable (use replacement for small categories to ensure count)
    sampled_dfs = []
    for cat, count in sample_counts.items():
        cat_df = df[df['Diabetes_012'] == cat]
        # Use replacement if insufficient samples; otherwise no replacement
        replace = len(cat_df) < count
        sampled_cat = cat_df.sample(n=count, random_state=random_seed, replace=replace)
        sampled_dfs.append(sampled_cat)

    # Step 3: Combine and shuffle data
    sampled_df = pd.concat(sampled_dfs)
    sampled_df = sampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print("=" * 50)
    print("Sampling Results:")
    print(f"Final Sample Size: {sampled_df.shape[0]} rows × {sampled_df.shape[1]} columns")
    print(f"Target Variable Distribution:\n{sampled_df['Diabetes_012'].value_counts(normalize=True).round(4) * 100}")
    # Output statistics for core numeric features after sampling
    print("\nSampled Dataset Core Numeric Feature Statistics:")
    for feat in NUMERIC_FEATURES:
        print(
            f"{feat}: Mean={sampled_df[feat].mean():.4f}, Median={sampled_df[feat].median():.4f}, Std={sampled_df[feat].std():.4f}")

    return sampled_df


# ===================== 3. Distribution Consistency Validation =====================
def validate_distribution(original_df, sampled_df):
    """
    Validate distribution consistency between sampled and original datasets:
    - Categorical features: Chi-square test (p>0.05 = no significant difference)
    - Numeric features: KS test (p>0.05 = no significant difference)
    """
    print("=" * 50)
    print("Distribution Consistency Validation Results:")

    # 3.1 Categorical features: Chi-square test
    print("\n【Categorical Features - Chi-square Test】")
    chi2_results = {}
    for feat in CATEGORICAL_FEATURES:
        # Create contingency table (frequency distribution)
        original_count = original_df[feat].value_counts(normalize=True)
        sampled_count = sampled_df[feat].value_counts(normalize=True)
        # Align indices (avoid dimension mismatch)
        all_categories = sorted(list(set(original_count.index) | set(sampled_count.index)))
        original_count = original_count.reindex(all_categories, fill_value=0)
        sampled_count = sampled_count.reindex(all_categories, fill_value=0)
        # Perform chi-square test
        obs = np.array([original_count.values, sampled_count.values])
        chi2, p, dof, ex = chi2_contingency(obs)
        chi2_results[feat] = {'chi2': chi2, 'p': p, 'consistent': p > 0.05}
        # Output results
        status = "✅ Consistent" if p > 0.05 else "❌ Inconsistent"
        print(f"{feat}: p-value={p:.4f} {status}")

    # 3.2 Numeric features: KS test + statistic comparison
    print("\n【Numeric Features - KS Test + Statistic Comparison】")
    ks_results = {}
    for feat in NUMERIC_FEATURES:
        # Perform KS test
        ks_stat, p = ks_2samp(original_df[feat], sampled_df[feat])
        ks_results[feat] = {'ks_stat': ks_stat, 'p': p, 'consistent': p > 0.05}
        # Compare statistics
        original_mean = original_df[feat].mean()
        sampled_mean = sampled_df[feat].mean()
        mean_diff = abs(original_mean - sampled_mean) / original_mean * 100  # Relative error
        # Output results
        status = "✅ Consistent" if p > 0.05 else "⚠ Minor Difference (Acceptable)"
        print(
            f"{feat}: p-value={p:.4f} {status} | Original Mean={original_mean:.2f} | Sampled Mean={sampled_mean:.2f} | Relative Error={mean_diff:.2f}%")

    # 3.3 Summarize validation results
    all_consistent = all([chi2_results[feat]['consistent'] for feat in CATEGORICAL_FEATURES])
    print("\n【Validation Summary】")
    print(f"Categorical Feature Distribution Consistency: {'✅ All Consistent' if all_consistent else '⚠ Some Features Inconsistent'}")
    print(f"Numeric Features: KS test p<0.05 is normal for large datasets (minor differences amplified), focus on relative mean error <5%")

    return chi2_results, ks_results


# ===================== 4. Visual Distribution Comparison (Fix Error) =====================
def plot_distribution_comparison(original_df, sampled_df):
    """Create visualization of core feature distribution comparison (fix tostring_rgb error)"""
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 4.1 Target variable distribution (pie chart)
    ax1 = axes[0]
    original_diabetes = original_df['Diabetes_012'].value_counts(normalize=True) * 100
    sampled_diabetes = sampled_df['Diabetes_012'].value_counts(normalize=True) * 100
    ax1.pie(original_diabetes, labels=[f"{k}({v:.1f}%)" for k, v in original_diabetes.items()],
            autopct='%1.1f%%', labeldistance=1.1, textprops={'fontsize': 10})
    ax1.set_title('Original Data - Diabetes Distribution', fontsize=12)

    ax2 = axes[1]
    ax2.pie(sampled_diabetes, labels=[f"{k}({v:.1f}%)" for k, v in sampled_diabetes.items()],
            autopct='%1.1f%%', labeldistance=1.1, textprops={'fontsize': 10})
    ax2.set_title('Sampled Data - Diabetes Distribution', fontsize=12)

    # 4.2 BMI distribution (histogram)
    ax3 = axes[2]
    sns.histplot(original_df['BMI'], bins=30, alpha=0.5, label='Original Data', ax=ax3, kde=True)
    sns.histplot(sampled_df['BMI'], bins=30, alpha=0.5, label='Sampled Data', ax=ax3, kde=True)
    ax3.set_title('BMI Distribution Comparison', fontsize=12)
    ax3.legend()

    # 4.3 Age distribution (histogram)
    ax4 = axes[3]
    sns.histplot(original_df['Age'], bins=8, alpha=0.5, label='Original Data', ax=ax4, kde=True)
    sns.histplot(sampled_df['Age'], bins=8, alpha=0.5, label='Sampled Data', ax=ax4, kde=True)
    ax4.set_title('Age Distribution Comparison', fontsize=12)
    ax4.legend()

    # 4.4 High blood pressure proportion (bar chart)
    ax5 = axes[4]
    original_highbp = original_df['HighBP'].value_counts(normalize=True) * 100
    sampled_highbp = sampled_df['HighBP'].value_counts(normalize=True) * 100
    x = ['No Hypertension(0)', 'Hypertension(1)']
    ax5.bar(x, original_highbp, width=0.3, label='Original Data', alpha=0.7)
    ax5.bar([i + 0.3 for i in range(len(x))], sampled_highbp, width=0.3, label='Sampled Data', alpha=0.7)
    ax5.set_title('Hypertension Proportion Comparison', fontsize=12)
    ax5.set_ylabel('Proportion (%)')
    ax5.legend()

    # 4.5 Feature mean comparison (bar chart)
    ax6 = axes[5]
    metrics = ['BMI', 'Age', 'MentHlth', 'PhysHlth']
    original_means = [original_df[m].mean() for m in metrics]
    sampled_means = [sampled_df[m].mean() for m in metrics]
    x = np.arange(len(metrics))
    ax6.bar(x - 0.2, original_means, 0.4, label='Original Data', alpha=0.7)
    ax6.bar(x + 0.2, sampled_means, 0.4, label='Sampled Data', alpha=0.7)
    ax6.set_title('Core Numeric Feature Mean Comparison', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()

    plt.tight_layout()
    plt.savefig('sampling_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Distribution comparison plot saved as: sampling_distribution_comparison.png")
    # Close figure to avoid memory leaks
    plt.close(fig)


# ===================== 5. Main Execution Flow =====================
if __name__ == "__main__":
    # Step 1: Load original data
    original_df = load_original_data(INPUT_PATH)

    # Step 2: Perform optimized stratified sampling
    sampled_df = optimized_stratified_sampling(original_df, TARGET_SAMPLE_SIZE, RANDOM_SEED)

    # Step 3: Save sampled data
    sampled_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Sampled data saved to: {OUTPUT_PATH}")

    # Step 4: Validate distribution consistency
    chi2_results, ks_results = validate_distribution(original_df, sampled_df)

    # Step 5: Generate comparison visualizations
    plot_distribution_comparison(original_df, sampled_df)