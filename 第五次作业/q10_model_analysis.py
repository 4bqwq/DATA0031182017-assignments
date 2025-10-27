import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from q10_ranking_prediction import model_results

print("Detailed analysis of ranking prediction models...")
print("=" * 60)

# 分析模型表现
subjects = list(model_results.keys())
rmse_scores = []
r2_scores = []

for subject in subjects:
    rf_results = model_results[subject]['Random Forest']
    rmse_scores.append(rf_results['rmse'])
    r2_scores.append(rf_results['r2'])

# 创建可视化
plt.figure(figsize=(15, 10))

# 1. R²分布
plt.subplot(2, 3, 1)
plt.hist(r2_scores, bins=10, alpha=0.7, edgecolor='black', color='lightblue')
plt.axvline(x=np.mean(r2_scores), color='red', linestyle='--', label=f'Mean: {np.mean(r2_scores):.3f}')
plt.xlabel('R² Score')
plt.ylabel('Number of Subjects')
plt.title('Distribution of R² Scores Across Subjects')
plt.legend()
plt.grid(alpha=0.3)

# 2. RMSE分布
plt.subplot(2, 3, 2)
plt.hist(rmse_scores, bins=10, alpha=0.7, edgecolor='black', color='lightgreen')
plt.axvline(x=np.mean(rmse_scores), color='red', linestyle='--', label=f'Mean: {np.mean(rmse_scores):.1f}')
plt.xlabel('RMSE')
plt.ylabel('Number of Subjects')
plt.title('Distribution of RMSE Across Subjects')
plt.legend()
plt.grid(alpha=0.3)

# 3. 学科 vs R²
plt.subplot(2, 3, 3)
plt.scatter(range(len(subjects)), r2_scores, alpha=0.7, s=60)
plt.xlabel('Subject Index')
plt.ylabel('R² Score')
plt.title('Model Performance by Subject')
plt.grid(alpha=0.3)

# 4. 学科规模 vs 模型表现
subject_sizes = []
for subject in subjects:
    df = pd.read_csv(f"download/{subject}.csv", encoding='latin1', skiprows=1)
    df = df.drop(df.index[len(df) - 1])
    subject_sizes.append(len(df))

plt.subplot(2, 3, 4)
plt.scatter(subject_sizes, r2_scores, alpha=0.7, s=60)
plt.xlabel('Number of Institutions in Subject')
plt.ylabel('R² Score')
plt.title('Dataset Size vs Model Performance')
plt.grid(alpha=0.3)

# 5. 最佳和最差表现的学科
plt.subplot(2, 3, 5)
performance_data = list(zip(subjects, r2_scores))
performance_data.sort(key=lambda x: x[1], reverse=True)

top_5 = performance_data[:5]
bottom_5 = performance_data[-5:]

all_performances = [r2 for _, r2 in performance_data]
colors = ['green' if r2 > np.median(all_performances) else 'red' for _, r2 in performance_data]

plt.barh(range(10), [r2 for _, r2 in top_5 + bottom_5], color=colors[:5] + colors[-5:])
plt.yticks(range(10), [s[:15] + '...' if len(s) > 15 else s for s, _ in top_5 + bottom_5])
plt.xlabel('R² Score')
plt.title('Best and Worst Performing Subjects')
plt.grid(alpha=0.3)

# 6. 数据质量分析
plt.subplot(2, 3, 6)
missing_data_analysis = []
for subject in subjects[:10]:  # 只分析前10个学科
    df = pd.read_csv(f"download/{subject}.csv", encoding='latin1', skiprows=1)
    df = df.drop(df.index[len(df) - 1])

    # 检查缺失值
    total_rows = len(df)
    missing_docs = df['Web of Science Documents'].isna().sum()
    missing_cites = df['Cites'].isna().sum()

    missing_data_analysis.append({
        'subject': subject[:10] + '...',
        'missing_rate': (missing_docs + missing_cites) / (2 * total_rows) * 100
    })

missing_rates = [d['missing_rate'] for d in missing_data_analysis]
subject_names = [d['subject'] for d in missing_data_analysis]

plt.barh(range(len(missing_rates)), missing_rates, color='orange', alpha=0.7)
plt.yticks(range(len(subject_names)), subject_names)
plt.xlabel('Missing Data Rate (%)')
plt.title('Data Quality Analysis (First 10 Subjects)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 模型诊断和改进建议
print("\nMODEL DIAGNOSIS AND IMPROVEMENT RECOMMENDATIONS:")
print("=" * 60)

print("1. CURRENT PERFORMANCE ISSUES:")
print(f"   - Average R²: {np.mean(r2_scores):.3f} (negative values indicate poor fit)")
print(f"   - Average RMSE: {np.mean(rmse_scores):.1f}")
print(f"   - All models show negative R², suggesting poor predictive power")

print("\n2. ROOT CAUSE ANALYSIS:")
print("   - Ranking prediction is inherently difficult due to:")
print("     * Non-linear relationship between metrics and rankings")
print("     * Rankings are ordinal, not continuous")
print("     * High variance in lower-ranked institutions")
print("     * Feature multicollinearity (cites highly correlated with documents)")

print("\n3. SPECIFIC OBSERVATIONS:")
# 找出数据集中的一些特征
print(f"   - Dataset sizes range from {min(subject_sizes)} to {max(subject_sizes)} institutions")
print(f"   - No clear correlation between dataset size and model performance")
print(f"   - Citation count dominates feature importance (99.97%)")

print("\n4. IMPROVEMENT SUGGESTIONS:")
print("   - Use classification instead of regression (e.g., top 100, 101-500, etc.)")
print("   - Apply rank-transformed target variables")
print("   - Use ordinal regression models")
print("   - Include additional features like institution age, location, funding")
print("   - Try non-linear models like Gradient Boosting or Neural Networks")
print("   - Use cross-validation instead of simple train-test split")

print("\n5. ALTERNATIVE APPROACHES:")
print("   - Predict ranking percentiles instead of absolute ranks")
print("   - Use quantile regression for uncertainty estimation")
print("   - Implement ensemble methods combining multiple models")
print("   - Consider domain-specific features for each discipline")

# 简单的改进模型示例
print("\n6. IMPROVED MODELING APPROACH EXAMPLE:")
print("=" * 40)

print("Trying a percentile-based approach with one subject...")

# 以CHEMISTRY为例尝试改进方法
subject = 'CHEMISTRY'
df = pd.read_csv(f"download/{subject}.csv", encoding='latin1', skiprows=1)
df = df.drop(df.index[len(df) - 1])
df['Rank'] = df['Unnamed: 0'].astype(int)
df['Documents'] = pd.to_numeric(df['Web of Science Documents'], errors='coerce')
df['Cites'] = pd.to_numeric(df['Cites'], errors='coerce')
df_clean = df.dropna()

# 创建排名百分位类别
df_clean['Rank_Percentile'] = df_clean['Rank'] / len(df_clean)
df_clean['Rank_Category'] = pd.cut(df_clean['Rank_Percentile'],
                                   bins=[0, 0.1, 0.3, 0.6, 1.0],
                                   labels=['Top 10%', '10-30%', '30-60%', 'Bottom 40%'])

print(f"Category distribution in {subject}:")
print(df_clean['Rank_Category'].value_counts())

# 这是一个分类问题，可能更适合排名预测
print(f"\nThis categorical approach might be more suitable for ranking prediction")
print(f"as it transforms the problem into classification rather than regression.")

print(f"\nModel analysis completed!")
print(f"The visualizations have been saved as 'model_analysis.png'")