import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from q9_ecnu_analysis import ecnu_df

# 创建可视化图表
plt.figure(figsize=(20, 15))

# 1. 学科排名百分位图
plt.subplot(2, 3, 1)
# 按排名百分位排序
ecnu_sorted = ecnu_df.sort_values('Rank_Percent')
plt.barh(range(len(ecnu_sorted)), ecnu_sorted['Rank_Percent'] * 100)
plt.yticks(range(len(ecnu_sorted)), ecnu_sorted['Subject'])
plt.xlabel('Rank Percentile (%)')
plt.title('ECNU Subject Ranking Position')
plt.grid(axis='x', alpha=0.3)

# 2. 发文量vs引用数散点图
plt.subplot(2, 3, 2)
plt.scatter(ecnu_df['Documents'], ecnu_df['Cites'], alpha=0.6)
for i, subject in enumerate(ecnu_df['Subject']):
    if ecnu_df['Rank_Percent'].iloc[i] < 0.15:  # 标注前15%的学科
        plt.annotate(subject.split()[0], (ecnu_df['Documents'].iloc[i], ecnu_df['Cites'].iloc[i]))
plt.xlabel('Publication Count')
plt.ylabel('Total Citations')
plt.title('Publications vs. Citations')
plt.grid(alpha=0.3)

# 3. 篇均引用分布
plt.subplot(2, 3, 3)
plt.hist(ecnu_df['Cites_Per_Paper'], bins=8, alpha=0.7, edgecolor='black')
plt.xlabel('Citations per Publication')
plt.ylabel('Number of Subjects')
plt.title('Distribution of Citations per Publication')
plt.grid(alpha=0.3)

# 4. 学科分类（按排名表现）
plt.subplot(2, 3, 4)
# 分为三类：强势(前20%)、中等(20%-50%)、弱势(50%以后)
strong = ecnu_df[ecnu_df['Rank_Percent'] < 0.2]
medium = ecnu_df[(ecnu_df['Rank_Percent'] >= 0.2) & (ecnu_df['Rank_Percent'] < 0.5)]
weak = ecnu_df[ecnu_df['Rank_Percent'] >= 0.5]

categories = ['Strong (Top 20%)', 'Medium (20%-50%)', 'Weak (50%+)']
counts = [len(strong), len(medium), len(weak)]
colors = ['green', 'orange', 'red']

plt.pie(counts, labels=categories, colors=colors, autopct='%d', startangle=90)
plt.title('ECNU Subject Strength Classification')

# 5. 高被引论文数量
plt.subplot(2, 3, 5)
top_subjects = ecnu_df.nlargest(8, 'Top_Papers')
plt.bar(range(len(top_subjects)), top_subjects['Top_Papers'])
plt.xticks(range(len(top_subjects)), [s.split()[0] for s in top_subjects['Subject']], rotation=45)
plt.ylabel('Highly Cited Papers')
plt.title('Subjects with Most Highly Cited Papers')
plt.grid(axis='y', alpha=0.3)

# 6. 综合指标雷达图
plt.subplot(2, 3, 6)
# 选择前8个学科显示雷达图
radar_data = ecnu_df.head(8)
angles = np.linspace(0, 2 * np.pi, len(radar_data), endpoint=False).tolist()

# 标准化数据到0-1范围
rank_normalized = 1 - radar_data['Rank_Percent']  # 排名越高，标准化值越大
docs_normalized = radar_data['Documents'] / radar_data['Documents'].max()
cites_normalized = radar_data['Cites'] / radar_data['Cites'].max()

angles += angles[:1]  # 闭合
rank_normalized = list(rank_normalized) + [rank_normalized.iloc[0]]
docs_normalized = docs_normalized.tolist() + [docs_normalized[0]]
cites_normalized = cites_normalized.tolist() + [cites_normalized[0]]

plt.plot(angles, rank_normalized, 'o-', linewidth=2, label='Ranking Performance')
plt.plot(angles, docs_normalized, 's-', linewidth=2, label='Publications')
plt.plot(angles, cites_normalized, '^-', linewidth=2, label='Citations')
plt.xticks(angles[:-1], [s.split()[0] for s in radar_data['Subject']])
plt.ylim(0, 1)
plt.legend()
plt.title('Top 8 Subjects Radar Chart')

plt.tight_layout()
plt.savefig('ecnu_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出统计信息
print("\n=== 华东师范大学学科画像分析 ===")
print(f"总计学科数量: {len(ecnu_df)}")
print(f"强势学科(前20%): {len(strong)}个")
print(f"中等学科(20%-50%): {len(medium)}个")
print(f"弱势学科(50%以后): {len(weak)}个")

print(f"\n平均发文量: {ecnu_df['Documents'].mean():.1f}")
print(f"平均引用数: {ecnu_df['Cites'].mean():.1f}")
print(f"平均篇均引用: {ecnu_df['Cites_Per_Paper'].mean():.2f}")
print(f"平均高被引论文: {ecnu_df['Top_Papers'].mean():.1f}")

print(f"\n最强的3个学科:")
top3 = ecnu_df.nsmallest(3, 'Rank_Percent')
for i, row in top3.iterrows():
    print(f"  {row['Subject']}: 排名前{row['Rank_Percent']*100:.2f}%")

print(f"\n最需要提升的3个学科:")
bottom3 = ecnu_df.nlargest(3, 'Rank_Percent')
for i, row in bottom3.iterrows():
    print(f"  {row['Subject']}: 排名前{row['Rank_Percent']*100:.2f}%")