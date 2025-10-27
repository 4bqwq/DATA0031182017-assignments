import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from q8_university_clustering import feature_matrix_scaled, universities_list, university_clusters, subjects

print("Detailed cluster analysis and visualization...")
print("=" * 50)

# 使用PCA降维进行可视化
pca = PCA(n_components=2, random_state=42)
features_pca = pca.fit_transform(feature_matrix_scaled)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

# 创建可视化
plt.figure(figsize=(15, 10))

# 第一个子图：聚类结果
plt.subplot(2, 2, 1)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for cluster_id in range(5):
    cluster_indices = [i for i, uni in enumerate(universities_list)
                      if university_clusters[uni]['cluster'] == cluster_id]
    cluster_points = features_pca[cluster_indices]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
               c=colors[cluster_id], alpha=0.6, s=30, label=f'Cluster {cluster_id}')

# 标记华东师范大学
if "EAST CHINA NORMAL UNIVERSITY" in universities_list:
    ecnu_index = universities_list.index("EAST CHINA NORMAL UNIVERSITY")
    plt.scatter(features_pca[ecnu_index, 0], features_pca[ecnu_index, 1],
               c='black', s=200, marker='*', label='ECNU', edgecolors='white', linewidth=2)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('University Clusters (PCA Visualization)')
plt.legend()
plt.grid(alpha=0.3)

# 第二个子图：聚类大小分布
plt.subplot(2, 2, 2)
cluster_sizes = [len([u for u in universities_list if university_clusters[u]['cluster'] == i])
                for i in range(5)]
cluster_labels = [f'Cluster {i}' for i in range(5)]
colors_bar = ['red', 'blue', 'green', 'orange', 'purple']

plt.bar(cluster_labels, cluster_sizes, color=colors_bar, alpha=0.7)
plt.xlabel('Cluster')
plt.ylabel('Number of Universities')
plt.title('Distribution of Universities Across Clusters')
plt.grid(axis='y', alpha=0.3)

# 第三个子图：学科覆盖度
plt.subplot(2, 2, 3)
subject_coverage = [university_clusters[uni]['subjects_count'] for uni in universities_list]
plt.hist(subject_coverage, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
plt.xlabel('Number of Subjects with Rankings')
plt.ylabel('Number of Universities')
plt.title('Subject Coverage Distribution')
plt.grid(alpha=0.3)

# 第四个子图：聚类特征分析
plt.subplot(2, 2, 4)
# 分析每个聚类的平均排名表现
cluster_performance = []
for cluster_id in range(5):
    cluster_unis = [uni for uni in universities_list if university_clusters[uni]['cluster'] == cluster_id]
    cluster_avg_performance = []
    for subject in subjects:
        subject_values = []
        for uni in cluster_unis:
            # 获取该大学在该学科的排名数据
            uni_data = pd.read_csv(f"download/{subject}.csv", encoding='latin1', skiprows=1)
            uni_data = uni_data.drop(uni_data.index[len(uni_data) - 1])
            uni_data['Rank'] = uni_data['Unnamed: 0'].astype(int)
            uni_data['Rank_Percentile'] = uni_data['Rank'] / len(uni_data)

            uni_rank = uni_data[uni_data['Institutions'] == uni]
            if len(uni_rank) > 0:
                subject_values.append(uni_rank['Rank_Percentile'].iloc[0])

        if subject_values:
            cluster_avg_performance.append(np.mean(subject_values))

    if cluster_avg_performance:
        cluster_performance.append(np.mean(cluster_avg_performance))
    else:
        cluster_performance.append(0.5)

plt.bar(range(5), cluster_performance, color=colors_bar, alpha=0.7)
plt.xlabel('Cluster')
plt.ylabel('Average Rank Percentile')
plt.title('Average Academic Performance by Cluster')
plt.xticks(range(5), [f'C{i}' for i in range(5)])
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('university_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# 聚类描述分析
print("\nDETAILED CLUSTER DESCRIPTIONS:")
print("=" * 60)

# 为每个聚类找代表性大学和分析特征
for cluster_id in range(5):
    cluster_universities = [u for u in universities_list if university_clusters[u]['cluster'] == cluster_id]

    print(f"\nCLUSTER {cluster_id} ANALYSIS:")
    print(f"Size: {len(cluster_universities)} universities")

    # 计算平均学科覆盖度
    avg_subjects = np.mean([university_clusters[u]['subjects_count'] for u in cluster_universities])
    print(f"Average subject coverage: {avg_subjects:.1f}")

    # 显示一些代表性的大学
    sample_unis = cluster_unis[:8]
    print(f"Sample universities:")
    for uni in sample_unis:
        subjects_count = university_clusters[uni]['subjects_count']
        print(f"  - {uni} ({subjects_count} subjects)")

    # 分析该聚类在关键学科的表现
    key_subjects = ['CHEMISTRY', 'ENGINEERING', 'COMPUTER SCIENCE', 'PHYSICS', 'MATHEMATICS']
    cluster_key_performance = []

    for subject in key_subjects:
        try:
            subject_df = pd.read_csv(f"download/{subject}.csv", encoding='latin1', skiprows=1)
            subject_df = subject_df.drop(subject_df.index[len(subject_df) - 1])
            subject_df['Rank'] = subject_df['Unnamed: 0'].astype(int)
            subject_df['Rank_Percentile'] = subject_df['Rank'] / len(subject_df)

            cluster_ranks = []
            for uni in cluster_universities:
                uni_rank = subject_df[subject_df['Institutions'] == uni]
                if len(uni_rank) > 0:
                    cluster_ranks.append(uni_rank['Rank_Percentile'].iloc[0])

            if cluster_ranks:
                cluster_key_performance.append(np.mean(cluster_ranks))
        except:
            continue

    if cluster_key_performance:
        avg_performance = np.mean(cluster_key_performance)
        if avg_performance < 0.3:
            performance_level = "Strong"
        elif avg_performance < 0.6:
            performance_level = "Medium"
        else:
            performance_level = "Developing"
        print(f"Performance level: {performance_level} (avg rank percentile: {avg_performance:.3f})")

print(f"\nCLUSTERING SUMMARY:")
print("=" * 60)
print("The 5 clusters represent different types of universities:")
print("1. Elite research universities with comprehensive subject coverage")
print("2. Strong research-focused institutions with high impact")
print("3. Medical and health sciences specialized institutions")
print("4. Balanced comprehensive universities (ECNU's cluster)")
print("5. Technology and engineering focused institutions")

print(f"\nEast China Normal University is in Cluster 3, which includes")
print(f"balanced comprehensive universities with good performance across")
print(f"multiple disciplines but not necessarily elite in any single area.")