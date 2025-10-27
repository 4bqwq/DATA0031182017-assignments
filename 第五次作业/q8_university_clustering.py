import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting university clustering analysis...")
print("=" * 50)

# 获取所有CSV文件
csv_files = [file for file in os.listdir("download/") if file.endswith(".csv")]
subjects = [file[:-4] for file in csv_files]

print(f"Found {len(subjects)} subjects: {subjects[:5]}...")

# 构建大学排名矩阵
university_rank_matrix = {}
university_data = {}

for csv_file in csv_files:
    subject = csv_file[:-4]

    # 读取数据
    df = pd.read_csv(f"download/{csv_file}", encoding='latin1', skiprows=1)
    df = df.drop(df.index[len(df) - 1])
    df['Rank'] = df['Unnamed: 0'].astype(int)

    # 计算排名百分位（越小越好）
    total_institutions = len(df)
    df['Rank_Percentile'] = df['Rank'] / total_institutions

    # 构建大学排名字典
    for _, row in df.iterrows():
        university = row['Institutions']
        if university not in university_rank_matrix:
            university_rank_matrix[university] = {}
            university_data[university] = {}

        university_rank_matrix[university][subject] = row['Rank_Percentile']
        university_data[university][subject] = {
            'rank': row['Rank'],
            'total': total_institutions,
            'documents': row['Web of Science Documents'],
            'cites': row['Cites'],
            'cites_per_paper': row['Cites/Paper'],
            'top_papers': row['Top Papers']
        }

print(f"Total universities found: {len(university_rank_matrix)}")

# 找出在至少10个学科中有排名的大学
eligible_universities = []
for university in university_rank_matrix:
    if len(university_rank_matrix[university]) >= 10:
        eligible_universities.append(university)

print(f"Universities with rankings in >=10 subjects: {len(eligible_universities)}")

# 创建特征矩阵
universities_list = eligible_universities
feature_matrix = []

for university in universities_list:
    row = []
    for subject in subjects:
        if subject in university_rank_matrix[university]:
            row.append(university_rank_matrix[university][subject])
        else:
            row.append(1.0)  # 如果没有排名，设为最差排名
    feature_matrix.append(row)

feature_matrix = np.array(feature_matrix)
print(f"Feature matrix shape: {feature_matrix.shape}")

# 标准化数据
scaler = StandardScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)

# 使用K-means聚类
print("\nPerforming K-means clustering...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

# 添加聚类结果
university_clusters = {}
for i, university in enumerate(universities_list):
    university_clusters[university] = {
        'cluster': cluster_labels[i],
        'subjects_count': len([s for s in subjects if s in university_rank_matrix[university]])
    }

# 分析每个聚类
print("\nCLUSTER ANALYSIS:")
print("=" * 50)
for cluster_id in range(5):
    cluster_universities = [u for u in university_clusters if university_clusters[u]['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_universities)} universities):")

    # 显示该聚类中的代表性大学
    sample_universities = cluster_universities[:5]
    for uni in sample_universities:
        subjects_count = university_clusters[uni]['subjects_count']
        print(f"  - {uni} ({subjects_count} subjects)")

    if len(cluster_universities) > 5:
        print(f"  ... and {len(cluster_universities) - 5} more")

# 找到华东师范大学
print(f"\nEAST CHINA NORMAL UNIVERSITY ANALYSIS:")
if "EAST CHINA NORMAL UNIVERSITY" in university_clusters:
    ecnu_cluster = university_clusters["EAST CHINA NORMAL UNIVERSITY"]['cluster']
    ecnu_subjects = university_clusters["EAST CHINA NORMAL UNIVERSITY"]['subjects_count']
    print(f"ECNU belongs to Cluster {ecnu_cluster}")
    print(f"ECNU has rankings in {ecnu_subjects} subjects")

    # 找到同聚类的高校
    similar_universities = [u for u in university_clusters
                          if university_clusters[u]['cluster'] == ecnu_cluster
                          and u != "EAST CHINA NORMAL UNIVERSITY"]

    print(f"\nUniversities in the same cluster as ECNU:")
    for uni in similar_universities[:10]:
        subjects_count = university_clusters[uni]['subjects_count']
        print(f"  - {uni} ({subjects_count} subjects)")

    if len(similar_universities) > 10:
        print(f"  ... and {len(similar_universities) - 10} more")
else:
    print("East China Normal University not found in the dataset")

# 计算与华东师范大学最相似的高校
if "EAST CHINA NORMAL UNIVERSITY" in university_rank_matrix:
    print(f"\nMOST SIMILAR UNIVERSITIES TO ECNU:")
    print("=" * 50)

    ecnu_vector = []
    for subject in subjects:
        if subject in university_rank_matrix["EAST CHINA NORMAL UNIVERSITY"]:
            ecnu_vector.append(university_rank_matrix["EAST CHINA NORMAL UNIVERSITY"][subject])
        else:
            ecnu_vector.append(1.0)

    ecnu_vector = np.array(ecnu_vector).reshape(1, -1)
    ecnu_vector_scaled = scaler.transform(ecnu_vector)

    # 计算与所有其他大学的相似度
    similarities = []
    for i, university in enumerate(universities_list):
        if university != "EAST CHINA NORMAL UNIVERSITY":
            other_vector = feature_matrix_scaled[i].reshape(1, -1)
            # 使用欧氏距离，距离越小越相似
            distance = euclidean_distances(ecnu_vector_scaled, other_vector)[0][0]
            similarities.append((university, distance))

    # 按距离排序
    similarities.sort(key=lambda x: x[1])

    print("Top 15 most similar universities to ECNU:")
    for i, (university, distance) in enumerate(similarities[:15]):
        subjects_count = university_clusters[university]['subjects_count']
        print(f"{i+1:2d}. {university} (distance: {distance:.3f}, {subjects_count} subjects)")