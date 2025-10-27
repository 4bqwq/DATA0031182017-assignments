import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取所有CSV文件
csv_files = [file for file in os.listdir("download/") if file.endswith(".csv")]

# 存储华师大的数据
ecnu_data = []

print("开始分析华东师范大学的学科排名...")
print("=" * 50)

for csv_file in csv_files:
    # 学科名称
    subject = csv_file[:-4]

    # 读取数据
    df = pd.read_csv(f"download/{csv_file}", encoding='latin1', skiprows=1)

    # 删除最后一行
    df = df.drop(df.index[len(df) - 1])

    # 添加排名列
    df['Rank'] = df['Unnamed: 0'].astype(int)

    # 查找华东师范大学
    ecnu_row = df[df['Institutions'] == "EAST CHINA NORMAL UNIVERSITY"]

    if len(ecnu_row) > 0:
        # 获取华师大的数据
        rank = ecnu_row['Rank'].values[0]
        total_institutions = len(df)
        rank_percent = rank / total_institutions

        # 获取各项指标
        docs = ecnu_row['Web of Science Documents'].values[0]
        cites = ecnu_row['Cites'].values[0]
        cites_per_paper = ecnu_row['Cites/Paper'].values[0]
        top_papers = ecnu_row['Top Papers'].values[0]

        print(f"{subject}")
        print(f"排名: {rank}/{total_institutions} (前{rank_percent:.2%})")
        print(f"发文量: {docs}, 引用数: {cites}, 篇均引用: {cites_per_paper:.2f}, 高被引论文: {top_papers}")
        print("-" * 30)

        ecnu_data.append({
            'Subject': subject,
            'Rank': rank,
            'Total': total_institutions,
            'Rank_Percent': rank_percent,
            'Documents': docs,
            'Cites': cites,
            'Cites_Per_Paper': cites_per_paper,
            'Top_Papers': top_papers
        })

# 转换为DataFrame
ecnu_df = pd.DataFrame(ecnu_data)

print(f"\n华东师范大学共有 {len(ecnu_df)} 个学科进入排名")