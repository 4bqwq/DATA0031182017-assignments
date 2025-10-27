import pandas as pd
import numpy as np
from q9_ecnu_analysis import ecnu_df

# 华东师范大学学科画像总结分析
print("=" * 60)
print("EAST CHINA NORMAL UNIVERSITY - ACADEMIC PROFILE ANALYSIS")
print("=" * 60)

# 基本统计
print(f"\nBASIC STATISTICS:")
print(f"Total disciplines with rankings: {len(ecnu_df)}")
print(f"Average documents per discipline: {ecnu_df['Documents'].mean():.1f}")
print(f"Average citations per discipline: {ecnu_df['Cites'].mean():.1f}")
print(f"Average citations per paper: {ecnu_df['Cites_Per_Paper'].mean():.2f}")
print(f"Average top papers per discipline: {ecnu_df['Top_Papers'].mean():.1f}")

# 学科分类
strong = ecnu_df[ecnu_df['Rank_Percent'] < 0.2]
medium = ecnu_df[(ecnu_df['Rank_Percent'] >= 0.2) & (ecnu_df['Rank_Percent'] < 0.5)]
weak = ecnu_df[ecnu_df['Rank_Percent'] >= 0.5]

print(f"\nDISCIPLINE CLASSIFICATION:")
print(f"Strong disciplines (top 20%): {len(strong)}")
print(f"Medium disciplines (20%-50%): {len(medium)}")
print(f"Weak disciplines (below 50%): {len(weak)}")

# 强势学科
print(f"\nSTRONG DISCIPLINES (Top 20%):")
for i, row in strong.iterrows():
    print(f"  {row['Subject']}: Rank {row['Rank']}/{row['Total']} (top {row['Rank_Percent']*100:.2f}%)")

# 中等学科
print(f"\nMEDIUM DISCIPLINES (20%-50%):")
for i, row in medium.iterrows():
    print(f"  {row['Subject']}: Rank {row['Rank']}/{row['Total']} (top {row['Rank_Percent']*100:.2f}%)")

# 弱势学科
print(f"\nWEAK DISCIPLINES (Below 50%):")
for i, row in weak.iterrows():
    print(f"  {row['Subject']}: Rank {row['Rank']}/{row['Total']} (top {row['Rank_Percent']*100:.2f}%)")

# 关键发现
print(f"\nKEY FINDINGS:")

# 最强学科
best_discipline = ecnu_df.loc[ecnu_df['Rank_Percent'].idxmin()]
print(f"1. Best performing discipline: {best_discipline['Subject']} (top {best_discipline['Rank_Percent']*100:.2f}%)")

# 最弱学科
worst_discipline = ecnu_df.loc[ecnu_df['Rank_Percent'].idxmax()]
print(f"2. Discipline needing most improvement: {worst_discipline['Subject']} (top {worst_discipline['Rank_Percent']*100:.2f}%)")

# 最高引用
most_cited = ecnu_df.loc[ecnu_df['Cites'].idxmax()]
print(f"3. Most cited discipline: {most_cited['Subject']} ({most_cited['Cites']:.0f} citations)")

# 最高篇均引用
highest_impact = ecnu_df.loc[ecnu_df['Cites_Per_Paper'].idxmax()]
print(f"4. Highest impact per paper: {highest_impact['Subject']} ({highest_impact['Cites_Per_Paper']:.2f} citations/paper)")

# 最多高被引论文
most_top_papers = ecnu_df.loc[ecnu_df['Top_Papers'].idxmax()]
print(f"5. Most top papers: {most_top_papers['Subject']} ({most_top_papers['Top_Papers']} top papers)")

# 特色分析
print(f"\nSPECIAL CHARACTERISTICS:")
chemistry_data = ecnu_df[ecnu_df['Subject'] == 'CHEMISTRY'].iloc[0]
print(f"- Chemistry stands out as the flagship discipline (top {chemistry_data['Rank_Percent']*100:.2f}%)")
print(f"- High research output in Chemistry: {chemistry_data['Documents']:.0f} papers")
print(f"- Strong citation impact in Chemistry: {chemistry_data['Cites']:.0f} total citations")

# 工程类学科表现
engineering_disciplines = ['ENGINEERING', 'COMPUTER SCIENCE', 'MATERIALS SCIENCE']
eng_data = ecnu_df[ecnu_df['Subject'].isin(engineering_disciplines)]
if len(eng_data) > 0:
    avg_eng_rank = eng_data['Rank_Percent'].mean() * 100
    print(f"- Engineering disciplines perform well overall (average top {avg_eng_rank:.2f}%)")

# 生命科学类学科表现
life_sciences = ['BIOLOGY & BIOCHEMISTRY', 'CLINICAL MEDICINE', 'MOLECULAR BIOLOGY & GENETICS', 'NEUROSCIENCE & BEHAVIOR']
life_data = ecnu_df[ecnu_df['Subject'].isin(life_sciences)]
if len(life_data) > 0:
    avg_life_rank = life_data['Rank_Percent'].mean() * 100
    print(f"- Life sciences show mixed performance (average top {avg_life_rank:.2f}%)")

print(f"\nCONCLUSION:")
print("East China Normal University shows strong research capabilities in")
print("Chemistry, Environmental Ecology, and Engineering disciplines.")
print("The university has a balanced research profile with 17 disciplines")
print("participating in international rankings, demonstrating comprehensive")
print("academic strength across multiple fields.")