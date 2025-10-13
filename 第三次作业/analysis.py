import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists('pictures'):
    os.makedirs('pictures')

# Load the dataset
df = pd.read_csv('./csv/esi_fields_of_East_China_Normal_University.csv')

# Data Cleaning and Preparation
df = df[df['name'] != 'ALL FIELDS']

df.rename(columns={
    'name': 'Discipline',
    'cites': 'Citations',
    'cites_per_paper': 'Citations_per_Paper'
}, inplace=True)

df['Citations'] = pd.to_numeric(df['Citations'])
df['Citations_per_Paper'] = pd.to_numeric(df['Citations_per_Paper'])

# Data Analysis
# Sort by Citations
df_sorted_by_citations = df.sort_values(by='Citations', ascending=False)

# Sort by Citations per Paper
df_sorted_by_cpp = df.sort_values(by='Citations_per_Paper', ascending=False)


# Visualization

# 1. Bar Chart: Citations by Discipline
plt.figure(figsize=(12, 8))
plt.barh(df_sorted_by_citations['Discipline'], df_sorted_by_citations['Citations'], color='skyblue')
plt.xlabel('Citations')
plt.ylabel('Discipline')
plt.title('Citations by Discipline at East China Normal University')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('pictures/citations_by_discipline.png')
plt.close()

# 2. Bar Chart: Citations per Paper by Discipline
plt.figure(figsize=(12, 8))
plt.barh(df_sorted_by_cpp['Discipline'], df_sorted_by_cpp['Citations_per_Paper'], color='lightcoral')
plt.xlabel('Citations per Paper')
plt.ylabel('Discipline')
plt.title('Citations per Paper by Discipline at East China Normal University')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('pictures/citations_per_paper_by_discipline.png')
plt.close()

# 3. Scatter Plot: Citations vs. Citations per Paper
plt.figure(figsize=(10, 6))
plt.scatter(df['Citations'], df['Citations_per_Paper'], alpha=0.7)
plt.xlabel('Citations')
plt.ylabel('Citations per Paper')
plt.title('Citations vs. Citations per Paper')
for i, txt in enumerate(df['Discipline']):
    plt.annotate(txt, (df['Citations'][i], df['Citations_per_Paper'][i]), fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig('pictures/citations_vs_cpp_scatter.png')
plt.close()


df_chem = pd.read_csv('./csv/esi_institutions_by_Chemistry.csv')
ecnu_chem_rank = df_chem[df_chem['name'] == 'EAST CHINA NORMAL UNIVERSITY'].index[0] + 1

df_cs = pd.read_csv('./csv/esi_institutions_by_Computer_Science.csv')
ecnu_cs_rank = df_cs[df_cs['name'] == 'EAST CHINA NORMAL UNIVERSITY'].index[0] + 1

print(ecnu_chem_rank, ecnu_cs_rank)

# Save
df_sorted_by_citations.to_csv('./csv/df_sorted_by_citations.csv', index=False)
df_sorted_by_cpp.to_csv('./csv/df_sorted_by_cpp.csv', index=False)
