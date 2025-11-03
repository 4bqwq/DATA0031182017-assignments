# DATA0031182017 - Introduction to Data Science 作业仓库

本仓库用于存放华东师范大学 **数据科学导论** (DATA0031182017) 课程的所有作业。  
每次作业均单独放在一个文件夹中，文件夹命名为第X次作业（X 表示作业序号）。

---

## 课程信息

- **课程名称**: 数据科学导论 (Introduction to Data Science)  
- **课程编号**: DATA0031182017

---

## 仓库结构

```bash
.
├── 第一次作业/            # 第一次作业
│   ├── dorm_manager.py
│   ├── students.json
│   └── README.md
├── 第二次作业/            # 第二次作业（房价数据预处理）
│   ├── data/
│   │   ├── data_description.txt
│   │   ├── train.csv
│   │   └── train_processed.csv
│   ├── figures/
│   │   ├── price_boxplot.png
│   │   ├── price_corr_heatmap.png
│   │   ├── price_hist.png
│   │   ├── table1.png
│   │   ├── table2.png
│   │   ├── table3.png
│   │   └── table4.png
│   ├── lab2.ipynb
│   └── README.md
├── 第三次作业/            # 第三次作业（ESI 数据分析与可视化）
│   ├── analysis.py
│   ├── scrape-esi.js
│   ├── csv/
│   │   ├── esi_fields_of_East_China_Normal_University.csv
│   │   ├── esi_institutions_by_*.csv
│   │   ├── df_sorted_by_citations.csv
│   │   └── df_sorted_by_cpp.csv
│   ├── pictures/
│   │   ├── citations_by_discipline.png
│   │   ├── citations_per_paper_by_discipline.png
│   │   └── citations_vs_cpp_scatter.png
│   ├── package.json
│   ├── package-lock.json
│   └── README.md
├── 第四次作业/            # 第四次作业（全球大学排名数据 SQL 分析）
│   ├── create_tables.sql
│   ├── insert_fields.sql
│   ├── create_staging_table.sql
│   ├── migrate_universities.sql
│   ├── migrate_rankings.sql
│   ├── query_ecnu_rankings.sql
│   ├── query_china_performance.sql
│   ├── query_global_analysis.sql
│   ├── query_top_regions.sql
│   ├── verification_queries.sql
│   ├── docker-compose.yml
│   ├── import_data.sh
│   ├── download/
│   │   └── *.csv
│   ├── pictures/
│   │   ├── original_csv_format.png
│   │   ├── database_tables.png
│   │   ├── data_import_complete.png
│   │   ├── ecnu_rankings.png
│   │   ├── china_performance.png
│   │   ├── global_analysis_basic.png
│   │   └── global_analysis_advanced.png
│   └── README.md
├── 第五次作业/            # 第五次作业（全球高校聚类分析与排名预测）
│   ├── download/
│   │   └── *.csv
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── q8_university_clustering.py
│   ├── q8_cluster_analysis.py
│   ├── q9_ecnu_analysis.py
│   ├── q9_visualization.py
│   ├── q9_final_summary.py
│   ├── q10_ranking_prediction.py
│   ├── q10_model_analysis.py
│   ├── ecnu_analysis.png
│   ├── university_clusters.png
│   ├── model_analysis.png
│   └── README.md
├── 第六次作业/            # 第六次作业（科研机构排名预测与聚类分析）
│   ├── download/
│   │   └── *.csv
│   ├── data/
│   │   ├── cleaned_records.json
│   │   ├── institution_summary.json
│   │   └── cleaned_records.meta.json
│   ├── src/
│   │   ├── data_processing.py
│   │   ├── ranking_data.py
│   │   └── ranking_model.py
│   ├── scripts/
│   │   ├── export_data.py
│   │   ├── train_ranking.py
│   │   ├── predict_rank.py
│   │   └── cluster_institutions.py
│   ├── models/
│   │   └── ranking_model.pt
│   ├── logs/
│   │   ├── ranking_metrics.json
│   │   ├── predicted_ranks.json
│   │   ├── cluster_similar_ecnu.json
│   │   └── cluster_assignments.json
│   ├── pictures/
│   │   ├── export_data_cleaned_json_preview.png
│   │   ├── pipline.png
│   │   ├── train-console.png
│   │   ├── predicted_ranks_ecnu.png
│   │   ├── predicted_ranks_thu.png
│   │   ├── predicted_ranks_sjtu.png
│   │   ├── ecnu-ranking-bar.png
│   │   ├── cluster_result.png
│   │   └── cluster-scatter.png
│   ├── pyproject.toml
│   ├── uv.lock
│   └── README.md

````

---

## 作业列表

| 作业编号 | 文件夹     | 内容简介                                                     |
| -------- | ---------- | ------------------------------------------------------------ |
| Lab 1    | 第一次作业 | 宿舍管理程序：读取学生信息、实现基本增删查改功能             |
| Lab 2    | 第二次作业 | 房价数据预处理：缺失值检测与填充，特征分析。                 |
| Lab 3    | 第三次作业 | ESI 数据分析与可视化：对学科与机构的引文数据进行统计与可视化展示。 |
| Lab 4    | 第四次作业 | 全球大学排名数据 SQL 分析：使用 PostgreSQL 数据库进行数据建模、导入和多维度查询分析。 |
| Lab 5    | 第五次作业 | 全球高校聚类分析与排名预测：使用机器学习算法对全球高校进行分类分析，华东师范大学学科画像分析，以及学科排名预测模型构建。 |
| Lab 6    | 第六次作业 | 科研机构排名预测与聚类分析：基于 ESI 学科数据，使用带嵌入层和残差结构的神经网络模型进行排名预测，并结合机构统计特征进行 KMeans 聚类分析。 |

---

## 使用说明

进入对应作业文件夹，阅读该文件夹下的 `README.md` 了解具体运行方式。
