import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Building ranking prediction models...")
print("=" * 50)

# 获取所有学科文件
csv_files = [file for file in os.listdir("download/") if file.endswith(".csv")]
subjects = [file[:-4] for file in csv_files]

print(f"Processing {len(subjects)} subjects for ranking prediction...")

# 存储每个学科的模型结果
model_results = {}

for csv_file in csv_files:
    subject = csv_file[:-4]

    # 读取数据
    df = pd.read_csv(f"download/{csv_file}", encoding='latin1', skiprows=1)
    df = df.drop(df.index[len(df) - 1])

    # 清理数据
    df['Rank'] = df['Unnamed: 0'].astype(int)
    df['Documents'] = pd.to_numeric(df['Web of Science Documents'], errors='coerce')
    df['Cites'] = pd.to_numeric(df['Cites'], errors='coerce')
    df['Cites_Per_Paper'] = pd.to_numeric(df['Cites/Paper'], errors='coerce')
    df['Top_Papers'] = pd.to_numeric(df['Top Papers'], errors='coerce')

    # 删除有缺失值的行
    df_clean = df.dropna(subset=['Documents', 'Cites', 'Cites_Per_Paper', 'Top_Papers'])

    if len(df_clean) < 50:  # 如果数据太少，跳过
        continue

    # 准备特征和目标变量
    features = ['Documents', 'Cites', 'Cites_Per_Paper', 'Top_Papers']
    X = df_clean[features]
    y = df_clean['Rank']

    # 按排名排序并分割数据（前60%训练，后20%测试，跳过中间20%）
    df_sorted = df_clean.sort_values('Rank')
    n_total = len(df_sorted)

    train_end = int(n_total * 0.6)
    test_start = int(n_total * 0.8)

    train_data = df_sorted.iloc[:train_end]
    test_data = df_sorted.iloc[test_start:]

    X_train = train_data[features]
    y_train = train_data['Rank']
    X_test = test_data[features]
    y_test = test_data['Rank']

    print(f"\n{subject}:")
    print(f"  Total: {n_total}, Train: {len(X_train)}, Test: {len(X_test)}")

    # 训练多个模型
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    subject_results = {}

    for model_name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 评估
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 计算平均绝对百分比误差
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        subject_results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'model': model
        }

        print(f"  {model_name}: RMSE={rmse:.2f}, R²={r2:.3f}, MAPE={mape:.2f}%")

    model_results[subject] = subject_results

print(f"\nOverall model performance summary:")
print("=" * 50)

# 统计所有学科的结果
all_subjects_rmse = []
all_subjects_r2 = []

for subject in model_results:
    # 使用随机森林的结果作为主要指标
    rf_results = model_results[subject]['Random Forest']
    all_subjects_rmse.append(rf_results['rmse'])
    all_subjects_r2.append(rf_results['r2'])

print(f"Average RMSE across all subjects: {np.mean(all_subjects_rmse):.2f}")
print(f"Average R² across all subjects: {np.mean(all_subjects_r2):.3f}")

# 找出表现最好和最差的学科
best_r2_idx = np.argmax(all_subjects_r2)
worst_r2_idx = np.argmin(all_subjects_r2)

best_subject = list(model_results.keys())[best_r2_idx]
worst_subject = list(model_results.keys())[worst_r2_idx]

print(f"Best predicted subject: {best_subject} (R²={all_subjects_r2[best_r2_idx]:.3f})")
print(f"Worst predicted subject: {worst_subject} (R²={all_subjects_r2[worst_r2_idx]:.3f})")

# 特征重要性分析
print(f"\nFeature importance analysis (using Random Forest):")
print("=" * 50)

feature_importance_dict = {'Documents': [], 'Cites': [], 'Cites_Per_Paper': [], 'Top_Papers': []}

for subject in model_results:
    rf_model = model_results[subject]['Random Forest']['model']
    importance = rf_model.feature_importances_
    for i, feature in enumerate(['Documents', 'Cites', 'Cites_Per_Paper', 'Top_Papers']):
        feature_importance_dict[feature].append(importance[i])

for feature in feature_importance_dict:
    avg_importance = np.mean(feature_importance_dict[feature])
    print(f"{feature}: {avg_importance:.4f}")

# 在某些学科上测试模型
print(f"\nDetailed prediction examples:")
print("=" * 50)

test_subjects = ['CHEMISTRY', 'COMPUTER SCIENCE', 'ENGINEERING']

for subject in test_subjects:
    if subject in model_results:
        print(f"\n{subject} predictions:")
        rf_model = model_results[subject]['Random Forest']['model']

        # 获取测试数据
        df = pd.read_csv(f"download/{subject}.csv", encoding='latin1', skiprows=1)
        df = df.drop(df.index[len(df) - 1])
        df['Rank'] = df['Unnamed: 0'].astype(int)
        df['Documents'] = pd.to_numeric(df['Web of Science Documents'], errors='coerce')
        df['Cites'] = pd.to_numeric(df['Cites'], errors='coerce')
        df['Cites_Per_Paper'] = pd.to_numeric(df['Cites/Paper'], errors='coerce')
        df['Top_Papers'] = pd.to_numeric(df['Top Papers'], errors='coerce')
        df_clean = df.dropna()

        df_sorted = df_clean.sort_values('Rank')
        test_data = df_sorted.iloc[int(len(df_sorted) * 0.8):]

        if len(test_data) > 0:
            sample = test_data.head(3)
            features = ['Documents', 'Cites', 'Cites_Per_Paper', 'Top_Papers']
            X_sample = sample[features]
            y_actual = sample['Rank'].values
            y_pred = rf_model.predict(X_sample)

            print("  Actual Rank vs Predicted Rank:")
            for i in range(len(sample)):
                actual_rank = y_actual[i]
                pred_rank = y_pred[i]
                error = abs(actual_rank - pred_rank)
                print(f"    {sample.iloc[i]['Institutions']}: {actual_rank:.0f} vs {pred_rank:.0f} (error: {error:.1f})")

print(f"\nModel building completed!")
print(f"Successfully built prediction models for {len(model_results)} subjects.")