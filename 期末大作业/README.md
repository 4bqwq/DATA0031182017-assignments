# COVID-19 社交媒体舆论注意力结构分析

### 安装依赖

```bash
uv sync
```

### 配置 API Key

在项目根目录创建一个 `.env` 文件，填入你的 API Key：

```env
API_KEY=your_key_here
```

### 数据采集 (Crawler)

数据采集模块位于 `TwitterCrawler-Selenium/` 目录下，采用 Selenium 模拟浏览器行为。

### 运行项目

可以直接通过 `main.py` 运行分析的各个步骤：

```bash
# 1. 清洗数据
uv run main.py clean

# 2. 生成基础图表 (EDA)
uv run main.py eda

# 3. 话题聚类 (需要 API)
uv run main.py cluster

# 4. 话题标注 (需要 API)
uv run main.py label

# 5. 计算指标
uv run main.py metrics
uv run main.py inequality

# 6. 统计建模
uv run main.py model

# 7. 稳健性检验
uv run main.py robustness
```

### 查看结果

所有结果将自动生成在 `outputs/` 文件夹中，包括 CSV 数据和图表。
### 报告

本次实验的结果报告详见 `./report.md` 和 `./report.pdf`，演示文稿（PPT）位于 `./PPT.pdf`。
