# 基于Selenium的Twitter爬虫

## 环境配置

1. 下载 Edge WebDriver
    
2. 安装依赖
    
   ```bash
   pip install selenium
   ```

## 使用方法

执行以下命令开始爬取数据：

```bash
python main.py
```

程序会默认进行登录状态校验。如果未登录，程序会弹出登录窗口，您需要手动完成登录操作。

### 修改搜索关键词

要更改搜索的关键词，可以编辑主程序中的以下代码段：

```python
main(keyword='')
```

将 `keyword=''` 替换为您希望搜索的关键词，例如：

```python
main(keyword='Python')
```

建议使用 Advanced Search，例如

```python
main(keyword = '("COVID-19" OR "coronavirus" OR "COVID" OR "vaccine" OR "vaccination" OR "remote working" OR "work from home" OR "WFH") since:2021-01-01 until:2021-12-31')
```

