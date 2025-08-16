# 数据可视化分析工具

## 概述
这个工具提供了完整的数据可视化分析功能，专门用于分析学术论文数据集中的研究趋势和关键作者。

## 功能特性

### 📈 Key Research Trends 分析
1. **年度发表量趋势** - 显示论文发表量的时间变化
2. **热门关键词频率** - 分析最常出现的研究关键词
3. **关键词时间趋势** - 追踪关键词随时间的变化

### 👥 Key Authors 分析
1. **作者发表量排名** - 统计每位作者的论文发表数量
2. **作者引用量排名** - 分析作者的学术影响力
3. **作者合作网络** - 可视化作者之间的合作关系

### 📊 综合分析
1. **综合仪表板** - 2x2 布局的综合分析图表
2. **会议分析** - 各会议的发表量和引用量统计
3. **引用量分布** - 论文引用量的分布情况
4. **高引用论文** - 识别最具影响力的论文

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行分析
```bash
python test.py
```

### 3. 在Jupyter Notebook中使用
```python
# 导入模块
from test import *

# 运行完整分析
results = main()

# 或者单独运行某个分析
df = load_and_preprocess_data()
yearly_pubs, top_keywords, keyword_trends = analyze_research_trends(df)
top_authors, top_cited_authors, collaborations = analyze_key_authors(df)
```

## 输出说明

### 图表类型
- **折线图**: 时间趋势分析
- **柱状图**: 频率和排名分析
- **散点图**: 合作关系网络
- **直方图**: 分布分析
- **热力图**: 相关性分析

### 数据输出
- 年度发表量统计
- 关键词频率统计
- 作者发表量和引用量排名
- 会议统计信息
- 高引用论文列表

## 自定义分析

### 修改关键词提取
```python
def extract_keywords(keywords_str):
    # 自定义关键词分割逻辑
    if pd.isna(keywords_str) or keywords_str == '':
        return []
    keywords = re.split(r'[;,]\s*', str(keywords_str))
    return [kw.strip().lower() for kw in keywords if kw.strip()]
```

### 添加新的可视化
```python
def custom_analysis(df):
    # 添加自定义分析逻辑
    fig = px.scatter(df, x='Year', y='CitationCount_CrossRef', 
                     color='Conference', size='CitationCount_Aminer')
    fig.show()
```

## 数据格式要求

CSV文件应包含以下列：
- `Year`: 发表年份
- `Title`: 论文标题
- `Authors`: 作者列表（用分号分隔）
- `AuthorKeywords`: 关键词（用分号分隔）
- `Conference`: 会议名称
- `CitationCount_CrossRef`: CrossRef引用量
- `CitationCount_Aminer`: Aminer引用量

## 注意事项

1. 确保数据集文件名为 `dataset.csv` 并放在同一目录下
2. 图表会自动在浏览器中打开
3. 所有分析结果都会打印到控制台
4. 支持中英文混合显示

## 故障排除

### 常见问题
1. **图表不显示**: 检查浏览器是否阻止弹窗
2. **数据加载错误**: 确认CSV文件格式正确
3. **内存不足**: 对于大数据集，考虑分批处理

### 调试模式
代码包含详细的调试信息，运行时会输出：
- 数据加载状态
- 分析进度
- 错误信息
- 统计摘要
