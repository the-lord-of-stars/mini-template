import pandas as pd
import altair as alt
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

# preprocessing

def run_topic_modeling(text_column: str, dataset_url: str = None, n_topics: int = 10, cache_file: str = "dataset_with_topics.csv"):
    """
    运行topic modeling，支持缓存结果避免重复计算
    
    Args:
        text_column: 文本列名
        dataset_url: 数据集路径
        n_topics: 主题数量
        cache_file: 缓存文件名
    """
    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        df_with_topics = pd.read_csv(cache_file)
        
        # 创建一个简单的topic model对象用于显示信息
        topic_model = BERTopic()
        topic_model._fitted = True
        
        # 从数据中提取topic信息
        topic_counts = df_with_topics['Topic_Label'].value_counts()
        topic_info = pd.DataFrame({
            'Topic': range(len(topic_counts)),
            'Name': topic_counts.index,
            'Count': topic_counts.values
        })
        topic_model.topic_info_ = topic_info
        
        return topic_model, df_with_topics, "Cached results loaded"

    if dataset_url is None:
        dataset_url = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"

    df = pd.read_csv(dataset_url)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset.")

    df_clean = df[[text_column, 'Year']].copy()
    df_clean[text_column] = df_clean[text_column].astype(str).str.strip()
    df_clean = df_clean[df_clean[text_column].str.len() > 10]
    df_clean = df_clean.dropna()

    texts = df_clean[text_column].tolist()

    # 配置BERTopic以限制主题数量
    vectorizer_model = CountVectorizer(stop_words="english", max_features=1000)
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        nr_topics=n_topics,  # 限制主题数量
        min_topic_size=5,    # 最小主题大小
        verbose=True
    )
    
    print(f"Running topic modeling with {n_topics} topics...")
    topics, probs = topic_model.fit_transform(texts)

    # 将topic结果添加到原始数据中
    df_with_topics = df.copy()
    df_with_topics['Topic'] = -1  # 默认值
    df_with_topics['Topic_Prob'] = 0.0  # 默认概率
    
    # 将topic结果映射回原始数据
    topic_idx = 0
    for idx, row in df.iterrows():
        if topic_idx < len(topics):
            df_with_topics.loc[idx, 'Topic'] = topics[topic_idx]
            # 处理概率值
            if isinstance(probs[topic_idx], (list, np.ndarray)):
                df_with_topics.loc[idx, 'Topic_Prob'] = probs[topic_idx].max() if len(probs[topic_idx]) > 0 else 0.0
            else:
                df_with_topics.loc[idx, 'Topic_Prob'] = probs[topic_idx] if probs[topic_idx] > 0 else 0.0
            topic_idx += 1

    # 获取topic标签并添加到数据中
    topic_info = topic_model.get_topic_info()
    topic_labels = {}
    for _, row in topic_info.iterrows():
        topic_labels[row['Topic']] = row['Name']
    
    df_with_topics['Topic_Label'] = df_with_topics['Topic'].map(topic_labels)
    df_with_topics['Topic_Label'] = df_with_topics['Topic_Label'].fillna('No Topic')

    # 保存结果到缓存文件
    df_with_topics.to_csv(cache_file, index=False)
    print(f"Results saved to {cache_file}")
    
    # 保存topic model信息
    topic_model_info = {
        "topic_info": topic_model.get_topic_info(),
        "n_topics": n_topics
    }
    with open("topic_model_info.pkl", "wb") as f:
        pickle.dump(topic_model_info, f)

    # Don't write to file — capture HTML string instead
    html_str = topic_model.visualize_topics().to_html()

    return topic_model, df_with_topics, html_str

def create_horizontal_ridgeline_plot(df_with_topics):
    """
    创建横向Ridgeline图，X轴是年份，参考Vega的U-District Cuisine示例
    """
    # 过滤掉无主题的文档
    df_filtered = df_with_topics[df_with_topics['Topic'] != -1].copy()
    
    # 按年份和主题统计论文数量
    year_topic_counts = df_filtered.groupby(['Year', 'Topic_Label']).size().reset_index(name='Count')
    
    # 确保所有年份都有数据（填充0）
    all_years = range(year_topic_counts['Year'].min(), year_topic_counts['Year'].max() + 1)
    all_topics = year_topic_counts['Topic_Label'].unique()
    
    # 创建完整的年份-topic组合
    complete_data = []
    for year in all_years:
        for topic in all_topics:
            count = year_topic_counts[
                (year_topic_counts['Year'] == year) & 
                (year_topic_counts['Topic_Label'] == topic)
            ]['Count'].iloc[0] if len(year_topic_counts[
                (year_topic_counts['Year'] == year) & 
                (year_topic_counts['Topic_Label'] == topic)
            ]) > 0 else 0
            complete_data.append({'Year': year, 'Topic_Label': topic, 'Count': count})
    
    plot_df = pd.DataFrame(complete_data)
    
    # 计算每个主题的平均数量用于颜色编码
    topic_means = plot_df.groupby('Topic_Label')['Count'].mean().reset_index()
    topic_means.columns = ['Topic_Label', 'Mean_Count']
    plot_df = plot_df.merge(topic_means, on='Topic_Label')
    
    # 创建横向Ridgeline图
    step = 30  # 每个主题的高度
    overlap = 0.8  # 重叠程度
    
    chart = alt.Chart(plot_df, height=step * len(all_topics)).transform_joinaggregate(
        mean_count='mean(Count)', groupby=['Topic_Label']
    ).transform_bin(
        ['bin_max', 'bin_min'], 'Count'
    ).transform_aggregate(
        value='count()', groupby=['Topic_Label', 'mean_count', 'bin_min', 'bin_max']
    ).transform_impute(
        impute='value', groupby=['Topic_Label', 'mean_count'], key='bin_min', value=0
    ).mark_area(
        interpolate='monotone',
        fillOpacity=0.8,
        stroke='white',
        strokeWidth=1
    ).encode(
        alt.X('bin_min:Q')
            .bin('binned')
            .title('Paper Count'),
        alt.Y('value:Q')
            .axis(None)
            .scale(range=[step, -step * overlap]),
        alt.Fill('mean_count:Q')
            .legend(None)
            .scale(scheme='viridis'),
        alt.Row('Topic_Label:N')
            .title(None)
            .header(labelAngle=0, labelAlign='left')
    ).properties(
        title='Topic Distribution Over Years (Horizontal Ridgeline Plot)',
        bounds='flush'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    ).configure_title(
        anchor='end'
    )
    
    return chart

# Run topic modeling on abstracts from filtered dataset
topic_model, df_clean, html_str = run_topic_modeling("Abstract", "dataset.csv", n_topics=10)

print("\nTopic modeling results:")
# 直接从数据中获取topic信息
topic_counts = df_clean['Topic_Label'].value_counts()
print(f"Number of topics: {len(topic_counts)}")
print("\nTopic information:")
topic_info_df = pd.DataFrame({
    'Topic': range(len(topic_counts)),
    'Name': topic_counts.index,
    'Count': topic_counts.values
})
print(topic_info_df.head(10))

# 创建横向Ridgeline图
print("\nCreating horizontal ridgeline plot...")
horizontal_ridgeline_chart = create_horizontal_ridgeline_plot(df_clean)
horizontal_ridgeline_chart.save('topic_year_horizontal_ridgeline.html')
print("Horizontal ridgeline plot saved as 'topic_year_horizontal_ridgeline.html'")

# 显示图表信息
print(f"Created horizontal ridgeline plot with {len(topic_counts)} topics")

