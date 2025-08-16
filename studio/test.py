import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from collections import Counter
import numpy as np
from datetime import datetime
import re

# 设置plotly离线模式
pyo.init_notebook_mode(connected=True)

# ============================================================================
# 数据加载和预处理
# ============================================================================

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    
    # 读取数据
    df = pd.read_csv("dataset.csv")
    
    # 转换年份为datetime
    df["Year"] = pd.to_datetime(df["Year"], format='%Y')
    
    # 处理缺失值
    df = df.fillna({
        'CitationCount_CrossRef': 0,
        'CitationCount_Aminer': 0,
        'AuthorKeywords': '',
        'Authors': '',
        'Title': '',
        'Conference': ''
    })
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"年份范围: {df['Year'].min().year} - {df['Year'].max().year}")
    
    return df

# ============================================================================
# Key Research Trends 分析
# ============================================================================

def analyze_research_trends(df):
    """分析研究趋势"""
    print("\n=== 分析研究趋势 ===")
    
    # 1. 年度发表量趋势
    yearly_publications = df.groupby(df['Year'].dt.year).size().reset_index(name='count')
    
    fig1 = px.line(yearly_publications, x='Year', y='count',
                   title='📈 年度发表量趋势',
                   labels={'Year': '年份', 'count': '发表数量'},
                   markers=True)
    fig1.update_layout(
        template='plotly_white',
        height=500,
        showlegend=False
    )
    fig1.show()
    
    # 2. 关键词趋势分析
    def extract_keywords(keywords_str):
        """提取关键词"""
        if pd.isna(keywords_str) or keywords_str == '':
            return []
        # 分割关键词（假设用分号或逗号分隔）
        keywords = re.split(r'[;,]\s*', str(keywords_str))
        return [kw.strip().lower() for kw in keywords if kw.strip()]
    
    # 提取所有关键词
    all_keywords = []
    for keywords in df['AuthorKeywords']:
        all_keywords.extend(extract_keywords(keywords))
    
    # 统计关键词频率
    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(15))
    
    # 关键词频率柱状图
    fig2 = px.bar(x=list(top_keywords.keys()), y=list(top_keywords.values()),
                  title='🔑 热门关键词频率',
                  labels={'x': '关键词', 'y': '出现次数'},
                  color=list(top_keywords.values()),
                  color_continuous_scale='viridis')
    fig2.update_layout(
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45
    )
    fig2.show()
    
    # 3. 关键词随时间变化趋势
    keyword_trends = {}
    for year in df['Year'].dt.year.unique():
        year_data = df[df['Year'].dt.year == year]
        year_keywords = []
        for keywords in year_data['AuthorKeywords']:
            year_keywords.extend(extract_keywords(keywords))
        keyword_trends[year] = Counter(year_keywords)
    
    # 选择前5个关键词进行趋势分析
    top_5_keywords = list(top_keywords.keys())[:5]
    
    trend_data = []
    for keyword in top_5_keywords:
        for year in sorted(keyword_trends.keys()):
            count = keyword_trends[year].get(keyword, 0)
            trend_data.append({'Year': year, 'Keyword': keyword, 'Count': count})
    
    trend_df = pd.DataFrame(trend_data)
    
    fig3 = px.line(trend_df, x='Year', y='Count', color='Keyword',
                   title='📊 关键词随时间变化趋势',
                   labels={'Year': '年份', 'Count': '出现次数', 'Keyword': '关键词'})
    fig3.update_layout(
        template='plotly_white',
        height=500
    )
    fig3.show()
    
    return yearly_publications, top_keywords, trend_df

# ============================================================================
# Key Authors 分析
# ============================================================================

def analyze_key_authors(df):
    """分析关键作者"""
    print("\n=== 分析关键作者 ===")
    
    # 1. 提取作者信息
    def extract_authors(authors_str):
        """提取作者姓名"""
        if pd.isna(authors_str) or authors_str == '':
            return []
        # 分割作者（假设用分号或逗号分隔）
        authors = re.split(r'[;,]\s*', str(authors_str))
        return [author.strip() for author in authors if author.strip()]
    
    # 统计作者发表量
    all_authors = []
    for authors in df['Authors']:
        all_authors.extend(extract_authors(authors))
    
    author_counts = Counter(all_authors)
    top_authors = dict(author_counts.most_common(20))
    
    # 作者发表量柱状图
    fig4 = px.bar(x=list(top_authors.keys()), y=list(top_authors.values()),
                  title='👥 作者发表量排名',
                  labels={'x': '作者', 'y': '发表数量'},
                  color=list(top_authors.values()),
                  color_continuous_scale='plasma')
    fig4.update_layout(
        template='plotly_white',
        height=600,
        xaxis_tickangle=-45
    )
    fig4.show()
    
    # 2. 作者引用量分析
    author_citations = {}
    for idx, row in df.iterrows():
        authors = extract_authors(row['Authors'])
        citations = row['CitationCount_CrossRef']
        for author in authors:
            if author in author_citations:
                author_citations[author] += citations
            else:
                author_citations[author] = citations
    
    # 按引用量排序
    top_cited_authors = dict(sorted(author_citations.items(), 
                                   key=lambda x: x[1], reverse=True)[:15])
    
    # 作者引用量柱状图
    fig5 = px.bar(x=list(top_cited_authors.keys()), y=list(top_cited_authors.values()),
                  title='📚 作者引用量排名',
                  labels={'x': '作者', 'y': '总引用量'},
                  color=list(top_cited_authors.values()),
                  color_continuous_scale='inferno')
    fig5.update_layout(
        template='plotly_white',
        height=600,
        xaxis_tickangle=-45
    )
    fig5.show()
    
    # 3. 作者合作网络分析
    collaboration_data = []
    for authors in df['Authors']:
        author_list = extract_authors(authors)
        if len(author_list) > 1:
            for i in range(len(author_list)):
                for j in range(i+1, len(author_list)):
                    collaboration_data.append({
                        'Author1': author_list[i],
                        'Author2': author_list[j]
                    })
    
    if collaboration_data:
        collaboration_df = pd.DataFrame(collaboration_data)
        collaboration_counts = collaboration_df.groupby(['Author1', 'Author2']).size().reset_index(name='count')
        top_collaborations = collaboration_counts.nlargest(10, 'count')
        
        # 合作网络图
        fig6 = px.scatter(top_collaborations, x='Author1', y='Author2', size='count',
                         title='🤝 作者合作网络（前10对）',
                         labels={'Author1': '作者1', 'Author2': '作者2', 'count': '合作次数'},
                         color='count',
                         color_continuous_scale='viridis')
        fig6.update_layout(
            template='plotly_white',
            height=500
        )
        fig6.show()
    
    return top_authors, top_cited_authors, collaboration_data

# ============================================================================
# 综合分析和可视化
# ============================================================================

def create_comprehensive_dashboard(df):
    """创建综合分析仪表板"""
    print("\n=== 创建综合分析仪表板 ===")
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('年度发表量趋势', '热门关键词', '作者发表量', '作者引用量'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. 年度发表量
    yearly_pubs = df.groupby(df['Year'].dt.year).size()
    fig.add_trace(
        go.Scatter(x=yearly_pubs.index, y=yearly_pubs.values, 
                  mode='lines+markers', name='发表量'),
        row=1, col=1
    )
    
    # 2. 热门关键词
    all_keywords = []
    for keywords in df['AuthorKeywords']:
        if pd.notna(keywords) and keywords != '':
            keywords_list = re.split(r'[;,]\s*', str(keywords))
            all_keywords.extend([kw.strip().lower() for kw in keywords_list if kw.strip()])
    
    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(10))
    
    fig.add_trace(
        go.Bar(x=list(top_keywords.keys()), y=list(top_keywords.values()),
               name='关键词频率'),
        row=1, col=2
    )
    
    # 3. 作者发表量
    all_authors = []
    for authors in df['Authors']:
        if pd.notna(authors) and authors != '':
            authors_list = re.split(r'[;,]\s*', str(authors))
            all_authors.extend([author.strip() for author in authors_list if author.strip()])
    
    author_counts = Counter(all_authors)
    top_authors = dict(author_counts.most_common(10))
    
    fig.add_trace(
        go.Bar(x=list(top_authors.keys()), y=list(top_authors.values()),
               name='作者发表量'),
        row=2, col=1
    )
    
    # 4. 作者引用量
    author_citations = {}
    for idx, row in df.iterrows():
        if pd.notna(row['Authors']) and row['Authors'] != '':
            authors = re.split(r'[;,]\s*', str(row['Authors']))
            citations = row['CitationCount_CrossRef']
            for author in authors:
                author = author.strip()
                if author:
                    author_citations[author] = author_citations.get(author, 0) + citations
    
    top_cited = dict(sorted(author_citations.items(), 
                           key=lambda x: x[1], reverse=True)[:10])
    
    fig.add_trace(
        go.Bar(x=list(top_cited.keys()), y=list(top_cited.values()),
               name='作者引用量'),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_layout(
        title_text="📊 研究趋势与作者分析综合仪表板",
        height=800,
        template='plotly_white',
        showlegend=False
    )
    
    # 更新x轴标签角度
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=2)
    
    fig.show()
    
    return fig

# ============================================================================
# 高级分析功能
# ============================================================================

def advanced_analysis(df):
    """高级分析功能"""
    print("\n=== 高级分析 ===")
    
    # 1. 会议分析
    conference_stats = df.groupby('Conference').agg({
        'CitationCount_CrossRef': ['mean', 'sum', 'count']
    }).round(2)
    conference_stats.columns = ['平均引用量', '总引用量', '发表数量']
    conference_stats = conference_stats.sort_values('发表数量', ascending=False)
    
    print("会议统计:")
    print(conference_stats.head(10))
    
    # 会议发表量可视化
    fig7 = px.bar(conference_stats.head(10), x=conference_stats.head(10).index, 
                  y='发表数量',
                  title='🏛️ 各会议发表量',
                  labels={'x': '会议', 'y': '发表数量'},
                  color='平均引用量',
                  color_continuous_scale='viridis')
    fig7.update_layout(
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45
    )
    fig7.show()
    
    # 2. 引用量分布分析
    fig8 = px.histogram(df, x='CitationCount_CrossRef', nbins=30,
                       title='📈 引用量分布',
                       labels={'CitationCount_CrossRef': '引用量', 'count': '论文数量'})
    fig8.update_layout(
        template='plotly_white',
        height=500
    )
    fig8.show()
    
    # 3. 高引用论文分析
    high_cited = df.nlargest(10, 'CitationCount_CrossRef')[['Title', 'Authors', 'Year', 'CitationCount_CrossRef']]
    
    print("\n高引用论文:")
    for idx, row in high_cited.iterrows():
        print(f"{row['Year'].year}: {row['Title'][:50]}... (引用量: {row['CitationCount_CrossRef']})")
    
    return conference_stats, high_cited

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("🚀 开始数据分析...")
    
    # 加载数据
    df = load_and_preprocess_data()
    
    # 研究趋势分析
    yearly_pubs, top_keywords, keyword_trends = analyze_research_trends(df)
    
    # 关键作者分析
    top_authors, top_cited_authors, collaborations = analyze_key_authors(df)
    
    # 综合仪表板
    dashboard = create_comprehensive_dashboard(df)
    
    # 高级分析
    conference_stats, high_cited_papers = advanced_analysis(df)
    
    print("\n✅ 分析完成！")
    print(f"📊 共分析了 {len(df)} 篇论文")
    print(f"👥 涉及 {len(set([author for authors in df['Authors'] for author in str(authors).split(';') if author.strip()]))} 位作者")
    print(f"🔑 包含 {len(set([kw for keywords in df['AuthorKeywords'] for kw in str(keywords).split(';') if kw.strip()]))} 个关键词")
    
    return {
        'data': df,
        'yearly_publications': yearly_pubs,
        'top_keywords': top_keywords,
        'keyword_trends': keyword_trends,
        'top_authors': top_authors,
        'top_cited_authors': top_cited_authors,
        'collaborations': collaborations,
        'conference_stats': conference_stats,
        'high_cited_papers': high_cited_papers
    }

if __name__ == "__main__":
    # 运行分析
    results = main()