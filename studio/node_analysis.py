import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import base64
from io import BytesIO
from typing import Dict, Any, List, TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from helpers import get_llm
from collections import Counter
import plotly.graph_objects as go
import plotly.offline as pyo

# TODO: add new data analysis functions for analysing the data + produce vis + produce narrative

class AnalysisAgentState(TypedDict):
    task: str
    data: pd.DataFrame
    analysis_result: str
    messages: List[AIMessage]


# Define tools outside the class to avoid issues
@tool
def analyze_publication_trends(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze publication trends over time by conference

    Args:
        data_dict: Dictionary containing 'data' (DataFrame) and 'domain' (str)
    """
    data = pd.DataFrame(data_dict['data'])
    domain = data_dict['domain']

    if data is None or len(data) == 0:
        return {"error": "No data provided for analysis"}

    # Check required columns
    if 'Year' not in data.columns or 'Conference' not in data.columns:
        return {"error": "Required columns 'Year' or 'Conference' not found"}

    # Group data by year and conference
    trend_data = data.groupby(['Year', 'Conference']).size().reset_index(name='count')

    # Generate summary statistics
    total_pubs = len(data)
    year_range = [int(data['Year'].min()), int(data['Year'].max())]
    conferences = data['Conference'].unique()
    top_conferences = data['Conference'].value_counts().head(3).to_dict()

    # Yearly totals for trend analysis
    yearly_totals = data.groupby('Year').size()
    peak_year = yearly_totals.idxmax()
    peak_count = yearly_totals.max()

    return {
        "trend_data": trend_data.to_dict('records'),
        "summary": {
            "total_publications": total_pubs,
            "year_range": year_range,
            "conferences_count": len(conferences),
            "top_conferences": top_conferences,
            "peak_year": int(peak_year),
            "peak_publications": int(peak_count)
        }
    }


@tool
def create_trend_visualization(viz_input: Dict[str, Any]) -> Dict[str, Any]:
    """Create publication trend visualization

    Args:
        viz_input: Dictionary containing 'trend_data' (list) and 'domain' (str)
    """
    trend_data = viz_input['trend_data']
    domain = viz_input['domain']

    if not trend_data:
        return {"error": "No trend data provided"}

    # Convert to DataFrame
    df = pd.DataFrame(trend_data)

    plt.style.use('default')
    sns.set_palette("husl")

    # Create the chart
    plt.figure(figsize=(14, 8))

    # Plot lines for each conference
    conferences = df['Conference'].unique()
    colors = plt.cm.Set1(range(len(conferences)))

    for i, conference in enumerate(conferences):
        conf_data = df[df['Conference'] == conference]
        plt.plot(conf_data['Year'], conf_data['count'],
                 marker='o', label=conference, linewidth=2.5,
                 markersize=6, color=colors[i])

    # Styling
    plt.title(f'Development of {domain.title()} Publications by Conference',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Publications', fontsize=12, fontweight='bold')

    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Grid and styling
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("img_create_trend_visualization.png", dpi=300, bbox_inches='tight', facecolor='white')

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Generate reproducible code
    chart_code = f'''
# Python code to reproduce the publication trend chart
import pandas as pd
import matplotlib.pyplot as plt

def create_publication_trend_chart(trend_data, domain="{domain}"):
    df = pd.DataFrame(trend_data)

    plt.figure(figsize=(14, 8))
    conferences = df['Conference'].unique()
    colors = plt.cm.Set1(range(len(conferences)))

    for i, conference in enumerate(conferences):
        conf_data = df[df['Conference'] == conference]
        plt.plot(conf_data['Year'], conf_data['count'], 
                marker='o', label=conference, linewidth=2.5, 
                markersize=6, color=colors[i])

    plt.title(f'Development of {{domain.title()}} Publications by Conference', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Publications', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
'''

    return {
        "image_base64": image_base64,
        "chart_code": chart_code,
        "description": f"Publication trend chart showing development of {domain} research across {len(conferences)} conferences"
    }


@tool
def generate_analysis_narrative(narrative_input: Dict[str, Any]) -> str:
    """Generate narrative analysis using LLM

    Args:
        narrative_input: Dictionary containing 'trend_summary' (dict) and 'domain' (str)
    """
    trend_summary = narrative_input['trend_summary']

    domain = narrative_input['domain']

    llm = get_llm(max_completion_tokens=1024)

    prompt = f"""
Based on the publication trend analysis for "{domain}", write a comprehensive narrative analysis.

Data Summary:
- Total publications: {trend_summary['total_publications']}
- Year range: {trend_summary['year_range'][0]}-{trend_summary['year_range'][1]}
- Number of conferences: {trend_summary['conferences_count']}
- Peak year: {trend_summary['peak_year']} with {trend_summary['peak_publications']} publications
- Top conferences: {trend_summary['top_conferences']}

Write a concise summary to describe (not discuss) results, covering:
1. Overall development pattern and growth trends
2. Key conferences driving research in this field
3. Notable changes or shifts over time
4. Implications for the field's evolution

Keep the analysis academic but accessible, focusing on insights valuable for researchers.
"""

    try:
        response = llm.invoke([
            SystemMessage(
                content="You are an expert academic data analyst specializing in publication trend analysis."),
            HumanMessage(content=prompt)
        ])

        return response.content.strip()

    except Exception as e:
        # Fallback narrative
        return generate_fallback_narrative(trend_summary, domain)


def generate_fallback_narrative(summary: Dict, domain: str) -> str:
    """Generate basic narrative when LLM fails"""

    total_pubs = summary['total_publications']
    year_range = summary['year_range']
    peak_year = summary['peak_year']
    top_conferences = summary['top_conferences']

    narrative = f"""
This analysis examines the development of {domain} research from {year_range[0]} to {year_range[1]}, 
based on {total_pubs} publications across multiple venues.

The research field shows activity across {summary['conferences_count']} different conferences, 
with peak publication activity occurring in {peak_year}. The leading venues include {', '.join(list(top_conferences.keys())[:3])}, 
which together represent a significant portion of the published work.

The temporal distribution of publications provides insights into the field's evolution and growth patterns, 
reflecting both emerging research interests and the maturation of key concepts in {domain}.

This trend analysis serves as a foundation for understanding the research landscape and identifying 
opportunities for future work in this important area of visualization research.
"""
    return narrative.strip()


class AnalysisTools:
    """Collection of analysis tools for the agent"""

    @staticmethod
    def get_tools():
        """Return list of available tools"""
        return [
            # analyze_publication_trends,
            # create_trend_visualization,
            # generate_analysis_narrative,
            # analyze_author_collaboration,
            # create_collaboration_network,
            # generate_collaboration_narrative,
            # Complete analysis pipelines (main tools)
            complete_publication_trends_analysis,
            complete_author_collaboration_analysis
        ]


@tool
def analyze_author_collaboration(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze author collaboration patterns and network structure

    Args:
        data_dict: Dictionary containing 'data' (DataFrame records) and 'domain' (str)
    """
    data_records = data_dict['data']
    domain = data_dict['domain']

    # Convert back to DataFrame
    data = pd.DataFrame(data_records)

    if data is None or len(data) == 0:
        return {"error": "No data provided for collaboration analysis"}

    # Check if AuthorNames column exists
    if 'AuthorNames' not in data.columns:
        return {"error": "AuthorNames column not found in data"}

    # Parse author collaborations
    collaboration_pairs = Counter()
    author_paper_counts = Counter()
    total_papers = len(data)
    multi_author_papers = 0

    for idx, authors_str in data['AuthorNames'].items():
        if pd.notna(authors_str) and isinstance(authors_str, str):
            # Split authors by semicolon and clean whitespace
            authors = [a.strip() for a in authors_str.split(';') if a.strip()]

            # Count papers per author
            for author in authors:
                author_paper_counts[author] += 1

            # Count collaborations (papers with multiple authors)
            if len(authors) > 1:
                multi_author_papers += 1
                # Add all pairs of collaborators
                for i, author1 in enumerate(authors):
                    for author2 in authors[i + 1:]:
                        pair = tuple(sorted([author1, author2]))
                        collaboration_pairs[pair] += 1

    # Calculate network statistics
    total_authors = len(author_paper_counts)
    collaboration_rate = multi_author_papers / total_papers if total_papers > 0 else 0

    # Get top authors by publication count
    top_authors = dict(author_paper_counts.most_common(10))

    # Get strong collaborations (2+ papers together)
    strong_collaborations = {pair: count for pair, count in collaboration_pairs.items() if count >= 2}

    # Calculate average papers per author
    avg_papers_per_author = sum(author_paper_counts.values()) / len(author_paper_counts) if author_paper_counts else 0

    # Calculate average collaborators per author
    author_collaborator_counts = {}
    for (author1, author2), count in collaboration_pairs.items():
        author_collaborator_counts[author1] = author_collaborator_counts.get(author1, 0) + 1
        author_collaborator_counts[author2] = author_collaborator_counts.get(author2, 0) + 1

    avg_collaborators = sum(author_collaborator_counts.values()) / len(
        author_collaborator_counts) if author_collaborator_counts else 0

    # Find most prolific collaborators
    frequent_collaborators = dict(Counter(collaboration_pairs).most_common(5))

    # Convert tuple keys to strings for LangChain compatibility
    collaboration_data_str = {}
    for (author1, author2), count in collaboration_pairs.items():
        key_str = f"{author1}|{author2}"
        collaboration_data_str[key_str] = count

    strong_collaborations_str = {}
    for (author1, author2), count in strong_collaborations.items():
        key_str = f"{author1}|{author2}"
        strong_collaborations_str[key_str] = count

    frequent_collaborators_str = {}
    for (author1, author2), count in frequent_collaborators.items():
        key_str = f"{author1}|{author2}"
        frequent_collaborators_str[key_str] = count

    collaboration_summary = {
        "total_authors": total_authors,
        "total_papers": total_papers,
        "multi_author_papers": multi_author_papers,
        "collaboration_rate": round(collaboration_rate, 3),
        "total_collaborations": len(collaboration_pairs),
        "strong_collaborations": len(strong_collaborations),
        "avg_papers_per_author": round(avg_papers_per_author, 2),
        "avg_collaborators_per_author": round(avg_collaborators, 2),
        "top_authors": top_authors,
        "frequent_collaborators": frequent_collaborators_str,
        "domain": domain
    }

    return {
        "collaboration_data": collaboration_data_str,
        "strong_collaborations": strong_collaborations_str,
        "author_stats": dict(author_paper_counts),
        "summary": collaboration_summary
    }


@tool
def create_collaboration_network(collab_input: Dict[str, Any]) -> Dict[str, Any]:
    """Create interactive author collaboration network visualization using Plotly

    Args:
        collab_input: Dictionary containing collaboration data and domain
    """
    collaboration_data_str = collab_input['collaboration_data']
    domain = collab_input['domain']
    author_stats = collab_input.get('author_stats', {})

    if not collaboration_data_str:
        return {"error": "No collaboration data provided"}

    # 将字符串key转换回tuple用于NetworkX
    collaboration_data = {}
    for key_str, count in collaboration_data_str.items():
        try:
            author1, author2 = key_str.split('|')
            collaboration_data[(author1, author2)] = count
        except ValueError:
            continue

    if not collaboration_data:
        return {"error": "No valid collaboration pairs found"}

    # 只选择前20个最强的协作关系
    top_collaborations = dict(sorted(collaboration_data.items(), key=lambda x: x[1], reverse=True)[:30])

    # 创建NetworkX图
    G = nx.Graph()
    for (author1, author2), weight in top_collaborations.items():
        G.add_edge(author1, author2, weight=weight)

    if len(G.nodes()) == 0:
        return {"error": "No collaboration network found in top 30 relationships"}

    # 计算布局
    # pos = nx.spring_layout(G, k=2, iterations=50)
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

    # 计算网络统计
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    density = nx.density(G)
    components = list(nx.connected_components(G))
    largest_component_size = len(max(components, key=len)) if components else 0

    # 计算中心性
    try:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        top_degree_authors = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_betweenness_authors = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    except:
        top_degree_authors = []
        top_betweenness_authors = []

    # 使用Plotly创建交互式图表
    import plotly.graph_objects as go
    import plotly.offline as pyo

    # 准备边的坐标
    edge_x = []
    edge_y = []
    edge_info = []

    for (author1, author2) in G.edges():
        x0, y0 = pos[author1]
        x1, y1 = pos[author2]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = G[author1][author2]['weight']
        edge_info.append(f"{author1} ↔ {author2}: {weight} collaborations")

    # 创建边
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='lightgray'),
        hoverinfo='none',
        mode='lines'
    )

    # 准备节点数据
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # 节点信息
        papers = author_stats.get(node, 0)
        collaborations = len([n for n in G.neighbors(node)])

        node_text.append(f"{node}<br>Papers: {papers}<br>Collaborators: {collaborations}")
        node_size.append(max(papers * 10, 20))  # 根据论文数调整大小
        node_color.append(papers)  # 颜色深度表示论文数

    # 创建节点
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[name.split()[-1] if len(name.split()) > 1 else name for name in G.nodes()],  # 只显示姓氏
        textposition="middle center",
        textfont=dict(size=8, color="white"),
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Publications", thickness=15, x=1.02),
            line=dict(width=2, color='white')
        )
    )

    # 创建图表
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Author Collaboration Network - {domain.title()}<br>Top 20 Collaborations ({num_nodes} authors, {num_edges} connections)',
                        title_font_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=80),
                        annotations=[dict(
                            text=f"Network density: {density:.3f} | Connected components: {len(components)}",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            font=dict(color="gray", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                    ))

    # 生成HTML字符串
    html_string = pyo.plot(fig, output_type='div', include_plotlyjs=True)

    with open("img_collaboration_network.html", "w", encoding='utf-8') as f:
        f.write(f"<html><head><title>Collaboration Network</title></head><body>{html_string}</body></html>")

    # 生成Plotly代码
    chart_code = f'''
# Python code to reproduce the interactive collaboration network
import plotly.graph_objects as go
import networkx as nx

def create_interactive_collaboration_network(collaboration_data, author_stats, domain="{domain}"):
    # Only use top 20 collaborations
    top_collaborations = dict(sorted(collaboration_data.items(), key=lambda x: x[1], reverse=True)[:20])

    G = nx.Graph()
    for (author1, author2), weight in top_collaborations.items():
        G.add_edge(author1, author2, weight=weight)

    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create edges
    edge_x, edge_y = [], []
    for (author1, author2) in G.edges():
        x0, y0 = pos[author1]
        x1, y1 = pos[author2]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                           line=dict(width=2, color='lightgray'), hoverinfo='none')

    # Create nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=[max(author_stats.get(node, 1)*10, 20) for node in G.nodes()],
                   color=[author_stats.get(node, 0) for node in G.nodes()],
                   colorscale='Viridis', showscale=True),
        text=list(G.nodes()), textposition="middle center"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=f'Author Collaboration Network - {{domain}}')
    fig.show()

    return fig
'''

    network_stats = {
        "nodes": num_nodes,
        "edges": num_edges,
        "density": round(density, 4),
        "connected_components": len(components),
        "largest_component_size": largest_component_size,
        "top_degree_authors": top_degree_authors,
        "top_betweenness_authors": top_betweenness_authors,
        "total_collaborations_shown": 20
    }

    return {
        "interactive_html": html_string,
        "chart_code": chart_code,
        "network_stats": network_stats,
        "description": f"Interactive author collaboration network for {domain} research showing top 20 collaboration relationships among {num_nodes} authors",
        "chart_type": "interactive_collaboration_network"
    }


@tool
def generate_collaboration_narrative(narrative_input: Dict[str, Any]) -> str:
    """Generate narrative analysis for author collaboration patterns

    Args:
        narrative_input: Dictionary containing collaboration summary and domain
    """
    collab_summary = narrative_input['collaboration_summary']
    domain = narrative_input['domain']
    network_stats = narrative_input.get('network_stats', {})

    llm = get_llm(max_completion_tokens=1024)

    prompt = f"""
Based on the author collaboration analysis for "{domain}", write a comprehensive narrative analysis.

Collaboration Summary:
- Total authors: {collab_summary['total_authors']}
- Total papers: {collab_summary['total_papers']}
- Multi-author papers: {collab_summary['multi_author_papers']}
- Collaboration rate: {collab_summary['collaboration_rate']} ({collab_summary['multi_author_papers']}/{collab_summary['total_papers']} papers)
- Total collaborations: {collab_summary['total_collaborations']}
- Strong collaborations (2+ papers): {collab_summary['strong_collaborations']}
- Average papers per author: {collab_summary['avg_papers_per_author']}
- Average collaborators per author: {collab_summary['avg_collaborators_per_author']}
- Top authors: {collab_summary['top_authors']}

Network Statistics:
- Network density: {network_stats.get('density', 'N/A')}
- Connected components: {network_stats.get('connected_components', 'N/A')}
- Largest component size: {network_stats.get('largest_component_size', 'N/A')}

Write one paragraph result description, covering:
1. Overall collaboration patterns and community structure
2. Key authors and their roles in the collaboration network
3. Network characteristics (density, fragmentation, clustering)
4. Implications for knowledge sharing and field development

Keep the analysis academic but accessible, focusing on insights valuable for understanding research collaboration dynamics.
"""

    try:
        response = llm.invoke([
            SystemMessage(
                content="You are an expert academic data analyst specializing in collaboration network analysis."),
            HumanMessage(content=prompt)
        ])

        return response.content.strip()

    except Exception as e:
        # Fallback narrative
        return generate_collaboration_fallback_narrative(collab_summary, domain, network_stats)


def generate_collaboration_fallback_narrative(summary: Dict, domain: str, network_stats: Dict) -> str:
    """Generate basic collaboration narrative when LLM fails"""

    total_authors = summary['total_authors']
    collaboration_rate = summary['collaboration_rate']
    strong_collabs = summary['strong_collaborations']
    top_authors = list(summary['top_authors'].keys())[:3]

    narrative = f"""
The author collaboration analysis for {domain} research reveals a community of {total_authors} researchers 
with a collaboration rate of {collaboration_rate:.1%}, indicating that {collaboration_rate:.1%} of papers 
involve multiple authors working together.

The network shows {strong_collabs} strong collaboration relationships where authors have co-authored 
multiple papers together, suggesting the presence of established research partnerships. Leading authors 
in this field include {', '.join(top_authors)}, who have made significant contributions through both 
individual work and collaborative efforts.

The collaboration network structure provides insights into knowledge flow and research coordination 
within the {domain} community. The presence of multiple connected components suggests both concentrated 
research groups and opportunities for increased cross-group collaboration.

These collaboration patterns reflect the interdisciplinary nature of {domain} research and highlight 
the importance of teamwork in advancing the field's theoretical and practical contributions.
"""
    return narrative.strip()


def generate_fallback_narrative(summary: Dict, domain: str) -> str:
    """Generate basic narrative when LLM fails"""

    total_pubs = summary['total_publications']
    year_range = summary['year_range']
    peak_year = summary['peak_year']
    top_conferences = summary['top_conferences']

    narrative = f"""
This analysis examines the development of {domain} research from {year_range[0]} to {year_range[1]}, 
based on {total_pubs} publications across multiple venues.

The research field shows activity across {summary['conferences_count']} different conferences, 
with peak publication activity occurring in {peak_year}. The leading venues include {', '.join(list(top_conferences.keys())[:3])}, 
which together represent a significant portion of the published work.

The temporal distribution of publications provides insights into the field's evolution and growth patterns, 
reflecting both emerging research interests and the maturation of key concepts in {domain}.

This trend analysis serves as a foundation for understanding the research landscape and identifying 
opportunities for future work in this important area of visualization research.
"""
    return narrative.strip()


@tool
# def complete_publication_trends_analysis(data_dict: Dict[str, Any]) -> Dict[str, Any]:
def complete_publication_trends_analysis() -> Dict[str, Any]:
    """Complete publication trends analysis pipeline: data analysis → visualization → narrative
    """
    try:
        data_dict = get_agent_data_context()
        # Step 1: Analyze publication trends
        trends_result = analyze_publication_trends.invoke({"data_dict": data_dict})

        if "error" in trends_result:
            return {"error": f"Trends analysis failed: {trends_result['error']}"}

        # Step 2: Create visualization
        viz_result = create_trend_visualization.invoke({
            "viz_input": {
                "trend_data": trends_result["trend_data"],
                "domain": data_dict["domain"]
            }
        })

        if "error" in viz_result:
            return {"error": f"Visualization failed: {viz_result['error']}"}

        # Step 3: Generate narrative
        narrative = generate_analysis_narrative.invoke({
            "narrative_input": {
                "trend_summary": trends_result["summary"],
                "domain": data_dict["domain"]
            }
        })

        # Package complete results
        analysis_results = {
            "analysis": "publication_trends",
            "visualization": {
                "image_base64": viz_result["image_base64"],
                "chart_code": viz_result["chart_code"],
                "chart_type": "publication_trend_by_conference",
                "description": viz_result["description"]
            },
            "narrative": narrative,
            "data_insights": trends_result["summary"],
            "analysis_type": "publication_trends"
        }

        return analysis_results

    except Exception as e:
        return {"error": f"Publication trends analysis failed: {str(e)}"}


@tool
# def complete_author_collaboration_analysis(data_dict: Dict[str, Any]) -> Dict[str, Any]:
def complete_author_collaboration_analysis() -> Dict[str, Any]:
    """Complete author collaboration analysis pipeline: data analysis → visualization → narrative
    """
    try:
        data_dict = get_agent_data_context()
        # Step 1: Analyze author collaboration
        collab_result = analyze_author_collaboration.invoke({"data_dict": data_dict})

        if "error" in collab_result:
            return {"error": f"Collaboration analysis failed: {collab_result['error']}"}

        # Step 2: Create network visualization
        network_result = create_collaboration_network.invoke({
            "collab_input": {
                "collaboration_data": collab_result["collaboration_data"],
                "domain": data_dict["domain"],
                "author_stats": collab_result["author_stats"]
            }
        })

        if "error" in network_result:
            return {"error": f"Network visualization failed: {network_result['error']}"}

        # Step 3: Generate narrative
        collab_narrative = generate_collaboration_narrative.invoke({
            "narrative_input": {
                "collaboration_summary": collab_result["summary"],
                "domain": data_dict["domain"],
                "network_stats": network_result["network_stats"]
            }
        })

        # Package complete results
        analysis_results = {
            "analysis": "author_collaboration",
            "visualization": {
                "interactive_html": network_result["interactive_html"],
                "chart_code": network_result["chart_code"],
                "chart_type": "collaboration_network",
                "description": network_result["description"]
            },
            "narrative": collab_narrative,
            "data_insights": collab_result["summary"],
            "network_stats": network_result["network_stats"],
            "analysis_type": "author_collaboration"
        }

        return analysis_results

    except Exception as e:
        return {"error": f"Author collaboration analysis failed: {str(e)}"}

# Context management for data sharing between agent and tools
_agent_data_context = None


def set_agent_data_context(data_dict: Dict[str, Any]):
    """Set the data context for agent tools"""
    global _agent_data_context
    _agent_data_context = data_dict


def get_agent_data_context() -> Dict[str, Any]:
    """Get the data context for agent tools"""
    global _agent_data_context
    if _agent_data_context is None:
        raise ValueError("Agent data context not set")
    return _agent_data_context
# -------------------------
# -------------------------
# -------------------------
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from helpers import get_llm


def analysis_tool_schemas() -> List[Dict[str, Any]]:
    """Define analysis tools in OpenAI Function Calling format"""
    return [
        {
            "type": "function",
            "function": {
                "name": "complete_publication_trends_analysis",
                "description": "Complete publication trends analysis pipeline including data analysis, trend visualization, and narrative generation. Analyzes publication patterns over time by conference.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "complete_author_collaboration_analysis",
                "description": "Complete author collaboration analysis pipeline including network analysis, interactive visualization, and collaboration insights generation. Analyzes co-authorship patterns and research networks.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_analysis",
                "description": "Submit final analysis results when all requested analyses are complete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of completed analyses"
                        }
                    },
                    "required": ["summary"]
                }
            }
        }
    ]


def discover_analysis_with_function_calls(state: AnalysisAgentState, llm, max_rounds: int = 5) -> Dict[str, Any]:
    """Main analysis discovery loop using OpenAI Function Calling"""

    task = state["task"]
    data = state["data"]

    # Extract domain
    domain = "visualization research"
    if "sensemaking" in task.lower():
        domain = "visualization for sensemaking"
    elif "storytelling" in task.lower():
        domain = "narrative visualization"
    elif "analytics" in task.lower():
        domain = "visual analytics"

    # Prepare data context
    data_dict = {
        "data": data.to_dict('records') if hasattr(data, 'to_dict') else data,
        "domain": domain
    }

    # Set global context for tools
    set_agent_data_context(data_dict)

    print(f"Analysis Agent: Starting function calling workflow")
    print(f"Analysis Agent: Domain - '{domain}'")
    print(f"Analysis Agent: Task - '{task}'")
    print(f"Analysis Agent: Data records - {len(data_dict['data'])}")

    # System message
    system = SystemMessage(content=(
        "You are an intelligent academic publication analysis agent. Your mission is to analyze user requests "
        "and execute appropriate analysis tools to provide comprehensive insights.\n"
        "\n"
        "Available analysis tools:\n"
        "• complete_publication_trends_analysis - For analyzing publication trends over time, conference patterns, development analysis\n"
        "• complete_author_collaboration_analysis - For analyzing author networks, collaboration patterns, co-authorship analysis\n"
        "\n"
        "Workflow:\n"
        "  (1) Analyze the user's request to understand what type of analysis they need\n"
        "  (2) Execute appropriate analysis tool(s) based on their request\n"
        "  (3) You can call multiple tools if the request warrants comprehensive analysis\n"
        "  (4) When all requested analyses are complete, call finalize_analysis\n"
        "\n"
        "Guidelines:\n"
        "• For questions about 'development', 'trends', 'evolution', 'over time' → use publication trends analysis\n"
        "• For questions about 'authors', 'collaboration', 'networks', 'co-authorship' → use collaboration analysis\n"
        "• For comprehensive analysis requests → use both tools\n"
        "• Each tool provides complete pipeline: data analysis + visualization + narrative\n"
        "• Tools work with academic publication data including authors, conferences, years, titles, etc.\n"
        "\n"
        "Always call finalize_analysis when you have completed the requested analyses.\n"
    ))

    # User message
    user = HumanMessage(content=f"""
Analyze this request: "{task}"

Domain: {domain}
Available data: {len(data_dict['data'])} publication records

Please execute the appropriate analysis tools to address this request comprehensively.
""")

    messages: List[Any] = [system, user]
    analysis_results = {}

    # Bind tools to LLM
    bound = llm.bind_tools(analysis_tool_schemas())

    for round_num in range(max_rounds):
        print(f"\nAnalysis Agent: Round {round_num + 1}")

        # Get AI response with tool calls
        ai: AIMessage = bound.invoke(messages)
        messages.append(ai)

        print("--- AI Response ---")
        if ai.content:
            print(ai.content)
        print("--- Tool Calls ---")
        print(ai.tool_calls if ai.tool_calls else "No tools called")

        if not ai.tool_calls:
            print("Analysis Agent: No tools called, continuing...")
            continue

        # Execute tool calls
        for tc in ai.tool_calls:
            name = tc["name"]

            # Parse arguments
            raw_args = tc.get("args") or {}
            if isinstance(raw_args, str):
                args = json.loads(raw_args)
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            print(f"Analysis Agent: Executing {name}...")

            # Execute tools
            if name == "complete_publication_trends_analysis":
                try:
                    result = complete_publication_trends_analysis.invoke({})
                    if "error" not in result:
                        analysis_results["publication_trends"] = result
                        tool_result = {"status": "success", "analysis_type": "publication_trends",
                                       "message": "Publication trends analysis completed successfully"}
                        print("Analysis Agent: ✅ Publication trends analysis completed")
                    else:
                        tool_result = {"status": "error",
                                       "message": f"Publication trends analysis failed: {result['error']}"}
                        print(f"Analysis Agent: ❌ Publication trends failed: {result['error']}")
                except Exception as e:
                    tool_result = {"status": "error", "message": f"Publication trends analysis error: {str(e)}"}
                    print(f"Analysis Agent: ❌ Publication trends error: {str(e)}")

            elif name == "complete_author_collaboration_analysis":
                try:
                    result = complete_author_collaboration_analysis.invoke({})
                    if "error" not in result:
                        analysis_results["author_collaboration"] = result
                        tool_result = {"status": "success", "analysis_type": "author_collaboration",
                                       "message": "Author collaboration analysis completed successfully"}
                        print("Analysis Agent: ✅ Author collaboration analysis completed")
                    else:
                        tool_result = {"status": "error",
                                       "message": f"Author collaboration analysis failed: {result['error']}"}
                        print(f"Analysis Agent: ❌ Author collaboration failed: {result['error']}")
                except Exception as e:
                    tool_result = {"status": "error", "message": f"Author collaboration analysis error: {str(e)}"}
                    print(f"Analysis Agent: ❌ Author collaboration error: {str(e)}")

            elif name == "finalize_analysis":
                summary = args.get("summary", "Analysis completed")
                print(f"Analysis Agent: 🎉 Finalizing with summary: {summary}")

                # Return final results
                execution_summary = {
                    "domain": domain,
                    "analyses_completed": list(analysis_results.keys()),
                    "total_analyses": len(analysis_results),
                    "agent_mode": "openai_function_calling",
                    "rounds_used": round_num + 1,
                    "final_summary": summary
                }

                final_result = {
                    "analyses": analysis_results,
                    "execution_summary": execution_summary,
                    "domain": domain
                }

                return {
                    **state,
                    "analysis_result": final_result
                }

            else:
                tool_result = {"status": "error", "message": f"Unknown tool: {name}"}
                print(f"Analysis Agent: ❌ Unknown tool: {name}")

            # Add tool message to conversation
            if name != "finalize_analysis":  # Don't add tool message for finalize
                messages.append(ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=name,
                    tool_call_id=tc["id"]
                ))

    # If we reach here, max rounds exceeded without finalization
    print(f"Analysis Agent: ⚠️ Max rounds ({max_rounds}) reached without finalization")

    if analysis_results:
        execution_summary = {
            "domain": domain,
            "analyses_completed": list(analysis_results.keys()),
            "total_analyses": len(analysis_results),
            "agent_mode": "openai_function_calling",
            "rounds_used": max_rounds,
            "final_summary": "Analysis completed but not finalized due to round limit"
        }

        final_result = {
            "analyses": analysis_results,
            "execution_summary": execution_summary,
            "domain": domain
        }

        return {
            **state,
            "analysis_result": final_result
        }
    else:
        return {
            **state,
            "analysis_result": {
                "error": "No analyses completed within round limit",
                "rounds_used": max_rounds
            }
        }


def create_analysis_agent_workflow():
    """Create analysis agent workflow using OpenAI Function Calling pattern"""

    def function_calling_analysis_node(state: AnalysisAgentState) -> AnalysisAgentState:
        """Analysis node using OpenAI Function Calling workflow"""

        try:
            # Get LLM
            llm = get_llm(max_completion_tokens=2048)

            # Run function calling workflow
            result = discover_analysis_with_function_calls(state, llm, max_rounds=5)

            print("Analysis Agent: Function calling workflow completed")
            return result

        except Exception as e:
            error_msg = f"Analysis agent workflow failed: {str(e)}"
            print(f"Analysis Agent: ❌ Critical error - {error_msg}")

            updated_state = state.copy()
            updated_state["analysis_result"] = {"error": error_msg}
            return updated_state

    # Build the LangGraph workflow
    workflow = StateGraph(AnalysisAgentState)
    workflow.add_node("function_calling_analysis", function_calling_analysis_node)
    workflow.set_entry_point("function_calling_analysis")
    workflow.add_edge("function_calling_analysis", END)

    return workflow.compile()


# Context management functions
_agent_data_context = None


def set_agent_data_context(data_dict: Dict[str, Any]):
    """Set the data context for agent tools"""
    global _agent_data_context
    _agent_data_context = data_dict


def get_agent_data_context() -> Dict[str, Any]:
    """Get the data context for agent tools"""
    global _agent_data_context
    if _agent_data_context is None:
        raise ValueError("Agent data context not set")
    return _agent_data_context


def analysis_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analysis agent node that performs data analysis and visualization"""

    analysis_tasks = state.get("user_query", [])
    dataset_url = state.get("dataset_url")
    data = state.get("processed_dataframe")
    messages = state.get("messages", [])

    if not analysis_tasks:
        err = "No analysis tasks provided"
        return {"analysis_result": {"error": err}, "final_messages": messages + [AIMessage(content=err)]}

    if not dataset_url:
        err = "No dataset URL provided"
        return {"analysis_result": {"error": err}, "final_messages": messages + [AIMessage(content=err)]}

    if data is None or len(data) == 0:
        err = "No processed data available for analysis"
        return {"analysis_result": {"error": err}, "final_messages": messages + [AIMessage(content=err)]}

    print(f"Analysis Agent Node: Starting analysis with {len(data)} records")

    # Create and run the analysis workflow
    workflow = create_analysis_agent_workflow()
    agent_input = {
        "task": analysis_tasks,
        "data": data,
        "analysis_result": ""
    }

    try:
        output = workflow.invoke(agent_input)

        # Check if analysis was successful
        if "error" in output["analysis_result"]:
            error_msg = f"Analysis failed: {output['analysis_result']['error']}"
            return {
                "analysis_result": output["analysis_result"],
                "final_messages": messages + [AIMessage(content=error_msg)]
            }

        success_msg = f"Analysis completed successfully for domain: {output['analysis_result'].get('domain', 'unknown')}"
        return {
            "analysis_result": output["analysis_result"],
            "final_messages": messages + [AIMessage(content=success_msg)]
        }

    except Exception as e:
        error_msg = f"Analysis workflow failed: {str(e)}"
        print(f"Analysis Agent Node Error: {error_msg}")
        return {
            "analysis_result": {"error": error_msg},
            "final_messages": messages + [AIMessage(content=error_msg)]
        }