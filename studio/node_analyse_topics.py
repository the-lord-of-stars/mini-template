from state import State
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from helpers import get_llm
from memory import shared_memory
import networkx as nx
from datetime import datetime
import os
from state import State, Visualization


class ToolDecision(BaseModel):
    """Schema for LLM tool selection decision"""
    tool_name: str = Field(..., description="Name of the tool to execute: top_keywords, temporal_evolution, or cooccurrence_matrix")
    reasoning: str = Field(..., description="Explanation for why this tool was selected")


def analyse_topics(state: State):
    """
    Analyse topics using LLM-based tool selection
    """
    print("=== ENTERING analyse_topics FUNCTION ===")
    current_iteration = state.get("iteration_count", 0)
    print(f"Current iteration: {current_iteration}")
    
    analysis_plan = state.get("analysis_plan", {})
    print(f"DEBUG: analysis_plan type: {type(analysis_plan)}")
    print(f"DEBUG: analysis_plan keys: {list(analysis_plan.keys()) if isinstance(analysis_plan, dict) else 'Not a dict'}")
    
    try:
        # Get LLM decision on which tool to use
        print("DEBUG: About to call get_tool_decision...")
        tool_decision = get_tool_decision(state, analysis_plan)
        
        print(f"LLM selected tool: {tool_decision.tool_name}")
        print(f"Reasoning: {tool_decision.reasoning}")
        
        # Execute the selected tool
        print(f"DEBUG: About to call execute_selected_tool with tool: {tool_decision.tool_name}")
        result = execute_selected_tool(tool_decision.tool_name, state, analysis_plan)
        
        print(f"DEBUG: execute_selected_tool returned type: {type(result)}")
        print(f"DEBUG: result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        if isinstance(result, dict):
            print(f"DEBUG: result has insights: {'insights' in result}")
            if 'insights' in result:
                print(f"DEBUG: insights value: {result['insights']}")
        
        # Update state with results
        new_state = state.copy()
        new_state["question"] = state["question"]
        
        new_state["topic_analysis_result"] = result
        
        # Extract insights from result and set them directly in new_state
        if isinstance(result, dict) and 'insights' in result:
            new_state["insights"] = result["insights"]
            print(f"DEBUG: Set insights in new_state: {result['insights']}")
        else:
            print("DEBUG: No insights found in result!")
        
        # Extract question from result if it exists
        if isinstance(result, dict) and 'question' in result:
            new_state["question"] = result["question"]
            print(f"DEBUG: Set question in new_state: {result['question']}")
        
        if isinstance(result, dict) and 'facts' in result:
            new_state["facts"] = result["facts"]
            print("DEBUG: Set facts in new_state")
        
        if isinstance(result, dict) and 'visualizations' in result:
            new_state["visualizations"] = {
                "visualizations": result["visualizations"]
            }
            print("DEBUG: Set visualizations in new_state")
        
        new_state["iteration_history"] = state.get("iteration_history", []) + [result]
        print(f"DEBUG: Updated iteration_history, length: {len(new_state['iteration_history'])}")

        # Save state to memory
        print("DEBUG: About to save state to memory...")
        shared_memory.save_state(new_state)
        print("DEBUG: State saved to memory successfully")

        print("=== EXITING analyse_topics FUNCTION SUCCESSFULLY ===")
        return new_state
        
    except Exception as e:
        print(f"ERROR in analyse_topics: {e}")
        print(f"ERROR type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e


def get_tool_decision(state: State, analysis_plan: dict) -> ToolDecision:
    """
    Use LLM to decide which analysis tool to execute based on analysis plan
    """
    # Prepare context for LLM
    context = prepare_decision_context(state, analysis_plan)
    
    sys_prompt = """
    You are an intelligent topic analysis agent. Based on the analysis plan and available data, select the most appropriate analysis tool.
    
    Available tools:
    1. top_keywords - Analyzes the most frequent keywords in the dataset
       - Use when: Need to understand overall topic distribution, find dominant themes
       - Outputs: Keyword frequency list, bar chart visualization
    
    2. temporal_evolution - Analyzes how keyword popularity changes over time
       - Use when: Need to understand topic trends, identify emerging/declining themes
       - Outputs: Time series visualization, trend analysis
    
    3. cooccurrence_matrix - Analyzes relationships between keywords
       - Use when: Need to understand topic relationships, find related themes
       - Outputs: Co-occurrence network, correlation analysis
    
    Decision criteria:
    - If the question focuses on "most common", "frequent", "dominant" topics → choose top_keywords
    - If the question focuses on "evolution", "trends", "over time", "emerging" → choose temporal_evolution  
    - If the question focuses on "relationships", "correlations", "connections" between topics → choose cooccurrence_matrix
    
    Always provide clear reasoning for your selection.
    """
    
    human_prompt = f"""
    Analysis Plan:
    {analysis_plan}
    
    Available data columns: {list(state.get('dataframe', pd.DataFrame()).columns)}
    
    Based on this analysis plan and available data, which tool should be executed?
    """
    
    llm = get_llm(temperature=0, max_tokens=1024)
    
    response = llm.with_structured_output(ToolDecision).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )
    
    return response


def prepare_decision_context(state: State, analysis_plan: dict) -> str:
    """
    Prepare context information for LLM decision
    """
    df = state.get("dataframe", pd.DataFrame())
    
    context = f"""
    Analysis Plan:
    - Primary attributes: {analysis_plan.primary_attributes if hasattr(analysis_plan, 'primary_attributes') else []}
    - Secondary attributes: {analysis_plan.secondary_attributes if hasattr(analysis_plan, 'secondary_attributes') else []}
    - Parameters: {analysis_plan.parameters if hasattr(analysis_plan, 'parameters') else {}}
    - Question: {analysis_plan.question_text if hasattr(analysis_plan, 'question_text') else ''}
    
    Data Overview:
    - Total rows: {len(df)}
    - Available columns: {list(df.columns)}
    - Has Year column: {'Year' in df.columns}
    - Has AuthorKeywords column: {'AuthorKeywords' in df.columns}
    """
    
    return context


def execute_selected_tool(tool_name: str, state: State, analysis_plan: dict) -> dict:
    """
    Execute the selected analysis tool
    """
    if tool_name == "top_keywords":
        return execute_top_keywords(state, analysis_plan)
    elif tool_name == "temporal_evolution":
        return execute_temporal_evolution(state, analysis_plan)
    elif tool_name == "cooccurrence_matrix":
        return execute_cooccurrence_matrix(state, analysis_plan)
    else:
        print(f"Unknown tool: {tool_name}")
        return {"error": f"Unknown tool: {tool_name}"}


def execute_top_keywords(state: State, analysis_plan: dict) -> dict:
    """
    1. top_keywords - 统计数据集中出现频率最高的N个关键词
    """
    print("=== Executing Top Keywords Analysis ===")

    question = state["question"].get("question", "")
    
    df = state["dataframe"]
    top_n = analysis_plan.top_n if hasattr(analysis_plan, 'top_n') else 10
    
    success = True  # 初始化success变量
    figure_html = ""
    fig_path = 'fig/top_keywords_chart.png'
    
    try:
        # Extract keywords from AuthorKeywords column
        df["keywords"] = df["AuthorKeywords"].fillna("").astype(str).apply(
            lambda x: x.split(",") if x.strip() else []
        )
        df["keywords"] = df["keywords"].apply(
            lambda x: [keyword.strip().lower() for keyword in x if keyword.strip()]
        )
        
        # Flatten all keywords and count frequency
        all_keywords = []
        for keywords_list in df["keywords"]:
            all_keywords.extend(keywords_list)
        
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(top_n)
        
        if not top_keywords:
            success = False
            print("No keywords found in the dataset")
            return state
        
        # Create bar chart with Plotly
        keywords, counts = zip(*top_keywords)
        
        fig = go.Figure(data=[
            go.Bar(
                x=keywords,
                y=counts,
                text=counts,
                textposition='auto',
                marker_color='skyblue',
                marker_line_color='navy',
                marker_line_width=1
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Most Frequent Keywords',
            xaxis_title='Keywords',
            yaxis_title='Frequency',
            xaxis_tickangle=-45,
            height=600,
            width=800,
            showlegend=False
        )
        
        # 确保目录存在
        os.makedirs('fig', exist_ok=True)
        
        # Save as HTML file
        html_path = fig_path.replace('.png', '.html')
        fig.write_html(html_path)
        
        # Save as PNG for compatibility
        fig.write_image(fig_path)
        
        # Generate HTML snippet for the figure
        figure_html = fig.to_html(
            full_html=False,           # Don't include full HTML structure
            include_plotlyjs='cdn'     # Include plotly.js from CDN
        )
        
        print(f"✅ Chart saved to {fig_path}")
        
    except Exception as e:
        print(f"❌ Error in top keywords analysis: {e}")
        success = False
        top_keywords = []
    
    # Print results
    if success and top_keywords:
        print(f"\nTop {top_n} Keywords:")
        print("-" * 50)
        for i, (keyword, count) in enumerate(top_keywords, 1):
            print(f"{i:2d}. {keyword:20s} - {count:4d} occurrences")
    
    # Create visualization object
    visualization = Visualization(
        insight=f"Generated top {top_n} keywords analysis",
        chart_type='bar_chart',
        altair_code="",  # Using matplotlib here, not altair
        description=f"Bar chart showing the top {top_n} most frequent keywords",
        is_appropriate=success,
        image_path=fig_path if success else "",
        success=success,
        figure_object=figure_html if success else "",
        code=""
    )
    
    
    # Facts - 修复语法错误
    if success and top_keywords:
        result_lines = [f"Top {top_n} Keywords:", "-" * 50]
        result_lines.extend([f"{i:2d}. {keyword:20s} - {count:4d} occurrences" 
                           for i, (keyword, count) in enumerate(top_keywords, 1)])
        result = "\n".join(result_lines)
    else:
        result = "No keywords analysis results available"
    
    facts = {
        "code": "",
        "stdout": result,
        "stderr": "",
        "exit_code": 0 if success else 1
    }
    
    # Insights
    if success and top_keywords:
        insights = [f"{keyword} appears {count} times in the dataset" 
                   for keyword, count in top_keywords[:5]]  # 取前5个作为insights
        insights.append(f"Total of {len(keyword_counts)} unique keywords found in the dataset")
    else:
        insights = ["No keyword insights available"]
    
    question = {
        "question": question,
        "handled": True,
        "spec": ""
    }
    

    current_iteration_data = {
        "question": question,
        "facts": facts,
        "insights": insights,
        "visualizations": [visualization]
    }
    
    return current_iteration_data

    


def execute_temporal_evolution(state: State, analysis_plan: dict) -> dict:
    """
    2. temporal_evolution - 分析关键词随时间的频率变化
    """
    print("=== Executing Temporal Evolution Analysis ===")
    question = state["question"].get("question", "")
    
    df = state["dataframe"]
    success = True
    figure_html = ""
    fig_path = 'fig/temporal_evolution_chart.png'
    
    try:
        if "Year" not in df.columns:
            success = False
            print("❌ Year column not found in dataset")
            return state
        
        # Extract keywords and years
        df["keywords"] = df["AuthorKeywords"].fillna("").astype(str).apply(
            lambda x: x.split(",") if x.strip() else []
        )
        df["keywords"] = df["keywords"].apply(
            lambda x: [keyword.strip().lower() for keyword in x if keyword.strip()]
        )
        
        # Create keyword-year dataframe
        keyword_year_data = []
        for _, row in df.iterrows():
            year = row["Year"]
            keywords = row["keywords"]
            for keyword in keywords:
                keyword_year_data.append({"year": year, "keyword": keyword})
        
        keyword_df = pd.DataFrame(keyword_year_data)
        
        if keyword_df.empty:
            success = False
            print("❌ No valid keyword data found")
            return state
        
        # Get top keywords overall for trend analysis
        top_keywords = keyword_df["keyword"].value_counts().head(10).index.tolist()
        
        # Calculate yearly frequency for top keywords
        yearly_freq = keyword_df[keyword_df["keyword"].isin(top_keywords)].groupby(
            ["year", "keyword"]
        ).size().reset_index(name="frequency")
        
        # Pivot for plotting
        pivot_data = yearly_freq.pivot(index="year", columns="keyword", values="frequency").fillna(0)
        
        # Create time series plot with Plotly
        fig = go.Figure()
        
        for keyword in top_keywords[:8]:  # Plot top 8 keywords for clarity
            if keyword in pivot_data.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_data.index,
                    y=pivot_data[keyword],
                    mode='lines+markers',
                    name=keyword,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title='Keyword Evolution Over Time',
            xaxis_title='Year',
            yaxis_title='Frequency',
            height=600,
            width=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # 确保目录存在
        os.makedirs('fig', exist_ok=True)
        
        # Save as HTML file
        html_path = fig_path.replace('.png', '.html')
        fig.write_html(html_path)
        
        # Save as PNG for compatibility
        fig.write_image(fig_path)
        
        # Generate HTML snippet for the figure
        figure_html = fig.to_html(
            full_html=False,           # Don't include full HTML structure
            include_plotlyjs='cdn'     # Include plotly.js from CDN
        )
        
        print(f"✅ Chart saved to {fig_path}")
        
        # Analyze trends
        trends = analyze_temporal_trends(pivot_data)
        
        # Print results
        print("\nTemporal Evolution Analysis:")
        print("-" * 50)
        print("Emerging Topics (increasing trend):")
        for topic in trends["emerging"]:
            print(f"  - {topic}")
        
        print("\nDeclining Topics (decreasing trend):")
        for topic in trends["declining"]:
            print(f"  - {topic}")
        
        print("\nStable Topics (consistent trend):")
        for topic in trends["stable"]:
            print(f"  - {topic}")
            
    except Exception as e:
        print(f"❌ Error in temporal evolution analysis: {e}")
        success = False
        trends = {"emerging": [], "declining": [], "stable": []}
        yearly_freq = pd.DataFrame()
        keyword_df = pd.DataFrame()
    
    # Create visualization object
    visualization = Visualization(
        insight="Generated temporal evolution analysis",
        chart_type='line_chart',
        altair_code="",  # Using matplotlib here, not altair
        description="Time series chart showing keyword evolution over time",
        is_appropriate=success,
        image_path=fig_path if success else "",
        success=success,
        figure_object=figure_html if success else "",
        code=""
    )
        
    # Facts
    if success and not yearly_freq.empty:
        result_lines = ["Temporal Evolution Analysis:", "-" * 50]
        result_lines.append("Emerging Topics (increasing trend):")
        for topic in trends["emerging"]:
            result_lines.append(f"  - {topic}")
        result_lines.append("\nDeclining Topics (decreasing trend):")
        for topic in trends["declining"]:
            result_lines.append(f"  - {topic}")
        result_lines.append("\nStable Topics (consistent trend):")
        for topic in trends["stable"]:
            result_lines.append(f"  - {topic}")
        result = "\n".join(result_lines)
    else:
        result = "No temporal evolution analysis results available"
    
    facts = {
        "code": "",
        "stdout": result,
        "stderr": "",
        "exit_code": 0 if success else 1
    }
    
    # Insights
    if success and trends:
        insights = []
        if trends["emerging"]:
            insights.append(f"Emerging topics: {', '.join(trends['emerging'][:3])}")
        if trends["declining"]:
            insights.append(f"Declining topics: {', '.join(trends['declining'][:3])}")
        if trends["stable"]:
            insights.append(f"Stable topics: {', '.join(trends['stable'][:3])}")
        if not insights:
            insights = ["No clear trends identified in the temporal analysis"]
    else:
        insights = ["No temporal evolution insights available"]
    
    
    # Set question field
    # new_state["question"] = {
    #     "question": question,
    #     "handled": True,
    #     "spec": ""
    # }
    question = {
        "question": question,
        "handled": True,
        "spec": ""
    }

    current_iteration_data = {
        "question": question,
        "facts": facts,
        "insights": insights,
        "visualizations": [visualization]
    }

    return current_iteration_data


def analyze_temporal_trends(pivot_data: pd.DataFrame) -> dict:
    """
    Analyze temporal trends to identify emerging, declining, and stable topics
    """
    trends = {"emerging": [], "declining": [], "stable": []}
    
    for keyword in pivot_data.columns:
        if keyword in pivot_data.columns:
            series = pivot_data[keyword]
            if len(series) < 3:
                continue
                
            # Calculate trend (simple linear regression slope)
            x = np.arange(len(series))
            y = series.values
            slope = np.polyfit(x, y, 1)[0]
            
            # Classify based on slope
            if slope > 0.5:  # Significant positive trend
                trends["emerging"].append(keyword)
            elif slope < -0.5:  # Significant negative trend
                trends["declining"].append(keyword)
            else:  # Relatively stable
                trends["stable"].append(keyword)
    
    return trends


def execute_cooccurrence_matrix(state: State, analysis_plan: dict) -> dict:
    """
    3. cooccurrence_matrix - 计算关键词共现矩阵
    """
    print("=== Executing Co-occurrence Matrix Analysis ===")
    question = state["question"].get("question", "")
    
    df = state["dataframe"]
    success = True
    figure_html = ""
    network_path = 'fig/cooccurrence_network.png'
    
    try:
        # Extract keywords
        df["keywords"] = df["AuthorKeywords"].fillna("").astype(str).apply(
            lambda x: x.split(",") if x.strip() else []
        )
        df["keywords"] = df["keywords"].apply(
            lambda x: [keyword.strip().lower() for keyword in x if keyword.strip()]
        )
        
        # Get top keywords for matrix analysis
        all_keywords = []
        for keywords_list in df["keywords"]:
            all_keywords.extend(keywords_list)
        
        if not all_keywords:
            success = False
            print("❌ No keywords found in the dataset")
            return state
        
        top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(15)]
        
        if not top_keywords:
            success = False
            print("❌ No valid keywords for co-occurrence analysis")
            return state
        
        # Create co-occurrence matrix
        cooccurrence_matrix = np.zeros((len(top_keywords), len(top_keywords)))
        
        for keywords_list in df["keywords"]:
            # Find which top keywords appear in this paper
            present_keywords = [kw for kw in keywords_list if kw in top_keywords]
            
            # Update co-occurrence matrix
            for i, kw1 in enumerate(top_keywords):
                for j, kw2 in enumerate(top_keywords):
                    if kw1 in present_keywords and kw2 in present_keywords:
                        cooccurrence_matrix[i][j] += 1
        
        # 确保目录存在
        os.makedirs('fig', exist_ok=True)
        
        # Create network graph with Plotly
        fig = create_cooccurrence_network_plotly(top_keywords, cooccurrence_matrix, network_path)
        
        # Find strongest associations
        strongest_pairs = find_strongest_associations(top_keywords, cooccurrence_matrix)
        
        # Print results
        print("\nCo-occurrence Analysis:")
        print("-" * 50)
        print("Strongest Keyword Associations:")
        for i, (pair, strength) in enumerate(strongest_pairs[:10], 1):
            print(f"{i:2d}. {pair[0]:20s} <-> {pair[1]:20s} (co-occurrence: {strength})")
        
        # Generate HTML snippet for the figure
        figure_html = fig.to_html(
            full_html=False,           # Don't include full HTML structure
            include_plotlyjs='cdn'     # Include plotly.js from CDN
        )
        
        print(f"✅ Network chart saved to {network_path}")
        
    except Exception as e:
        print(f"❌ Error in co-occurrence matrix analysis: {e}")
        success = False
        strongest_pairs = []
        top_keywords = []
        cooccurrence_matrix = np.array([])
    
    # Create visualization object
    visualization = Visualization(
        insight="Generated co-occurrence network analysis",
        chart_type='network_graph',
        altair_code="",  # Using matplotlib here, not altair
        description="Network visualization of keyword co-occurrences",
        is_appropriate=success,
        image_path=network_path if success else "",
        success=success,
        figure_object=figure_html if success else "",
        code=""
    )
        
    # Facts
    if success and strongest_pairs:
        result_lines = ["Co-occurrence Analysis:", "-" * 50]
        result_lines.append("Strongest Keyword Associations:")
        for i, (pair, strength) in enumerate(strongest_pairs[:10], 1):
            result_lines.append(f"{i:2d}. {pair[0]:20s} <-> {pair[1]:20s} (co-occurrence: {strength})")
        result = "\n".join(result_lines)
    else:
        result = "No co-occurrence analysis results available"
    
    facts = {
        "code": "",
        "stdout": result,
        "stderr": "",
        "exit_code": 0 if success else 1
    }
    
    # Insights
    if success and strongest_pairs:
        insights = []
        for i, (pair, strength) in enumerate(strongest_pairs[:5], 1):
            insights.append(f"Strong association {i}: {pair[0]} and {pair[1]} (co-occurrence: {strength})")
        insights.append(f"Analyzed co-occurrences among {len(top_keywords)} top keywords")
    else:
        insights = ["No co-occurrence network insights available"]
    
    question = {
        "question": question,
        "handled": True,
        "spec": ""
    }
    

    current_iteration_data = {
        "question": question,
        "facts": facts,
        "insights": insights,
        "visualizations": [visualization]
    }
    
    return current_iteration_data


def create_cooccurrence_network_plotly(keywords: List[str], cooccurrence_matrix: np.ndarray, network_path: str):
    """
    Create network graph from co-occurrence matrix using Plotly
    """
    G = nx.Graph()
    
    # Add nodes
    for keyword in keywords:
        G.add_node(keyword)
    
    # Add edges with weights
    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i < j and cooccurrence_matrix[i][j] > 0:  # Avoid duplicates and zero weights
                G.add_edge(kw1, kw2, weight=cooccurrence_matrix[i][j])
    
    # Get positions using spring layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Prepare data for Plotly
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    # Create edge traces
    edge_traces = []
    for i in range(0, len(edge_x), 3):
        if i + 2 < len(edge_x):
            edge_trace = go.Scatter(
                x=edge_x[i:i+3],
                y=edge_y[i:i+3],
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color='black'),
        marker=dict(
            showscale=True,
            colorscale='Blues',
            size=20,
            color=[],
            line_width=2,
            line_color='white'
        )
    )
    
    # Color nodes by degree
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                   layout=go.Layout(
                       title='Keyword Co-occurrence Network',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[dict(
                           text="Node size and color indicate degree (number of connections)",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(size=10)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    # Save as HTML file
    html_path = network_path.replace('.png', '.html')
    fig.write_html(html_path)
    
    # Save as PNG for compatibility
    fig.write_image(network_path)
    
    return fig


def create_cooccurrence_network(keywords: List[str], cooccurrence_matrix: np.ndarray):
    """
    Create network graph from co-occurrence matrix (matplotlib version - kept for compatibility)
    """
    G = nx.Graph()
    
    # Add nodes
    for keyword in keywords:
        G.add_node(keyword)
    
    # Add edges with weights
    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i < j and cooccurrence_matrix[i][j] > 0:  # Avoid duplicates and zero weights
                G.add_edge(kw1, kw2, weight=cooccurrence_matrix[i][j])
    
    # Create network visualization
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    
    # Draw edges with varying thickness based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w/2 for w in weights], 
                          alpha=0.5, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title('Keyword Co-occurrence Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('fig/cooccurrence_network.png', dpi=300, bbox_inches='tight')
    plt.close()


def find_strongest_associations(keywords: List[str], cooccurrence_matrix: np.ndarray) -> List[tuple]:
    """
    Find the strongest keyword associations
    """
    associations = []
    
    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i < j:  # Avoid duplicates
                strength = cooccurrence_matrix[i][j]
                if strength > 0:
                    associations.append(((kw1, kw2), strength))
    
    # Sort by strength (descending)
    associations.sort(key=lambda x: x[1], reverse=True)
    
    return associations


if __name__ == "__main__":
    # Test the framework
    state = State()
    state["dataframe"] = pd.read_csv("dataset.csv")
    
    # Create a mock analysis plan for testing
    state["analysis_plan"] = {
        "primary_attributes": ["AuthorKeywords"],
        "secondary_attributes": ["Year", "Conference"],
        "parameters": {
            "top_n": 10,
            "time_range": "1990-2024",
            "conference_filter": True
        },
        # "question_text": "What are correlations between frequent topics?"
        "question_text": "analyse the temporal evolution of the most frequent topics?"
    }
    
    result = analyse_topics(state)
    print("Analysis completed!")
    
    # Create HTML template to display the visualization
    if 'topic_analysis_result' in result and 'visualizations' in result['topic_analysis_result']:
        visualizations = result['topic_analysis_result']['visualizations']['visualizations']
        if visualizations and len(visualizations) > 0:
            figure_html = visualizations[0]['figure_object']
            
            html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topic Analysis Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}
        .visualization iframe {{
            border: none;
            width: 100%;
            max-width: 800px;
            height: 600px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Topic Analysis Visualization</h1>
        <div class="visualization">
            {figure_html}
        </div>
    </div>
</body>
</html>
"""
            
            # Ensure fig directory exists
            os.makedirs('fig', exist_ok=True)
            
            # Save the HTML file
            with open('fig/test.html', 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            print("✅ HTML visualization saved to fig/test.html")
        else:
            print("❌ No visualizations found in results")
    else:
        print("❌ No visualization data found in results")