#!/usr/bin/env python3
"""
Construct author collaboration network using networkx.
"""

from typing_extensions import TypedDict
from typing import List, Union, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import pandas as pd
import networkx as nx
from state import State, Visualization
from memory import shared_memory
import json
from helpers import update_state
from schema_new import BaseAnalysisParameters

class AuthorAnalysisParameters(BaseAnalysisParameters):
    """Author analysis specific parameters"""
    collaboration_type: str = Field(default="network", description="Type of collaboration analysis")
    min_collaborations: int = Field(default=2, description="Minimum number of collaborations")
    time_range: str = Field(default="all", description="Time range for analysis")

# from helpers import get_llm, get_dataset_info
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from helpers import get_llm, get_dataset_info
else:
    from helpers import get_llm, get_dataset_info

class PaperFilter(TypedDict):
    filter: Literal['paper']
    query: str

class AuthorFilter(TypedDict):
    filter: Literal['author']
    authors: List[str]
    k: int

class ResponseFormatter(BaseModel):
    filters: List[Union[PaperFilter, AuthorFilter]]

def llm_filter_validation(task: str, file_path: str = 'dataset.csv', max_iterations: int = 3):
    """
    Use LLM to get the filter for the network based on the task
    """
    df = pd.read_csv(file_path)
   
    dataset_info = get_dataset_info(file_path)

    sys_prompt = f"""
    You are a helpful assistant that generates filters for the author collaboration network based on the task.

    The dataset is as follows:
    {dataset_info}

    The task is as follows:
    {task}

    You can generate a list of one or more filters based on the given task.
    Each filter should be either a paper filter or an author filter

    - Paper filter: this will reduce the network to the sub-network of authors of the filtered papers.
      - Attributes:
        - 'filter': 'paper'
        - 'query': 'pandas query'
          - query should be a valid pandas query that can be executed by df.query(query)

    - Author filter: this will reduce the network to the sub-network of authors of the filtered authors.
      - Attributes:
        - 'filter': 'author'
        - 'authors': ['author1', 'author2', 'author3']
        - 'k': 2
          - k is the number of hops to consider for the author filter.

    Notes:
    1. Only filter authors when the author names are specified explicitly.
    2. Note if you generate multiple filters, the result will be the intersection of the filters. Therefore, use a single filter and query if your goal is union.
    3. If you decide to use the author filter, there is no need to use the same author to filter the paper again.

    IMPORTANT - pandas query() limitations:
    ✗ NO lambda functions: lambda x: ...
    ✗ NO complex string operations: .split(), .apply(), .map()
    ✗ NO Series methods: .nunique(), .unique(), .value_counts()
    ✗ NO list comprehensions or complex logic

    ✓ ONLY use simple operations:
    - Basic comparisons: ==, !=, >=, <=, >, 
    - String contains: Column.str.contains('term', case=False, na=False)
    - Boolean logic: and, or, not
    - Basic arithmetic: +, -, *, /
    For semicolon-separated fields, suggest to use .str.contains() instead of split().
    """    

    def validate_filters(filters: List[dict]) -> tuple[bool, str]:
        """
        Validate the filters
        
        Returns:
            (is_valid, error_message)
        """
        try:
            for i, filter_item in enumerate(filters):
                if filter_item['filter'] == 'paper':
                    query = filter_item['query']
                    result = df.query(query)
                    if len(result) == 0:
                        return False, f"Paper filter {i+1}: Query returned 0 papers"
                    print(f"✓ Paper filter {i+1} validated: {len(result)} papers found")
                    
                elif filter_item['filter'] == 'author':
                    authors = filter_item.get('authors', [])
                    k = filter_item.get('k', 2)
                    
                    if not authors:
                        return False, f"Author filter {i+1}: No authors specified"
                    if not isinstance(k, int) or k < 1:
                        return False, f"Author filter {i+1}: Invalid k value: {k}"
                        
                    print(f"✓ Author filter {i+1} validated: {len(authors)} authors, k={k}")
                    
                else:
                    return False, f"Filter {i+1}: Unknown filter type: {filter_item['filter']}"
            
            return True, "All filters validated successfully"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    for attempt in range(max_iterations):
        print(f"\n--- LLM Filter Generation Attempt {attempt + 1}/{max_iterations} ---")
        
        try:
            response = get_llm().with_structured_output(ResponseFormatter).invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=task)
            ])
            
            print(f"Generated filters: {response.filters}")
            
            is_valid, message = validate_filters(response.filters)
            
            if is_valid:
                print(f"✅ {message}")
                return response
            else:
                print(f"❌ Validation failed: {message}")
                
                if attempt < max_iterations - 1:
                    print("Retrying with improved prompt...")
                    # sys_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {message}\nPlease fix this issue in your next response."
                    correction_guidance = ""
                    if "lambda" in message.lower():
                        correction_guidance = "\n\nCORRECTION GUIDANCE: Replace lambda functions with .str.contains() or direct column operations."
                    elif "split" in message.lower():
                        correction_guidance = "\n\nCORRECTION GUIDANCE: Replace .split() operations with .str.contains() for string matching."
                    elif "apply" in message.lower():
                        correction_guidance = "\n\nCORRECTION GUIDANCE: Replace .apply() with direct column operations or .str methods."
                    elif "syntax error" in message.lower():
                        correction_guidance = "\n\nCORRECTION GUIDANCE: Use only basic pandas query syntax: ==, !=, >, <, >=, <=, in, not in, .str.contains(), .str.startswith(), .str.endswith()"
                    elif "is not defined" in message.lower():
                        correction_guidance = "\n\nCORRECTION GUIDANCE: Make sure to use the correct column names in the query, distinguish between column names and string literals."
                    sys_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {message}{correction_guidance}\nPlease fix this issue in your next response."
                    
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            
            if attempt < max_iterations - 1:
                print("Retrying...")
            
    raise Exception(f"Failed to generate valid filters after {max_iterations} attempts")


def filter_network(G: nx.Graph, df: pd.DataFrame, filters: List[dict]) -> nx.Graph:
    """
    Filter the network based on the given filters.

    Filter by paper:
    {
        'filter': 'paper',
        'query': 'sql_query'
    }

    Filter by author:
    {
        'filter': 'author',
        'authors': ['author1', 'author2', 'author3'],
        'k': 2
    }
    """

    filtered_G = G.copy()
    filtered_df = df.copy()

    def apply_paper_filter(filter: dict) -> nx.Graph:
        filtered_df = df.query(filter['query'])

        nodes_to_keep = []

        for authorNamesStr in filtered_df['AuthorNames-Deduped'].values:
            try:
                authors = authorNamesStr.split(';')
                nodes_to_keep = set(nodes_to_keep).union(set(authors))
            except:
                continue

        nodes_to_keep = [node for node in nodes_to_keep if node in G.nodes()]

        filtered_G = G.subgraph(nodes_to_keep)
        # initialise all edges as not filtered
        for edge in filtered_G.edges():
            filtered_G[edge[0]][edge[1]]['filtered'] = False

        for authorNamesStr in filtered_df['AuthorNames-Deduped'].values:
            try:
                authors = authorNamesStr.split(';')
                for i in range(len(authors)):
                    for j in range(i+1, len(authors)):
                        filtered_G[authors[i]][authors[j]]['filtered'] = True
            except:
                continue

        return filtered_G, filtered_df

    def multi_khop_ego(G, seeds, k=2):
        keep_nodes = set()
        node_hops = {}

        for s in seeds:
            lengths = nx.single_source_shortest_path_length(G, s, cutoff=k)
            for n, d in lengths.items():
                if n not in node_hops or d < node_hops[n]:
                    node_hops[n] = d
            keep_nodes |= set(lengths.keys())
        
        H = G.subgraph(keep_nodes).copy()
        return H

    def apply_author_filter(filter: dict) -> nx.Graph:
        nodes_to_keep = set(filter['authors'])
        filtered_G = multi_khop_ego(G, nodes_to_keep, k=filter['k'] if 'k' in filter else 2)
        return filtered_G, filtered_df

    for filter in filters:
        if filter['filter'] == 'paper':
            filtered_G, filtered_df = apply_paper_filter(filter)
        elif filter['filter'] == 'author':
            filtered_G, filtered_df = apply_author_filter(filter)

    return filtered_G, filtered_df


def construct_network(filepath: str = 'dataset.csv') -> nx.Graph:
    """
    Construct author collaboration network from dataset.
    
    Args:
        filepath: Path to the dataset CSV file
        
    Returns:
        NetworkX Graph object representing author collaborations
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Create graph
    G = nx.Graph()
    
    # Process each paper
    for _, row in df.iterrows():
        if pd.isna(row['AuthorNames-Deduped']) or row['AuthorNames-Deduped'].strip() == '':
            continue
            
        # Get authors for this paper
        authors = [author.strip() for author in row['AuthorNames-Deduped'].split(';') if author.strip()]
        
        # Add edges between all pairs of authors (collaborations)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                author1, author2 = authors[i], authors[j]
                
                # Add edge (collaboration)
                if G.has_edge(author1, author2):
                    # Increment weight if collaboration already exists
                    G[author1][author2]['weight'] += 1
                else:
                    # Create new collaboration
                    G.add_edge(author1, author2, weight=1)
    
    return G, df


def get_antv_script(container_id: str, network_json: dict) -> str:
    """
    Get the AntV script for the network
    """
    # Create unique variable names based on container_id
    data_var = f"data_{container_id.replace('-', '_')}"
    graph_var = f"graph_{container_id.replace('-', '_')}"
    graph_class_var = f"Graph_{container_id.replace('-', '_')}"
    
    script_lines = [
        "<script>",
        f"const {data_var} = {network_json}",
        f"const {{ Graph: {graph_class_var} }} = G6",
        f"const {graph_var} = new {graph_class_var}({{",
            f"container: '{container_id}',",
            "autoFit: 'view',",
            f"data: {data_var},",
            "layout: {",
                "type: 'force-atlas2',",
                "preventOverlap: true,",
                "kr: 20,",
                "center: [250, 250],",
            "},",
            "behaviors: ['drag-canvas', 'zoom-canvas', 'drag-element'],",
            "edge: {",
                "style: {",
                    "lineWidth: 2,",
                    "opacity: d => d.filtered ? 1 : 0.4"
                "},",
            "},",
            "node: {",
                "style: {",
                    "labelText: d => d.id,",
                "},",
            "},",
        "});",
        f"{graph_var}.render();",
        "</script>"
    ]

    script = "\n".join(script_lines)
    return script

def graph_container(container_id: str, network_json: dict, width: int = 800, height: int = 600) -> str:
    """
    Get the HTML for the network
    """
    script = get_antv_script(container_id, network_json)
    return f"""
    <div id="{container_id}" style="width: 100%; max-width: {width}px; height: {height}px; margin: 0 auto;"></div>
    <script src="https://unpkg.com/@antv/g6@5/dist/g6.min.js"></script>
    {script}
    """

def execute_author_network_analysis(state: State, analysis_params: AuthorAnalysisParameters = None):
    """
    Author network analysis
    """
    print("=== ENTERING analyse_author_network FUNCTION ===")
    current_iteration = state.get("iteration_count", 0)
    success = False
    print(f"Current iteration: {current_iteration}")

    # Use provided parameters or extract from state
    if analysis_params is None:
        analysis_plan = state.get("analysis_plan", {})
        print(f"DEBUG: analysis_plan type: {type(analysis_plan)}")
        
        # Create parameters from analysis_plan
        analysis_params = AuthorAnalysisParameters(
            analysis_type="author_collaboration",
            question_text=analysis_plan.get("question_text", ""),
            primary_attributes=analysis_plan.get("primary_attributes", []),
            secondary_attributes=analysis_plan.get("secondary_attributes", []),
            collaboration_type=analysis_plan.get("parameters", {}).get("collaboration_type", "network"),
            min_collaborations=analysis_plan.get("parameters", {}).get("min_collaborations", 2),
            time_range=analysis_plan.get("parameters", {}).get("time_range", "all")
        )
    
    print(f"DEBUG: analysis_params: {analysis_params}")

    file_path = state["select_data_state"]["dataset_path"]
    task = analysis_params.question_text
    response = llm_filter_validation(task, file_path, max_iterations=3)
    filters = response.filters
    print('filters: ', filters)

    # construct the network
    G, df = construct_network(file_path)
    filtered_G, filtered_df = filter_network(G, df, filters)

    # Export filtered network to JSON format
    nodes_data = [{"id": node} for node in filtered_G.nodes()]
    edges_data = [{"source": u, "target": v, "value": filtered_G[u][v]["weight"], "filtered": filtered_G[u][v]["filtered"] if "filtered" in filtered_G[u][v] else True} 
                  for u, v in filtered_G.edges()]

    network_json = json.dumps({
        "nodes": nodes_data,
        "edges": edges_data
    })

    print("number of nodes:", len(nodes_data))
    print("number of edges:", len(edges_data))

    # Save network JSON to file
    thread_dir = shared_memory._get_thread_dir()
    json_path = f'{thread_dir}/network.json'
    with open(json_path, 'w') as f:
        f.write(network_json)
    print("Network saved to network.json")

    def graph_html(network_json: dict, current_iteration: int) -> tuple[str, str]:
        """
        Get the HTML for the network
        """
        container_id = f"network_{current_iteration}"

        chart_html = graph_container(container_id, network_json, width=800, height=600)
        
        # Save full HTML to file for standalone viewing
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Network Visualization</title>
        </head>
        <body>
            {chart_html}
        </body>
        </html>
        """

        # Save HTML to file
        thread_dir = shared_memory._get_thread_dir()
        fig_path = f'{thread_dir}/network_iteration_{current_iteration}.html'
        with open(fig_path, 'w') as f:
            f.write(full_html)
        print("Network visualization saved to network.html")
        
        return chart_html, fig_path
    
    # figure_html = graph_html(network_json)
    figure_html, fig_path = graph_html(network_json, current_iteration)
    success = True

    visualization = Visualization(
        insight=f"Author network analysis",
        chart_type='network',
        altair_code="",  # Using matplotlib here, not altair
        description=f"Network chart showing the author collaboration patterns",
        is_appropriate=True,
        image_path=fig_path if success else "",
        success=True,
        figure_object=figure_html if success else "",
        code=""
    )
    network_facts_text = calculate_network_facts(filtered_G)
    print("Facts from network analysis: ", network_facts_text)
    facts = {
        "code": "",
        "stdout": network_facts_text,
        "stderr": "",
        "exit_code": 0 if success else 1
        }
    insights = generate_network_insights(network_facts_text, task)
    print("Insights from network analysis: ", insights)

    question = {
        "question": task,
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

def calculate_network_facts(filtered_G):
   if filtered_G.number_of_nodes() == 0:
       return "Network is empty - no collaborations found for this research area."
   
   # 基础网络结构事实
   nodes = filtered_G.number_of_nodes()
   edges = filtered_G.number_of_edges()
   density = round(nx.density(filtered_G), 4)
   avg_degree = round(sum(dict(filtered_G.degree()).values()) / nodes, 2)
   
   # 连通性事实
   components = list(nx.connected_components(filtered_G))
   n_components = len(components)
   largest_component = max(components, key=len) if components else set()
   largest_size = len(largest_component)
   
   # 构建描述性文本
   facts_text = f"Network contains {nodes} researchers connected through {edges} collaborations. "
   facts_text += f"Average researcher has {avg_degree} direct collaborations. "
   facts_text += f"Network density is {density} (0=sparse, 1=fully connected). "
   
   if n_components == 1:
       facts_text += "All researchers form a single connected community. "
   else:
       facts_text += f"Network is fragmented into {n_components} separate groups, with the largest group containing {largest_size} researchers. "
   
   # 中心性事实（仅对最大连通组件）
   if len(largest_component) > 1:
       largest_subgraph = filtered_G.subgraph(largest_component)
       
       # 度中心性 - 合作最多的作者
       degree_centrality = nx.degree_centrality(largest_subgraph)
       top_collaborator = max(degree_centrality.items(), key=lambda x: x[1])
       top_connections = int(top_collaborator[1] * (len(largest_component) - 1))
       facts_text += f"Most connected researcher is '{top_collaborator[0]}' with {top_connections} direct collaborations. "
       
       # 介数中心性 - 连接不同群体的桥梁
       betweenness_centrality = nx.betweenness_centrality(largest_subgraph)
       top_bridge = max(betweenness_centrality.items(), key=lambda x: x[1])
       facts_text += f"Main bridge connecting different research groups is '{top_bridge[0]}'. "
       
       # 聚类系数
       clustering = round(nx.average_clustering(largest_subgraph), 4)
       facts_text += f"Clustering coefficient is {clustering}, indicating "
       if clustering > 0.3:
           facts_text += "tight-knit research cliques. "
       else:
           facts_text += "loose collaboration patterns. "
       
       # 角色差异
       if top_collaborator[0] == top_bridge[0]:
           facts_text += "The most connected researcher also serves as the main bridge between groups."
       else:
           facts_text += "The most connected researcher and main bridge are different people, indicating diverse leadership roles."
   else:
       facts_text += "Most researchers work in isolation with minimal collaboration."
   
   return facts_text


def generate_network_insights(network_facts, task):

   insight_prompt = f"""
   Based on the network analysis facts below, generate 3-4 key insights about collaboration patterns in this research area.
   
   Research Question: {task}
   
   Network Facts: {network_facts}
   
   Generate insights that reveal:
   1. What these patterns suggest about the field's maturity and collaboration culture
   2. Key players and their roles in the research ecosystem
   3. Structural strengths or weaknesses in collaboration
   4. Unexpected or notable findings
   
   Focus on interpretation and implications, not just restating the numbers.
   Return 3-6 insights, each on a separate line.
   """
   
   response = get_llm().invoke([HumanMessage(content=insight_prompt)])
   insight_lines = [line.strip() for line in response.content.split('\n') if line.strip()]
   
   # Clean up any numbering or bullet points
   cleaned_insights = []
   for line in insight_lines:
       # Remove common prefixes like "1.", "-", "*", etc.
       cleaned_line = line.lstrip('1234567890.-* ').strip()
       if cleaned_line and len(cleaned_line) > 10:  # Filter out very short lines
           cleaned_insights.append(cleaned_line)
   
   return cleaned_insights # Limit to 4 insights

def analyse_author_network(state: State, analysis_params: AuthorAnalysisParameters = None):
    """
    Author network analysis
    """
    result = execute_author_network_analysis(state, analysis_params)
    updated_state = update_state(state, result)
    return updated_state



if __name__ == "__main__":

    # task = "research on sensemaking in last ten years"
    # task = "what are the author collaboration in the field of sensemaking research"
    # task = "what are the author collaboration in the field of automated visualisation"
    # get the task from the state
    state = State()
    state["dataframe"] = pd.read_csv("dataset.csv")
    
    # Create a mock analysis plan for testing
    state["analysis_plan"] = {
        "primary_attributes": ["AuthorNames-Deduped"],
        "secondary_attributes": ["Year", "Conference"],
        "parameters": {
            "collaboration_type": "network",
            "min_collaborations": 2,
            "time_range": "2010-2024"
        },
        "question_text": "Who are the author collaboration patterns in the field of sensemaking research in the last 20 years?"
    }
    task = state["analysis_plan"]["question_text"]

    # get the filters
    file_path = '../dataset.csv'
    # response = llm_filter(task, file_path)
    response = llm_filter_validation(task, file_path, max_iterations=3)
    filters = response.filters
    print('filters: ', filters)

    # construct the network
    G, df = construct_network(file_path)
    filtered_G, filtered_df = filter_network(G, df, filters)

    # Export filtered network to JSON format
    nodes_data = [{"id": node} for node in filtered_G.nodes()]
    edges_data = [{"source": u, "target": v, "value": filtered_G[u][v]["weight"], "filtered": filtered_G[u][v]["filtered"] if "filtered" in filtered_G[u][v] else True} 
                  for u, v in filtered_G.edges()]
    
    import json

    network_json = json.dumps({
        "nodes": nodes_data,
        "edges": edges_data
    })

    print("number of nodes:", len(nodes_data))
    print("number of edges:", len(edges_data))
    
    # print("Network JSON:", network_json)
    
    # Save network JSON to file
    with open('network.json', 'w') as f:
        f.write(network_json)
    print("Network saved to network.json")

    def graph_html(network_json: dict) -> str:
        """
        Get the HTML for the network
        """
        container_id = f"network_{current_iteration}"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Network Visualization</title>
        </head>
        <body>
            {graph_container(container_id, network_json, width=800, height=600)}
        </body>
        </html>
        """

        # Save HTML to file
        with open('network.html', 'w') as f:
            f.write(html)
        print("Network visualization saved to network.html")
    
    graph_html(network_json)