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

def llm_filter(topic: str, file_path: str = 'dataset.csv', filtered_df_path: str = '', domain_knowledge: str = ''):
    """
    Use LLM to get the filter for the network based on the task
    """
   
    dataset_info = get_dataset_info(file_path)

    sys_prompt = f"""
    You are a helpful assistant that generates filters for the author collaboration network based on the task.

    The dataset is as follows:
    {dataset_info}

    The task is to explore the dataset about the topic of {topic}.

    The following knowledge may help you make the report:
    {domain_knowledge}

    The filtered dataset is at the following path:
    {filtered_df_path} if it is available, otherwise use the original dataset path: {file_path}

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
    """    

    response = get_llm().with_structured_output(ResponseFormatter).invoke(
        [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=topic)
        ]
    )
    # print('llm_filter response: ', response)
    return response


def filter_network(G: nx.Graph, filtered_df_path: str, filters: List[dict]) -> nx.Graph:
    """
    Filter the network based on the given filtered dataset.

    The filtered dataset is at the following path:
    {filtered_df_path}
    """

    filtered_G = G.copy()
    filtered_df = pd.read_csv(filtered_df_path)

    def apply_paper_filter(filter: dict) -> nx.Graph:
        # filtered_df = filtered_df.query(filter['query'])

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

    # Add author attributes after filtering
    def add_author_attributes(filtered_G, filtered_df):
        # Count filtered papers for each author
        author_paper_counts = {}
        author_institutions = {}
        
        for _, row in filtered_df.iterrows():
            try:
                authors = row['AuthorNames-Deduped'].split(';') if pd.notna(row['AuthorNames-Deduped']) else []
                affiliations = row['AuthorAffiliation'].split(';') if pd.notna(row['AuthorAffiliation']) else []
                
                # Count papers for each author
                for author in authors:
                    author = author.strip()
                    if author in filtered_G.nodes():
                        author_paper_counts[author] = author_paper_counts.get(author, 0) + 1
                
                # Extract institution for each author
                for i, author in enumerate(authors):
                    author = author.strip()
                    if author in filtered_G.nodes():
                        # Get corresponding affiliation if available
                        if i < len(affiliations):
                            institution = affiliations[i].strip()
                            # If this author already has an institution, keep the first one found
                            if author not in author_institutions:
                                author_institutions[author] = institution
                        else:
                            # If no affiliation available, set to empty string
                            if author not in author_institutions:
                                author_institutions[author] = ""
            except:
                continue
        
        # Add attributes to graph nodes
        for node in filtered_G.nodes():
            filtered_G.nodes[node]['filtered_paper_count'] = author_paper_counts.get(node, 0)
            filtered_G.nodes[node]['institution'] = author_institutions.get(node, "")
        
        return filtered_G

    # Apply the attribute addition
    filtered_G = add_author_attributes(filtered_G, filtered_df)

    return filtered_G, filtered_df

def filter_network_with_sql(G: nx.Graph, df: pd.DataFrame, filters: List[dict]) -> nx.Graph:
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
        # Create directory if it doesn't exist
        import os
        # Use a default thread_id if memory is not available
        try:
            thread_id = memory.thread_id
        except:
            thread_id = "default"

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

    # Add author attributes after filtering
    def add_author_attributes(filtered_G, filtered_df):
        # Count filtered papers for each author
        author_paper_counts = {}
        author_institutions = {}
        
        for _, row in filtered_df.iterrows():
            try:
                authors = row['AuthorNames-Deduped'].split(';') if pd.notna(row['AuthorNames-Deduped']) else []
                affiliations = row['AuthorAffiliation'].split(';') if pd.notna(row['AuthorAffiliation']) else []
                
                # Count papers for each author
                for author in authors:
                    author = author.strip()
                    if author in filtered_G.nodes():
                        author_paper_counts[author] = author_paper_counts.get(author, 0) + 1
                
                # Extract institution for each author
                for i, author in enumerate(authors):
                    author = author.strip()
                    if author in filtered_G.nodes():
                        # Get corresponding affiliation if available
                        if i < len(affiliations):
                            institution = affiliations[i].strip()
                            # If this author already has an institution, keep the first one found
                            if author not in author_institutions:
                                author_institutions[author] = institution
                        else:
                            # If no affiliation available, set to empty string
                            if author not in author_institutions:
                                author_institutions[author] = ""
            except:
                continue
        
        # Add attributes to graph nodes
        for node in filtered_G.nodes():
            filtered_G.nodes[node]['filtered_paper_count'] = author_paper_counts.get(node, 0)
            filtered_G.nodes[node]['institution'] = author_institutions.get(node, "")
        
        return filtered_G

    # Apply the attribute addition
    filtered_G = add_author_attributes(filtered_G, filtered_df)

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
    import json
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    print(f"unique_id: {unique_id}")
    data_var = f"data_{unique_id}"
    graph_var = f"graph_{unique_id}"
    
    # Create color palette for institutions
    color_palette = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2',
        '#A9CCE3', '#F9E79F', '#D5A6BD', '#A2D9CE', '#FAD7A0'
    ]
    
    script_lines = [
        "<script>",
        f"const {data_var} = {network_json}",
        "const { Graph } = G6",
        "",
        "// Color palette for institutions",
        f"const colorPalette = {color_palette};",
        "",
        "// Function to get color for institution",
        "function getInstitutionColor(institution) {",
        "    if (!institution || institution === '') return '#CCCCCC';",
        "    const hash = institution.split('').reduce((a, b) => {",
        "        a = ((a << 5) - a) + b.charCodeAt(0);",
        "        return a & a;",
        "    }, 0);",
        "    return colorPalette[Math.abs(hash) % colorPalette.length];",
        "}",
        "",
        "// Function to get node size based on paper count",
        "function getNodeSize(paperCount) {",
        "    const minSize = 40;",
        "    const maxSize = 120;",
        "    const minPapers = 1;",
        f"    const maxPapers = Math.max(...{data_var}.nodes.map(n => n.filtered_paper_count || 1));",
        "    if (paperCount <= minPapers) return minSize;",
        "    if (paperCount >= maxPapers) return maxSize;",
        "    return minSize + (paperCount - minPapers) * (maxSize - minSize) / (maxPapers - minPapers);",
        "}",
        "",
        f"const {graph_var} = new Graph(",
            "{",
            f"container: '{container_id}',",
            "autoFit: 'view',",
            f"data:{data_var},",
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
                    "size: d => getNodeSize(d.filtered_paper_count || 1),",
                    "fill: d => getInstitutionColor(d.institution || ''),",
                    "stroke: '#333333',",
                    "lineWidth: 1,",
                    "labelFontSize: 18,",
                    "fillOpacity: 0.8,",
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
    <div id="{container_id}" style="width: {width}px; height: {height}px;"></div>
    <script src="https://unpkg.com/@antv/g6@5/dist/g6.min.js"></script>
    {script}
    """


if __name__ == "__main__":
    # Test the function
    # G, df = construct_network('../../dataset.csv')
    # print(f"Network constructed with {G.number_of_nodes()} authors and {G.number_of_edges()} collaborations")

    task = "research on sensemaking in last ten years"
    file_path = '../../../dataset.csv'
    # file_path = 'dataset.csv'
    response = llm_filter(task, file_path)
    
    filters = response.filters
    print('filters: ', filters)

    G, df = construct_network(file_path)
    # filtered_G, filtered_df = filter_network(G, df, filters)
    filtered_G, filtered_df = filter_network_with_sql(G, df, filters)

    # Export filtered network to JSON format
    nodes_data = [{"id": node, "filtered_paper_count": filtered_G.nodes[node]["filtered_paper_count"], "institution": filtered_G.nodes[node]["institution"]} for node in filtered_G.nodes()]
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
        container_id = "network"

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
