#!/usr/bin/env python3
"""
Process collaboration data from filtered_dataset.csv and generate insights
for the D3.js force layout visualization.
"""

import pandas as pd
import json
import networkx as nx
from collections import Counter, defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

def load_and_process_data(filepath):
    """Load the CSV data and process it for network analysis."""
    print(f"Loading data from {filepath}...")
    
    # Load CSV
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} papers")
    
    # Clean and process data
    df['AuthorNames-Deduped'] = df['AuthorNames-Deduped'].fillna('')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['AminerCitationCount'] = pd.to_numeric(df['AminerCitationCount'], errors='coerce').fillna(0)
    
    # Filter out papers with no authors
    df = df[df['AuthorNames-Deduped'].str.strip() != '']
    
    return df

def extract_authors_and_collaborations(df):
    """Extract authors and their collaborations from the dataset."""
    print("Extracting authors and collaborations...")
    
    authors = {}
    collaborations = []
    
    for _, row in df.iterrows():
        if pd.isna(row['AuthorNames-Deduped']) or row['AuthorNames-Deduped'].strip() == '':
            continue
            
        # Split authors and clean
        paper_authors = [author.strip() for author in row['AuthorNames-Deduped'].split(';') if author.strip()]
        
        # Add authors to the dictionary
        for author in paper_authors:
            if author not in authors:
                authors[author] = {
                    'id': author,
                    'name': author,
                    'papers': [],
                    'total_citations': 0,
                    'conferences': set(),
                    'collaborators': set(),
                    'years': set()
                }
            
            # Update author data
            authors[author]['papers'].append({
                'title': row['Title'],
                'year': row['Year'],
                'conference': row['Conference'],
                'citations': row['AminerCitationCount'],
                'doi': row['DOI']
            })
            authors[author]['total_citations'] += row['AminerCitationCount']
            authors[author]['conferences'].add(row['Conference'])
            authors[author]['years'].add(row['Year'])
        
        # Create collaboration pairs
        for i in range(len(paper_authors)):
            for j in range(i + 1, len(paper_authors)):
                author1, author2 = sorted([paper_authors[i], paper_authors[j]])
                
                # Add to collaborators set
                authors[author1]['collaborators'].add(author2)
                authors[author2]['collaborators'].add(author1)
                
                # Check if collaboration already exists
                existing_collab = None
                for collab in collaborations:
                    if (collab['source'] == author1 and collab['target'] == author2) or \
                       (collab['source'] == author2 and collab['target'] == author1):
                        existing_collab = collab
                        break
                
                if existing_collab:
                    existing_collab['weight'] += 1
                    existing_collab['papers'].append({
                        'title': row['Title'],
                        'year': row['Year'],
                        'conference': row['Conference']
                    })
                else:
                    collaborations.append({
                        'source': author1,
                        'target': author2,
                        'weight': 1,
                        'papers': [{
                            'title': row['Title'],
                            'year': row['Year'],
                            'conference': row['Conference']
                        }]
                    })
    
    return authors, collaborations

def calculate_network_metrics(authors, collaborations):
    """Calculate various network metrics for analysis."""
    print("Calculating network metrics...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for author_id, author_data in authors.items():
        G.add_node(author_id, **author_data)
    
    # Add edges
    for collab in collaborations:
        G.add_edge(collab['source'], collab['target'], weight=collab['weight'])
    
    # Calculate metrics
    metrics = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'density': nx.density(G),
        'average_clustering': nx.average_clustering(G),
        'average_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else 'Disconnected',
        'number_of_components': nx.number_connected_components(G),
        'largest_component_size': len(max(nx.connected_components(G), key=len)),
        'degree_centrality': nx.degree_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G),
        'closeness_centrality': nx.closeness_centrality(G)
    }
    
    return G, metrics

def get_top_authors(authors, metric='total_citations', top_n=20):
    """Get top authors by various metrics."""
    if metric == 'total_citations':
        sorted_authors = sorted(authors.items(), key=lambda x: x[1]['total_citations'], reverse=True)
    elif metric == 'papers':
        sorted_authors = sorted(authors.items(), key=lambda x: len(x[1]['papers']), reverse=True)
    elif metric == 'collaborators':
        sorted_authors = sorted(authors.items(), key=lambda x: len(x[1]['collaborators']), reverse=True)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return sorted_authors[:top_n]

def analyze_conferences(authors):
    """Analyze conference distribution and cross-conference collaborations."""
    print("Analyzing conference patterns...")
    
    conference_stats = defaultdict(lambda: {
        'authors': set(),
        'papers': 0,
        'citations': 0
    })
    
    cross_conference_collaborations = []
    
    for author_id, author_data in authors.items():
        for conference in author_data['conferences']:
            conference_stats[conference]['authors'].add(author_id)
        
        for paper in author_data['papers']:
            conference_stats[paper['conference']]['papers'] += 1
            conference_stats[paper['conference']]['citations'] += paper['citations']
    
    # Convert sets to counts
    for conf in conference_stats:
        conference_stats[conf]['author_count'] = len(conference_stats[conf]['authors'])
        del conference_stats[conf]['authors']
    
    return dict(conference_stats)

def generate_visualization_data(authors, collaborations):
    """Generate data specifically formatted for the D3.js visualization."""
    print("Generating visualization data...")
    
    # Convert authors to list format
    nodes = []
    for index, (author_id, author_data) in enumerate(authors.items()):
        node = {
            'id': author_id,
            'name': author_data['name'],
            'value': author_data['total_citations'],
            'index': index,
            'paperCount': len(author_data['papers']),
            'totalCitations': author_data['total_citations'],
            'conferences': list(author_data['conferences']),
            'collaboratorCount': len(author_data['collaborators']),
            'primaryConference': get_primary_conference(author_data['conferences']),
            'years': list(author_data['years']),
            'avgCitationsPerPaper': author_data['total_citations'] / len(author_data['papers']) if author_data['papers'] else 0,
            # 'degreeCentrality': metrics['degree_centrality'].get(author_id, 0),
            # 'betweennessCentrality': metrics['betweenness_centrality'].get(author_id, 0),
            # 'closenessCentrality': metrics['closeness_centrality'].get(author_id, 0)
        }
        nodes.append(node)
    
    # Convert collaborations to list format
    links = []
    for collab in collaborations:
        link = {
            'source': collab['source'],
            'target': collab['target'],
            'value': collab['weight'],
            'weight': collab['weight'],
            'paperCount': len(collab['papers']),
            'conferences': list(set(paper['conference'] for paper in collab['papers'])),
            'years': list(set(paper['year'] for paper in collab['papers']))
        }
        links.append(link)

    return {
        # {
        #     "name": "node-data",
        #     "values": nodes
        # },
        # {
        #     "name": "link-data",
        #     "values": links,
        # }
        
        'nodes': nodes,
        'edges': links
    }
    
    # return {
    #     'nodes': nodes,
    #     'links': links,
    #     'metadata': {
    #         'totalNodes': len(nodes),
    #         'totalLinks': len(links),
    #         'totalPapers': sum(len(author['papers']) for author in authors.values()),
    #         'networkMetrics': {
    #             'density': metrics['density'],
    #             'averageClustering': metrics['average_clustering'],
    #             'numberOfComponents': metrics['number_of_components'],
    #             'largestComponentSize': metrics['largest_component_size']
    #         },
    #         'generatedAt': datetime.now().isoformat()
    #     }
    # }

def get_primary_conference(conferences):
    """Get the primary conference for an author based on paper count."""
    if not conferences:
        return 'Unknown'
    
    # For now, return the first conference (can be enhanced with paper count logic)
    return list(conferences)[0]

def save_visualization_data(data, output_file):
    """Save the visualization data to a JSON file."""
    print(f"Saving visualization data to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Data saved successfully!")

def generate_summary_report(authors, collaborations, metrics, conference_stats):
    """Generate a summary report of the analysis."""
    print("\n" + "="*60)
    print("COLLABORATION NETWORK ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nNetwork Overview:")
    print(f"  Total Authors: {metrics['total_nodes']}")
    print(f"  Total Collaborations: {metrics['total_edges']}")
    print(f"  Network Density: {metrics['density']:.4f}")
    print(f"  Average Clustering: {metrics['average_clustering']:.4f}")
    print(f"  Number of Components: {metrics['number_of_components']}")
    print(f"  Largest Component Size: {metrics['largest_component_size']}")
    
    print(f"\nConference Statistics:")
    for conf, stats in conference_stats.items():
        print(f"  {conf}: {stats['author_count']} authors, {stats['papers']} papers, {stats['citations']:.0f} citations")
    
    print(f"\nTop Authors by Citations:")
    top_cited = get_top_authors(authors, 'total_citations', 10)
    for i, (author_id, author_data) in enumerate(top_cited, 1):
        print(f"  {i:2d}. {author_id}: {author_data['total_citations']:.0f} citations ({len(author_data['papers'])} papers)")
    
    print(f"\nTop Authors by Paper Count:")
    top_papers = get_top_authors(authors, 'papers', 10)
    for i, (author_id, author_data) in enumerate(top_papers, 1):
        print(f"  {i:2d}. {author_id}: {len(author_data['papers'])} papers ({author_data['total_citations']:.0f} citations)")
    
    print(f"\nTop Authors by Collaborators:")
    top_collabs = get_top_authors(authors, 'collaborators', 10)
    for i, (author_id, author_data) in enumerate(top_collabs, 1):
        print(f"  {i:2d}. {author_id}: {len(author_data['collaborators'])} collaborators ({len(author_data['papers'])} papers)")

def main():
    """Main function to process the data and generate insights."""
    input_file = 'filtered_dataset.csv'
    # input_file = '../dataset.csv'
    output_file = 'collaboration_network_data.json'
    
    try:
        # Load and process data
        df = load_and_process_data(input_file)
        
        # Extract authors and collaborations
        authors, collaborations = extract_authors_and_collaborations(df)
        
        # Calculate network metrics
        # G, metrics = calculate_network_metrics(authors, collaborations)
        
        # Analyze conferences
        # conference_stats = analyze_conferences(authors)
        
        # Generate visualization data
        viz_data = generate_visualization_data(authors, collaborations)
        
        # Save data
        save_visualization_data(viz_data, output_file)
        
        # Generate summary report
        # generate_summary_report(authors, collaborations, metrics, conference_stats)
        
        print(f"\nProcessing complete! Visualization data saved to {output_file}")
        print("You can now open the HTML file to view the interactive network visualization.")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
