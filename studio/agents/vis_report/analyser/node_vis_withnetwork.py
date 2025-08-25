from typing_extensions import TypedDict
from typing import List, Union, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers import get_llm, get_dataset_info

from agents.vis_report.analyser.state import State, Visualisation
from agents.vis_report.analyser.memory import memory

from agents.vis_report.load_config import config
import json
import os

def visualise(state: State):
    """
    Visualise the data based on the information need.
    """

    if config["dev"]:
        if "visualisation" in state and state["visualisation"]:
            return state

    new_state = state.copy()
    question = state["analysis_schema"]["information_needed"]["question_text"]
    if is_network_analysis_llm(question):
        print("🔍 Network analysis")
        visualisation = get_antv_visualisation(state)
        new_state["visualisation"] = visualisation
    else:
        print("🔍 Vega-lite visualisation")
        visualisation = get_vega_lite_spec(state)
        new_state["visualisation"] = visualisation.visualisation
    memory.add_state(new_state)
    return new_state

def is_network_analysis(question_text: str):
    network_keywords = ["co-authorship network"]
    return any(keyword in question_text.lower() for keyword in network_keywords)

def is_network_analysis_llm(question_text: str, dataset_info: str = None):
    """
    Use LLM to determine if network analysis is needed.
    """
    llm = get_llm()
    
    system_message = SystemMessage(content=f"""
    You are a data analysis expert, you need to determine if the user's question needs co-authorship network analysis to answer.
    
    Network analysis is suitable for the following cases:
    1. Analyse the co-authorship relationship, connection or interaction of entities
    2. Study co-authorship, reference, social network structure
    3. Discover co-authorship communities, clusters or patterns
    4. Analyse co-authorship centrality, influence, etc.

    It is not suitable for the following cases:
    1. Analyse the temporal relationship, connection or interaction of entities
    2. Analyse the spatial relationship, connection or interaction of entities
    
    
    Please analyse the user's question, determine if network analysis is needed. Only answer "yes" or "no".
    """)
    
    human_message = HumanMessage(content=f"User question: {question_text}")
    
    response = llm.invoke([system_message, human_message])
    return "yes" in response.content.lower()

def get_antv_visualisation(state: State):
    """
    Get the antv visualisation for the network analysis.
    """
    from agents.vis_report.analyser.network import llm_filter, construct_network, filter_network, graph_container
    topic = config["topic"]
    file_path = config["dataset"]
    G, df = construct_network(file_path)
    filtered_G, filtered_df = G, df

    if state["global_filter_state"]:
        print("Using global filter processed dataset ...")
        filtered_file_path = state["global_filter_state"]["dataset_path"]
        response = llm_filter(topic, file_path, filtered_file_path, config["domain_knowledge"])
        filters = response.filters
        print('LLM generatedfilters: ', filters)
        filtered_G, filtered_df = filter_network(G, filtered_file_path, filters)
    else:
        print("Using original dataset with LLM filter...") 
        response = llm_filter(topic, file_path, '', config["domain_knowledge"])
        filters = response.filters
        print('LLM generatedfilters: ', filters)
        filtered_G, filtered_df = filter_network(G, df, filters)

    print('number of nodes before filtering: ', len(G.nodes()))
    print('number of nodes after filtering: ', len(filtered_G.nodes()))
    print('filtered_df: ', filtered_df.shape)

    nodes_data = [{"id": node} for node in filtered_G.nodes()]
    edges_data = [{"source": u, "target": v, "value": filtered_G[u][v]["weight"], "filtered": filtered_G[u][v]["filtered"] if "filtered" in filtered_G[u][v] else True} 
                  for u, v in filtered_G.edges()]
    network_json = json.dumps({
        "nodes": nodes_data,
        "edges": edges_data
    })

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
        </head>
        <body>
            {graph_container(container_id, network_json, width='100%', height='100%')}
        </body>
        </html>
        """
        return html
    
    html = graph_html(network_json)
    visualisation = Visualisation(
        library="antv",
        specification=html
    )

    return visualisation

def get_vega_lite_spec_simple(state: State):
    """
    Get the vega-lite specification for the visualisation.
    """
    llm = get_llm(temperature=0.0)
    dataset_info = get_dataset_info(config["dataset"])

    system_message = SystemMessage(content=f"""
    You are an expert in creating vega-lite specifications for visualisations.

    Use this dataset: {config["dataset_url"]}

    The dataset information is as follows:
    {dataset_info}

    The task is to generate a vega-lite specificiation for the following information, but not necessary to follow the information strictly:
    {state["analysis_schema"]["information_needed"]}

    """)

    human_message = HumanMessage(content=f"""
    Please generate the vega-lite specification for the visualisation. Robustness is prioritised over complexity. Do not generate more than 2 subplots.
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
    return response


def get_vega_lite_spec(state: State):
    """
    Get the vega-lite specification for the visualisation.
    """
    llm = get_llm(temperature=0.0)
    dataset_info = get_dataset_info(config["dataset"])

    system_message = SystemMessage(content=f"""
    You are an expert in creating vega-lite specifications for visualisations.

    Use this dataset: {config["dataset_url"]}

    The dataset information is as follows:
    {dataset_info}

    Please following the information need when you designing and generating the visualisation:
    {state["analysis_schema"]["information_needed"]}

    Requirements:
    1. Generate valid vega-lite specification that can be rendered by Vega-Lite, not violating the critical expression syntax rules and not using old fashion syntax.
    2. Robustness is prioritised over complexity. Do not generate more than 2 subplots.
    3. If the information need is too complex (e.g., the question requires more than 3 levels of data transformation), you don't need to fulfil the complete need. You may generate a visualisation that is relevant to the core need.
    4. All keys and string values must be enclosed in double quotes. Do not use single quotes or unquoted keys.
    5. Ensure all fields referenced in transformation are valid and exist in the preceding data pipeline step.
    6. Double check the data types for all fields used in the encoding block to match the transformation output.
    7. Avoid complex regular expressions in calculate transforms that may be difficult for the vega-lite engine to parse.

    CRITICAL DATA HANDLING RULES:
    8. Always handle null/empty values before operations:
    - Use (datum.field || '') pattern for string operations
    - Add filter "datum.field != null && datum.field != ''" before processing fields with high null rates

    9. For delimited fields (semicolon, comma separated):
    - First filter out empty rows: filter "datum.field != ''"
    - Then split: calculate "split(datum.field, delimiter)"
    - Always validate array before flatten: calculate "isArray(datum.array) ? datum.array : []"
    - Use simple flatten syntax: "flatten": ["field_name"]

    10. For text search operations:
        - Use indexof(lower(field), 'keyword') >= 0 instead of regex
        - Combine multiple conditions with || operator
        - Pre-clean text with lower() function

    11. For numeric fields:
        - Use toNumber() explicitly for calculations
        - For year data already in numeric format, use type: "quantitative" not "temporal"

    ERROR PREVENTION:
    12. Do NOT use: regex test(), anonymous functions, map(), filter(), reduce()
    13. Do NOT assume any field is always non-null - check the dataset info above

    """)

    human_message = HumanMessage(content=f"""
    Please generate the vega-lite specification for the visualisation.
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
    return response
