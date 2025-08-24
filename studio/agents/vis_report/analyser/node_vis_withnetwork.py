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
    if is_network_analysis(question):
        visualisation = get_antv_visualisation(state)
        new_state["visualisation"] = visualisation
    else:
        visualisation = get_vega_lite_spec(state)
        new_state["visualisation"] = visualisation.visualisation
    memory.add_state(new_state)
    return new_state

def is_network_analysis(question_text: str):
    network_keywords = ["collaboration", "co-author", "co-authorship"]
    return any(keyword in question_text.lower() for keyword in network_keywords)

def get_antv_visualisation(state: State):
    """
    Get the antv visualisation for the network analysis.
    """
    from agents.vis_report.analyser.network import llm_filter, construct_network, filter_network, graph_container
    topic = config["topic"]
    file_path = config["dataset"]
    response = llm_filter(topic, file_path, config["domain_knowledge"])

    filters = response.filters
    print('filters: ', filters)

    G, df = construct_network(file_path)
    filtered_G, filtered_df = filter_network(G, df, filters)
    print('number of nodes before filtering: ', len(G.nodes()))
    print('number of nodes after filtering: ', len(filtered_G.nodes()))

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

def get_vega_lite_spec(state: State):
    """
    Get the vega-lite specification for the visualisation.
    """
    llm = get_llm()
    dataset_info = get_dataset_info(config["dataset"])

    system_message = SystemMessage(content=f"""
    You are an expert in creating vega-lite specifications for visualisations. 

    Use this dataset: {config["dataset_url"]}

    The dataset information is as follows:
    {dataset_info}

    Please following the information need when you designing and generating the visualisation:
    {state["analysis_schema"]["information_needed"]}

    You may refer to the following domain knowledge:
    {config["domain_knowledge"]}

    Requirements:
    1. Generate valid vega-lite specification that can be rendered by Vega-Lite, not violating the critical expression syntax rules and not using old fashion syntax.
    2. Robustness is prioritised over complexity.
    3. If the information need is too complex, you don't need to fulfil the complete need. You may generate a visualisation that is relevant to the core need.
    """
    )

    human_message = HumanMessage(content=f"""
    Please generate the vega-lite specification for the visualisation.
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
    return response
