from langgraph.graph import StateGraph, END
from state import State, InputState, OutputState
from node_data_loading import data_loading_node
from node_data_exploration import data_exploration_node
from node_task_planning import task_planning_node
from node_analysis import analysis_agent_node
from node_task_identification import task_identification_node
from node_data_process import data_process_node

# ----------Graph ----------
def create_workflow():
    # create the agentic workflow using LangGraph
    builder = StateGraph(State, input_schema=InputState, output_schema=OutputState)

    # --- Add the nodes ---
    builder.add_node("data_loading", data_loading_node) # load the dataframe to main state
    # builder.add_node("data_exploration",data_exploration_node) # not used atm to test the functionality of data analysis for one specific task
    # builder.add_node("task_planning",task_planning_node) # not used atm to test the functionality of data analysis for one specific task
    builder.add_node("task_identification", task_identification_node) # an LLM identifies the domain/topic and time range that the user request to analyse
    builder.add_node("data_process", data_process_node) # pre-process/filter the data based on keyword (an LLM identified based on user query) + time range
    builder.add_node("data_analysis", analysis_agent_node) # an agent calls (two) tools based on user query to analyse the data. more tools to add.
    # TODO: More functions to add to data_analysis.
    # TODO: the network visualisation needs to change - adapt font size and bubble size etc.

    # builder.set_entry_point("data_loading")
    # builder.add_edge("data_loading", "data_exploration")
    # builder.add_edge("data_exploration", "task_planning")
    # builder.add_edge("task_planning", "data_analysis")
    # builder.add_edge("data_analysis", END)

    # this workflow is mainly to test whether the analysis works for a specific task.
    builder.set_entry_point("data_loading")
    builder.add_edge("data_loading", "task_identification")
    builder.add_edge("task_identification", "data_process")
    builder.add_edge("data_process", "data_analysis")
    builder.add_edge("data_analysis", END)
    # TODO: make the final report generation to be the last node

    # Compile the graph
    return builder.compile()