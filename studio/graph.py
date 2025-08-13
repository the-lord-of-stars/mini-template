from langgraph.graph import StateGraph, END
from state import State, InputState, OutputState
from node_data_loading import data_loading_node
from node_data_exploration import data_exploration_node
from node_task_planning import task_planning_node
from node_analysis import analysis_agent_node

# ----------Graph ----------
def create_workflow():
    # create the agentic workflow using LangGraph
    builder = StateGraph(State, input_schema=InputState, output_schema=OutputState)

    # --- Add the nodes ---
    builder.add_node("data_loading", data_loading_node)
    builder.add_node("data_exploration",data_exploration_node)
    builder.add_node("task_planning",task_planning_node)
    builder.add_node("data_analysis", analysis_agent_node)

    builder.set_entry_point("data_loading")
    builder.add_edge("data_loading", "data_exploration")
    builder.add_edge("data_exploration", "task_planning")
    builder.add_edge("task_planning", "data_analysis")
    builder.add_edge("data_analysis", END)

    # Compile the graph
    return builder.compile()