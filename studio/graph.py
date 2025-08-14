from langgraph.graph import StateGraph, END
from state import State, InputState, OutputState


# ----------Graph ----------
def create_workflow():
    # create the agentic workflow using LangGraph
    builder = StateGraph(State, input_schema=InputState, output_schema=OutputState)

    # --- Add the nodes ---
    from node_data_loading import data_loading_node
    builder.add_node("data_loading", data_loading_node)

    from node_inspect import task_discovery_node
    builder.add_node("task_discovery", task_discovery_node)
    from node_draw import react_analysis_node
    builder.add_node("react_analysis", react_analysis_node)
    from module_report import report_generation_node
    builder.add_node("report_generation", report_generation_node)

    builder.set_entry_point("data_loading")
    builder.add_edge("data_loading", "task_discovery")
    builder.add_edge("task_discovery", "react_analysis")
    builder.add_edge("react_analysis", "report_generation")
    builder.add_edge("report_generation", END)

    # Compile the graph
    return builder.compile()