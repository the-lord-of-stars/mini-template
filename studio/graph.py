from langgraph.graph import START, StateGraph, END
from state import State, InputState, OutputState


# ----------Graph ----------
def create_workflow():
    # create the agentic workflow using LangGraph
    # builder = StateGraph(State, input_schema=InputState, output_schema=OutputState)
    builder = StateGraph(State)

    # --- Add the nodes ---
    # from node_data_loading import data_loading_node
    # builder.add_node("data_loading", data_loading_node)

    from node_select_data import select_data
    builder.add_node("select_data", select_data)

    from node_question import question
    builder.add_node("question", question)

    from node_fatcs import get_facts
    builder.add_node("facts", get_facts)

    from node_insights import get_insights
    builder.add_node("insights", get_insights)

    from node_draw_new import draw
    builder.add_node("draw", draw)

    from node_follow_up_decision import follow_up_decision
    builder.add_node("follow_up_decision", follow_up_decision)

    from node_synthesise import synthesise
    builder.add_node("synthesise", synthesise)



    # from node_inspect import task_discovery_node
    # builder.add_node("task_discovery", task_discovery_node)
    # from node_draw import react_analysis_node
    # builder.add_node("react_analysis", react_analysis_node)
    # from node_report import report_generation_node
    # builder.add_node("report_generation", report_generation_node)

    # --- Add the edges ---
    builder.add_edge(START, "select_data")
    builder.add_edge("select_data", "question")
    builder.add_edge("question", "facts")
    builder.add_edge("facts", "insights")
    builder.add_edge("insights", "draw")

    # Always go to follow_up_decision after draw
    builder.add_edge("draw", "follow_up_decision")
    
    # Add conditional edge from follow_up_decision based on should_continue
    def route_after_follow_up_decision(state):
        if state.get("should_continue", False):
            # Continue with next iteration - always go to question to generate new question
            return "question"
        else:
            # End iteration, go to synthesise for final report
            return "synthesise"
    
    builder.add_conditional_edges("follow_up_decision", route_after_follow_up_decision, {
        "question": "question",
        "synthesise": "synthesise"
    })
    
    # synthesise always goes to END
    builder.add_edge("synthesise", END)


    # builder.add_edge("data_loading", "task_discovery")
    # builder.add_edge("task_discovery", "react_analysis")
    # builder.add_edge("react_analysis", "report_generation")
    # builder.add_edge("report_generation", END)

    # Compile the graph
    return builder.compile()