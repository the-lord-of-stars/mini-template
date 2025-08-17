from langgraph.graph import START, StateGraph, END
from state import State, InputState, OutputState


# ----------Graph ----------
def create_workflow():
    # create the agentic workflow using LangGraph
    builder = StateGraph(State)

    # --- Add the nodes ---

    from node_select_data import select_data
    builder.add_node("select_data", select_data)

    # from node_question import question
    # builder.add_node("question", question)

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

    # -----------------------------without question node-----------------------------
    # --- Add the edges ---
    # builder.add_edge(START, "select_data")
    # builder.add_edge("select_data", "question")
    # builder.add_edge("question", "facts")
    # builder.add_edge("facts", "insights")
    # builder.add_edge("insights", "draw")

    # builder.add_edge("draw", "follow_up_decision")
    # def route_after_follow_up_decision(state):
    #     if state.get("should_continue", False):
    #         return "question"
    #     else:
    #         return "synthesise"
    # builder.add_conditional_edges("follow_up_decision", route_after_follow_up_decision, {
    #     "question": "question",
    #     "synthesise": "synthesise"
    # })
    # builder.add_edge("synthesise", END)

    # -----------------------------with updated question node-----------------------------

    from node_question_structured import question
    builder.add_node("question", question)

    from node_analyse_topics import analyse_topics
    builder.add_node("analyse_topics", analyse_topics)

    from node_analyse_authors import analyse_author_network
    builder.add_node("analyse_authors", analyse_author_network)

    def analysis_condition(state: State) -> str:
        # Check if analysis_plan exists and get q_type
        if 'analysis_plan' in state and hasattr(state['analysis_plan'], 'q_type'):
            q_type = state['analysis_plan'].q_type
        elif 'analysis_plan' in state and isinstance(state['analysis_plan'], dict):
            q_type = state['analysis_plan'].get('q_type', '')
        else:
            q_type = ''
        
        print(f"Analysis condition: q_type = {q_type}")
        
        # Convert enum to string for comparison
        q_type_str = str(q_type).split('.')[-1] if '.' in str(q_type) else str(q_type)
        print(f"Analysis condition: q_type_str = {q_type_str}")
        
        if q_type_str == "TOPIC_ANALYSIS":
            return "analyse_topics"
        elif q_type_str == "COLLABORATION_ANALYSIS":
            return "analyse_authors"
        elif q_type_str == "GENERAL_ANALYSIS":
            return "facts"  # Route general analysis to facts
        else:
            return "facts"

    builder.add_edge(START, "select_data")
    builder.add_edge("select_data", "question")
    builder.add_conditional_edges(
        "question",
        analysis_condition,
        {
            "analyse_topics": "analyse_topics",
            "analyse_authors": "analyse_authors",
            "facts": "facts"
        }
    )
    builder.add_edge("analyse_topics", "follow_up_decision")
    builder.add_edge("analyse_authors", "follow_up_decision")
    builder.add_edge("facts", "insights")
    builder.add_edge("insights", "draw")

    builder.add_edge("draw", "follow_up_decision")

    def route_after_follow_up_decision(state):
        if state.get("should_continue", False):
            return "question"
        else:
            return "synthesise"

    builder.add_conditional_edges("follow_up_decision", route_after_follow_up_decision, {
        "question": "question",
        "synthesise": "synthesise"
    })
    
    builder.add_edge("synthesise", END)


    # Compile the graph
    return builder.compile()
    # return builder.compile(recursion_limit=50)