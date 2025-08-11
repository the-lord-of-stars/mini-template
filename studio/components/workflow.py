from langgraph.graph import StateGraph, START, END
from studio.components.state import State
from studio.process.process.generate import generate_msg


def create_workflow():
    # create the agentic workflow using LangGraph
    builder = StateGraph(State)
    builder.add_node("generate_msg", generate_msg)
    builder.add_edge(START, "generate_msg")
    builder.add_edge("generate_msg", END)
    return builder.compile()