from typing import Dict, Any
import json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from helpers import get_llm

class AnalysisAgentState(TypedDict):
    task: Dict[str, Any]
    dataset_url: str
    analysis_result: str

def react_analysis_node(state: AnalysisAgentState) -> AnalysisAgentState:
    llm = get_llm(temperature=0, max_tokens=2048)
    agent = create_react_agent(
        model=llm,
        tools=[],
        prompt=(
            "You are a data analysis assistant. "
            "Based on the task and the dataset URL, "
            "please generate a Vega-Lite JSON specification for visualization, "
            "and provide a clear textual analysis of the data and the chart."
            "\n\nOutput format:\n"
            "1. Vega-Lite JSON spec in a markdown code block ```json ... ```\n"
            "2. Narrative analysis text explaining insights.\n"
            "\nIMPORTANT: Use the dataset URL given below directly in the Vega-Lite spec to load data.\n"
        )
    )

    task = state["task"]
    dataset_url = state["dataset_url"]

    messages = [
        {
            "role": "user",
            "content": f"Task: {json.dumps(task)}\nDataset CSV URL: {dataset_url}"
        }
    ]

    result = agent.invoke({"messages": messages})
    # analysis_text = getattr(result, "content", str(result))
    #
    # return {**state, "analysis_result": analysis_text}

    # print("agent.invoke result:", result)
    return {**state, "analysis_result": result}

def create_analysis_agent_workflow():
    builder = StateGraph(AnalysisAgentState)
    builder.add_node("react_analysis", react_analysis_node)
    builder.add_edge(START, "react_analysis")
    builder.add_edge("react_analysis", END)
    return builder.compile()

def analysis_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    analysis_tasks = state.get("analysis_tasks", [])
    dataset_url = state.get("dataset_url")
    messages = state.get("messages", [])

    if not analysis_tasks:
        err = "No analysis tasks provided"
        return {"analysis_result": {"error": err}, "final_messages": messages + [AIMessage(content=err)]}
    if not dataset_url:
        err = "No dataset URL provided"
        return {"analysis_result": {"error": err}, "final_messages": messages + [AIMessage(content=err)]}

    selected_task = analysis_tasks[0]
    workflow = create_analysis_agent_workflow()
    agent_input = {
        "task": selected_task,
        "dataset_url": dataset_url,
        "analysis_result": ""
    }
    output = workflow.invoke(agent_input)

    return {
        "analysis_result": output["analysis_result"],
        "final_messages": messages + [AIMessage(content=str(output["analysis_result"]))]
    }
