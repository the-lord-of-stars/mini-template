from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from helpers import get_llm
from report import generate_report

class State(TypedDict):
    message: str

def generate_msg(state: State):
    message = state["message"]
    sys_prompt = f"Please generate a Python code to visualize insights from the dataset, output should be codes and narrative: {message}"
    llm = get_llm(temperature=0, max_tokens=4096)
    answer = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content="Generate a response.")]
    )
    return {"message": answer}


def create_workflow():
    builder = StateGraph(State)
    builder.add_node("generate_msg", generate_msg)
    builder.add_edge(START, "generate_msg")
    builder.add_edge("generate_msg", END)
    return builder.compile()

class Agent:
    def __init__(self):
        self.workflow = create_workflow()

    def initialize(self):
        self.workflow = create_workflow()

    def process(self, task: dict):

        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        state = {
            "message": str(task.get("message", task)),       
        }
        
        output_state = self.workflow.invoke(state)

        def _flatten(value):
            return getattr(value, "content", value)

        result = {k: _flatten(v) for k, v in output_state.items()}
        
        # Generate pdf report as output
        generate_report(result, "output.pdf")
        
        # Return the agent's result
        return result 