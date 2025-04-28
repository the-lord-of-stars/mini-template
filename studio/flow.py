import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

# High-level Procedure to create a multi-agent workflow
# A. Decompose the task into several sub-tasks
# B. Create an LLM-based agent for each sub-task
# C. Connect the agents in the workflow

# Tips to create a multi-agent workflow in Python
# 1. Define a state to store the messages
#   State[message] to store the message
#   State[sender] to store the which agent sends the message
#   other fields to store the msgs

# 2. Define a node for the workflow
    # 2.1 Get state
    # 2.2 Define a prompt to invoke LLM
    # 2.3 we want the generated message in the format of message: string 
    # 2.4 define LLM for the agent node
    # 2.5 Invoke LLM
    # 2.6 Update the state with the generated message

class State(TypedDict):
    message: str
    refinement: str

def generate_msg(state):
    """ Node to generate a message """
    message = state["message"]
    answer_template = """Generate a message {message}"""
    answer_instructions = answer_template.format(message=message)    
    llm = ChatOpenAI(model="gpt-4o", temperature=0) 
    answer = llm.invoke(
        [SystemMessage(content=answer_instructions)]
        +[HumanMessage(content=f"Generate a message.")]
    )
    return {"message": answer}

def generate_refinement(state):
    """ Node to generate a refinement """
    refinement = state["refinement"]
    message = state["message"]
    answer_template = """Generate a refinment for the {message}"""
    answer_instructions = answer_template.format(message=message)    
    llm = ChatOpenAI(model="gpt-4o", temperature=0) 
    answer = llm.invoke(
        [SystemMessage(content=answer_instructions)]
        +[HumanMessage(content=f"Refine the current message to incorporate these suggestions: {refinement}.")]
    )
    return {"refinement": answer}
# 3. Add agent nodes into the graph workflow
# 4. Add edges to connect the agents 
# 5. Compile the workflow
# register the workflow in langgraph.json with the graph name


def create_workflow():
    builder = StateGraph(State)
    builder.add_node("generate_msg", generate_msg)
    builder.add_node("generate_refinement", generate_refinement)
    builder.add_edge(START, "generate_msg")
    builder.add_edge("generate_msg", "generate_refinement")
    builder.add_edge("generate_refinement", END)
    graph = builder.compile()
    return graph

class Agent:
    def __init__(self) -> None:
        self.workflow = None  

    def initialize(self) -> None:
        """Load models, build the graph, etc."""
        self.workflow = create_workflow()

    def process(self, task: dict[str, any]) -> dict[str, any]:
        # TBD: what is the input format; dataset or a single row?
        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        # Convert dataset info to LangGraph state as input
        state = {
            "message": str(task.get("message", task)),        # fallback: entire row as message
            "refinement": str(task.get("refinement", "")),    # empty string if absent
        }
        # Invoke the workflow
        output_state = self.workflow.invoke(state)
        # Flatten AIMessage objects so json.dump doesn’t choke
        def _flatten(val):
            return getattr(val, "content", val)

        return {k: _flatten(v) for k, v in output_state.items()}
    def _demo(self) -> None:
        # self.initialize()
        demo_out = self.process(
            {"message": "Hello world!", "refinement": "make it cheerful"}
        )
        print(demo_out)


if __name__ == "__main__":
    agent = Agent()
    # agent.initialize()
    # agent._demo()




