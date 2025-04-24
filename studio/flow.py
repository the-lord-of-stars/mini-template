import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

# llm = ChatOpenAI(model="gpt-4o", temperature=0) 


# High-level Procedure to create a multi-agent workflow
# A. Decompose the task into several sub-tasks
# B. Create an LLM-based agent for each sub-task
# C. Connect the agents in the workflow

# Tips to create a multi-agent workflow in Python
# 1. Define a state to store the messages
#   State[message] to store the message
#   State[sender] to store the which agent sends the message

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
# register the workflow in langgraph.json with the graph_1 name

builder_1 = StateGraph(State)
builder_1.add_node("generate_msg", generate_msg)
builder_1.add_node("generate_refinement", generate_refinement)
builder_1.add_edge(START, "generate_msg")
builder_1.add_edge("generate_msg", "generate_refinement")
builder_1.add_edge("generate_refinement", END)
graph_1 = builder_1.compile()


# 6. Define a second workflow in the same way
builder_2 = StateGraph(State)
builder_2.add_node("generate_msg", generate_msg)
builder_2.add_edge(START, "generate_msg")
builder_2.add_edge("generate_msg", END)
graph_2 = builder_2.compile()



