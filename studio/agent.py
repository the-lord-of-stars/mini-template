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
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()
    
    def initialize_state(self):    
        example_input = """
        There is a dataset of IEEE VIS papers from 1990-2024, there are following 19 attributes. 
        Conference	Year, Title, DOI, Link, FirstPage, LastPage, PaperType, Abstract, AuthorNames-Deduped, AuthorNames, AuthorAffiliation, InternalReferences, AuthorKeywords, AminerCitationCount, CitationCount_CrossRef, PubsCited_CrossRef, Downloads_Xplore, Award. 
        The following is one data point:
        InfoVis	2011	D³ Data-Driven Documents	10.1109/tvcg.2011.185	http://dx.doi.org/10.1109/TVCG.2011.185	2301	2309	J	Data-Driven Documents (D3) is a novel representation-transparent approach to visualization for the web. Rather than hide the underlying scenegraph within a toolkit-specific abstraction, D3 enables direct inspection and manipulation of a native representation: the standard document object model (DOM). With D3, designers selectively bind input data to arbitrary document elements, applying dynamic transforms to both generate and modify content. We show how representational transparency improves expressiveness and better integrates with developer tools than prior approaches, while offering comparable notational efficiency and retaining powerful declarative components. Immediate evaluation of operators further simplifies debugging and allows iterative development. Additionally, we demonstrate how D3 transforms naturally enable animation and interaction with dramatic performance improvements over intermediate representations.	Michael Bostock;Vadim Ogievetsky;Jeffrey Heer	Michael Bostock;Vadim Ogievetsky;Jeffrey Heer	Computer Science Department, Stanford University, Stanford, CA, USA;Computer Science Department, Stanford University, Stanford, CA, USA;Computer Science Department, Stanford University, Stanford, CA, USA	10.1109/infvis.2000.885091;10.1109/infvis.2000.885098;10.1109/tvcg.2010.144;10.1109/tvcg.2009.174;10.1109/infvis.2004.12;10.1109/tvcg.2006.178;10.1109/infvis.2005.1532122;10.1109/tvcg.2008.166;10.1109/infvis.2004.64;10.1109/tvcg.2007.70539;10.1109/infvis.2000.885091	Information visualization, user interfaces, toolkits, 2D graphics	3795	2178	41	11668	TT							
        Name of csv file is - "dataset.csv"
        """
        state = {
            "message": str(example_input),       
        }
        return state
    def decode_output(self, output: dict):
        # Decode the output and save it as a pdf
        generate_report(output, "output.pdf")

    def process(self):

        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        state = self.initialize_state()
        output_state = self.workflow.invoke(state)
        def _flatten(value):
            return getattr(value, "content", value)
        result = {k: _flatten(v) for k, v in output_state.items()}
        self.decode_output(result)
        return result