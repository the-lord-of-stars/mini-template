from langchain_core.messages import SystemMessage, HumanMessage

from studio.components.llm import get_llm
from studio.components.state import State


def generate_msg(state: State):
    dataset_info = state["dataset_info"]
    # if the prompt is to generate Vega-Lite charts, then specify in sys_prompt and use generate_html_report()
    sys_prompt = f"Please generate Vega-Lite graphs to visualize insights from the dataset, output should be graphs and narrative: {dataset_info}"

    # if the prompt is to generate Python codes, then specify in sys_prompt and use generate_pdf_report()
    # sys_prompt = f"Please generate Python code to visualize insights from the dataset, output should be graphs and narrative: {dataset_info}"

    # get the LLM instance
    llm = get_llm(temperature=0, max_tokens=4096)

    # generate the response
    response = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content="Generate a response.")]
    )
    return {"message": response}