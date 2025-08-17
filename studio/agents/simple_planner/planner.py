from typing_extensions import TypedDict
from typing import List, Union, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import pandas as pd
import networkx as nx

# from helpers import get_llm, get_dataset_info
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from studio.helpers import get_llm, get_dataset_info
else:
    from helpers import get_llm, get_dataset_info


class ReportSection(TypedDict):
    section_number: int
    section_name: str
    section_size: Literal["short", "medium", "long"]
    # section_goal: Literal["present", "exploratory analysis", "confirmatory analysis"]
    section_description: str

class ResponseFormatter(BaseModel):
    report_sections: List[ReportSection]


def initiate(topic: str, dataset: str):
    """
    Initiate the report plan with the topic and dataset info
    """

    dataset_info = get_dataset_info(dataset)
    llm = get_llm()

    system_message = SystemMessage(content=f"""
    You are an expert in data analysis and visualization.
    You are preparing a visualization report based on analysis of the vis publication dataset.

    The topic of the report is {topic}.

    The information about the dataset is as follows:
    {dataset_info}

    You need to plan a report for the topic using the dataset.

    Requirements:
    1. You will only have a limited amount of time to finish the report, so the report should be concise and to the point.
    2. Target audience is researchers in the visualization community, they might be interested in both topic evolution and how people in the field interact and shape the research.
    3. The report should be interesting, coherent, insightful, and visually compelling.
    4. You can decide the number of sections, but no more than 8.

    Response format:
    A list of report sections, each section contains:
    - section_number: the number of the section
    - section_name: the name of the section
    - section_size: the size of the section, either "short", "medium", or "long"
    - section_description: a short description of the section
    """

        # - section_goal: the goal of the section
        # - "present" means the section mainly consists of a visualisation overview, but does not require complex computation, data processing, or deep analysis.
        # - "exploratory analysis" means you are not sure about what the content in the section would be like, you need to explore the data to formulate the tasks and content of the section.
        # - "confirmatory analysis": you have some knowledge about the section and you have a few hypotheses on the content in mind.
    )

    human_message = HumanMessage(content=f"""
    Please plan a report for the topic {topic} using the dataset.
    """
    )

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [system_message, human_message]
    )
    return response


if __name__ == "__main__":
    topic = "What happened to research on automated visualization?"
    response = initiate(topic, "../../dataset.csv")

    import json
    with open("report_plan.json", "w") as f:
        json.dump(response.model_dump(), f, indent=4)
