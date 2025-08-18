from typing_extensions import TypedDict
from typing import List, Union, Literal, Optional
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


def initiate(topic: str, dataset: str = "dataset.csv", domain_knowledge: str = ""):
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

    The following knowledge may help you make the report:
    {domain_knowledge}

    Requirements:
    1. You will only have a limited amount of time to finish the report, so the report should be concise and to the point.
    2. Target audience is researchers in the visualization community, they might be interested in both topic evolution and how people in the field shape the research （such as their interactions and key players).
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

    class ResponseFormatter(BaseModel):
        report_sections: List[ReportSection]

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [system_message, human_message]
    )
    return response


def simple_action_plan(section: ReportSection, dataset: str = "dataset.csv", domain_knowledge: str = ""):
    """
    Decide the action to perform for each section.

    Choices:
    1. Generate vis & insights
    2. Plan for exploratory analysis
    3. Pend
    """

    dataset_info = get_dataset_info(dataset)
    llm = get_llm()

    class InformationNeededPresent(TypedDict):
        question_text: str # your exploration or analysis question
        primary_attributes: List[str] # primary attributes to use for the analysis
        secondary_attributes: List[str] # secondary attributes to use for the analysis
        transformation: List[str] # possible transformations to apply to the data
        expected_insight_types: List[str] # expected insight types from the analysis, such as top, trend, distribution, outlier, etc.
    
    class InformationNeededExplore(TypedDict):
        question_text: str # your exploration or analysis question
        key_uncertainty: Optional[str] = None
        expected_outputs: List[str] # expected outputs from the exploration to resolve uncertainty

    # class InformationNeeded(TypedDict):
    #     question_text: str # your exploration or analysis question
    #     key_uncertainty: Optional[str] = None
    #     primary_attributes: List[str] # primary attributes to use for the analysis
    #     secondary_attributes: List[str] # secondary attributes to use for the analysis
    #     transformation: List[str] # possible transformations to apply to the data
    #     expected_insight_types: List[str] # expected insight types from the analysis, such as top, trend, distribution, outlier, etc.

    class ResponseFormatter(BaseModel):
        action: Literal["pend", "present", "explore"]
        information_needed: Optional[Union[InformationNeededPresent, InformationNeededExplore]] = None

    system_message = SystemMessage(content=f"""
    You are an expert in data analysis and visualization.
    You are preparing a visualization report based on analysis of the vis publication dataset.

    You have created an outline of the report and now you are working on a specific section.
    You need to decide the action to perform for completing the section.
    
    Section information:
    {section}

    Overall outline:
    {';'.join([f"{section['section_number']}. {section['section_name']}" for section in report_plan["report_sections"]])}

    The information of the dataset that you will analyse is as follows:
    {dataset_info}

    Your steps to take (see detailed instructions below):
    1. Decide the action (present, explore, or pend)
    2. Specify the information need based on the action you choose

    ### Instructions for the action ###

    You can choose between three types of actions:
    1. Pend: pend to wait for the other sections to be completed
       - If this section is an introduction, executive summary, synthesis, or conclusion, probably you may choose pending the exploration as you can reuse the results from the other sections.
    2. Present: Generate visualisation and corresponding insights
        - If you are confident about which visualisation to use and that you can generate the vega-lite specification without further exploration, you can choose this action.
        - You don't need to know what the insights are, you will be able to analyse and generate the insights.
        - The question is whether you are confident about the visualisation to use.
    3. Explore: Plan for exploratory analysis
        - If you are not sure about the content of the section, you can choose this action.
        - You need to specify the information need (what you expect through the exploration and analysis)

    ### Instructions for the information need ###

    Specify the information need if you choose 'present' or 'explore'.
    - For 'present', basically you need to structure how you plan to design the visualisation.
        - It is a high level plan (such as the important attributes), you don't need to work on the visual design.
        - You can decide how many visualisations, but usually no more than 2.
        - There is no need to fully fulfil the planned description, you can do a trade-off and focus on the most important parts.
        - Note the visualisation will be generated by vega-lite, so it can not be too complex.
    - For 'explore', you need to specify the information need (what you expect through the exploration and analysis)
        - You can specify multiple information needs, but usually no more than 3.
        - You are encouraged to re-formulate the need as you perform the exploration. Therefore, you can start with simple need if you are not very sure.
        - Example situations that need exploration:
            - You are not sure about which filter or transformation to apply to the data.
            - Certain attributes need to be derived from the data (which can not be easily derived through vega-lite).
            - When you decide to present subset or examples but you are not sure about which subset or examples to use.
            - Others...
        - Similarly, focus on the most important parts.

    Form the information need for 'present' as follows:
    - question_text: the question you want to answer through the exploration and analysis. The question should be focused (just one specifc point).
    - primary_attributes: the primary attributes to use for visual encoding (visual channels), usually no more than 2.
    - secondary_attributes: the secondary attributes to use for visual encoding, usually no more than 2.
    - transformation: the transformation to apply to the data, can be none, usually no more than 3.
    - expected_insight_types: the expected insight types from the analysis, such as top, trend, distribution, outlier, etc.
    - parameters: the parameters for the analysis, such as the number of top items, the threshold for the trend, the threshold for the distribution, the threshold for the outlier, etc.

    Form the information need for 'explore' as follows:
    - question_text: the question you want to answer through the exploration and analysis. The question should be focused (just one specifc point).
    - key_uncertainty: if you choose 'explore', specify the key uncertainty you have about the data, which reflects the main goal of the exploration.
    - expected_outputs: the expected outputs from the exploration to resolve uncertainty.
        - Should be practical, such as parameter values for filtering, transformation; or important insights that help you decide the next step.
        - Consider the limit of tools that you can use: you may use pandas and simple networkx, but time-consuming tasks or machine learning tasks are not recommended.
        - Usually no more than 3.

    Please make the information need as simple and focused as possible.

    The following knowledge may help you make the decision:

    Domain knowledge:
    {domain_knowledge}

    Requirements:
    1. You don't need to fully follow the section description, there is a limit of time and space, so focus on the most important parts.
    2. Be simple and concise. If you feel confident about which visualisation to use, please choose present.
    """
    )

    human_message = HumanMessage(content=f"""
    Please decide the action to perform for the section.
    """
    )

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [system_message, human_message]
    )
    return response

# {question_text:“…”,
# q_type: “author_ranking”,
# primary_attributes: [“authors”],
# secondary_attributes:[“publication_count”, “citation_count”],
# transformation: “aggregation_with_filtering”,
# expected_insights:[“top_contributors”],
# parameters:{top_n: 10, min_papers: 2},
# …}

if __name__ == "__main__":
    dataset = "../../dataset.csv"
    topic = "What happened to research on automated visualization?"

    domain_knowledge = """
    Regarding identification of automated visualization papers:

    The following keywords are used to identify the automated visualization papers:
    - automatic vis
    - automated vis
    - visualization recommendation
    - mixed initiative
    - mixed-initiative
    - visualization generation
    - vis generation
    - agent
    
    An exmaple vega-lite filter:
    test(/automatic vis|automated vis|visualization recommendation|mixed initiative|mixed-initiative|visualization generation|vis generation|agent/i, (datum.AuthorKeywords || '') + ' ' + (datum.Abstract || '') + ' ' + (datum.Title || '')) ? 'AutoVis' : 'Other'
    """

    report_plan = initiate(topic, dataset, domain_knowledge)
    report_plan = report_plan.model_dump()

    import json
    with open("report_plan.json", "w") as f:
        json.dump(report_plan, f, indent=4)

    for section in report_plan["report_sections"]:
        response = simple_action_plan(section, dataset, domain_knowledge)
        print(response)

        section["action"] = response.model_dump()

        import json
        with open("report_plan_actions.json", "w") as f:
            json.dump(report_plan, f, indent=4)
