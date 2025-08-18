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


def get_analysis_script(info_need: str, dataset: str = "dataset.csv", domain_knowledge: str = ""):
    """
    Get the analysis python script for the exploration.
    """
    llm = get_llm()
    dataset_info = get_dataset_info(dataset)

    class ResponseFormatter(BaseModel):
        script: str

    system_message = SystemMessage(content=f"""
    You are an data scientist. You are given a dataset and you need to write a python script to perform the analysis.

    The path of the dataset is:
    {dataset}

    The dataset information is as follows:
    {dataset_info}

    The information you need to generate the analysis script is as follows:
    {info_need}

    You may refer to the following domain knowledge:
    {domain_knowledge}

    Requirements:
    1. Generate valid python script.
    2. Robustness is prioritised over complexity.
    3. If the information need is too complex, you don't need to fulfuil the complete need. You may generate a script that is relevant to the core need.
    4. Libraries that you may use: pandas, numpy, networkx, matplotlib, seaborn
    5. The script should be executable.
    6. The script should be concise and to the point.
    7. Print the outputs
    """
    )

    human_message = HumanMessage(content=f"""
    Please generate the python script to perform the analysis.
    """
    )

    response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
    return response

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

    info_need = {
                        "question_text": "For each automated-visualization subtype in the operational taxonomy (recommendation, generation, mixed-initiative, agents, pipeline automation), which papers from the dataset are clear exemplars? (Return a short ranked list per subtype.)",
                        "primary_attributes": [
                            "Title",
                            "Abstract"
                        ],
                        "secondary_attributes": [
                            "Year",
                            "Conference"
                        ],
                        "transformation": [
                            "filter by keyword regex (automatic vis|automated vis|visualization recommendation|mixed initiative|mixed-initiative|visualization generation|vis generation|agent) applied to Title+Abstract+AuthorKeywords",
                            "group by subtype label inferred from which keyword matched",
                            "rank within each subtype by CitationCount_CrossRef or AminerCitationCount"
                        ],
                        "expected_insight_types": [
                            "top (representative examples per subtype)",
                            "distribution (how many exemplars found per subtype)",
                            "outlier (papers that match multiple subtypes) "
                        ]
                    }

    response = get_analysis_script(info_need, dataset, domain_knowledge)
    print(response.script)

    with open("analysis_script.py", "w") as f:
        f.write(response.script)
