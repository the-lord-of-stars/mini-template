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


def get_vega_lite_spec(info_need: str, context: str, dataset: str = "dataset.csv", domain_knowledge: str = ""):
    """
    Get the vega-lite specification for the visualisation.
    """
    llm = get_llm()
    dataset_info = get_dataset_info(dataset)

    class ResponseFormatter(BaseModel):
        spec: str

    system_message = SystemMessage(content=f"""
    You are an expert in creating vega-lite specifications for visualisations.

    Use this dataset: file_name = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"

    The dataset information is as follows:
    {dataset_info}

    The context that the visualisation will be used is:
    {context}

    The information you need to generate the vega-lite specification is as follows:
    {info_need}

    You may refer to the following domain knowledge:
    {domain_knowledge}

    Requirements:
    1. Generate valid vega-lite specification.
    2. Robustness is prioritised over complexity.
    3. If the information need is too complex, you don't need to fulfil the complete need. You may generate a visualisation that is relevant to the core need.
    """
    )

    human_message = HumanMessage(content=f"""
    Please generate the vega-lite specification for the visualisation.
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

    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Interactive Vega-Lite Report</title>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>",
        "  <style>",
        "    body { font-family: sans-serif; margin: 2em; }",
        "    h3 { margin-top: 1em; }",
        "    hr { margin: 1.5em 0; }",
        "  </style>",
        "</head>",
        "<body>",
    ]


    def add_info_html(info_need: dict, spec: str, id: str):
        div_id = f"vis{vis_counter}"
        html_lines.append(f"  <div>{info_need['question_text']}</div>")
        html_lines.append(f"  <div id='{div_id}'></div>")
        html_lines.extend([
            "  <script>",
            f"    vegaEmbed('#{div_id}', {spec})",
            "      .catch(console.error);",
            "  </script>",
        ])



    import json
    report_plan = json.load(open("report_plan_actions.json"))

    vis_counter = 0

    for section in report_plan["report_sections"]:
        html_lines.append(f"  <h3>{section['section_name']}</h3>")
        html_lines.append(f"  <p>{section['section_description']}</p>")

        if section["action"]["action"] == "present" or section["action"]["action"] == "explore":
            information_needed = section["action"]["information_needed"]
            for info_need in information_needed:
                context = {
                    "topic": topic,
                    "section_info": section,
                }
                response = get_vega_lite_spec(info_need, context, dataset, domain_knowledge)
                spec = response.spec
                print(spec)
                add_info_html(info_need, spec, vis_counter)
                vis_counter += 1

            with open("report_html_temp.html", "w") as f:
                f.write("\n".join(html_lines) + "\n" + "</body>" + "\n" + "</html>")

    html_lines.append("</body>")
    html_lines.append("</html>")

    with open("report_html.html", "w") as f:
        f.write("\n".join(html_lines))
    
