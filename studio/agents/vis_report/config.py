from typing_extensions import TypedDict


class Config(TypedDict):
    dataset: str
    topic: str
    target_audience: str
    domain_knowledge: str


dataset = "dataset.csv"
topic = "What happened to research on automated visualization?"
target_audience = "researchers in the visualization community, they might be interested in both topic evolution and how people in the field shape the research (such as their interactions and key players)"
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


config = {
    "dataset": dataset,
    "dataset_url": "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv",
    "topic": topic,
    "target_audience": target_audience,
    "domain_knowledge": domain_knowledge,


    "max_section_number": 8,
    "max_analyses_per_section": 3,

    # TODO: remove this when submitting
    "dev": True,
    "thread_to_load": "thread_20250819_125735",
}
