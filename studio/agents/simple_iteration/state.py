from typing_extensions import TypedDict

class SelectDataState(TypedDict):
    description: str
    sql_query: str
    dataset_path: str

class Question(TypedDict):
    question: str
    handled: bool
    spec: str  # Vega-Lite specification

class Facts(TypedDict):
    code: str
    stdout: str
    stderr: str
    exit_code: int

class Visualization(TypedDict):
    insight: str
    chart_type: str
    altair_code: str
    description: str
    is_appropriate: bool
    image_path: str

class Visualizations(TypedDict):
    visualizations: list[Visualization]

class ListQuestionsState(TypedDict):
    question: Question

class FollowUpDecision(TypedDict):
    should_reselect_data: bool
    reasoning: str

class State(TypedDict):
    # message: str
    # dataset_info: str
    topic: str # the topic to explore
    select_data_state: SelectDataState
    question: Question
    facts: Facts
    insights: list[str]
    iteration_count: int  # Add this field to track iterations
    max_iterations: int   # Add this field to set max iterations
    should_continue: bool # Add this field to control the loop
    follow_up_decision: FollowUpDecision  # Add this field for follow-up decisions
    # visualizations: Visualizations
