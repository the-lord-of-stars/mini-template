from typing_extensions import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import pandas as pd
# the state that LangGraph workflow expects as its initial input
class InputState(TypedDict):
    topic: str  # the topic to explore
    iteration_count: int  # Add this field to track iterations
    max_iterations: int  # Add this field to set max iterations
    should_continue: bool  # Add this field to control the loop

class SelectDataState(TypedDict):
    description: str
    sql_query: str
    dataset_path: str

class Question(TypedDict):
    question: str
    handled: bool
    spec: str  # Vega-Lite specification

class AnalysisPlan(TypedDict):
    question_text: str
    q_type: str
    primary_attributes: list[str]
    secondary_attributes: list[str]
    transformation: str
    expected_insights: list[str]
    parameters: dict
    visualization_types: list[str]
    analysis_focus: str

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
    success: bool
    figure_object: Optional[Any]
    code: str


class Visualizations(TypedDict):
    visualizations: list[Visualization]

class ListQuestionsState(TypedDict):
    question: Question

class FollowUpDecision(TypedDict):
    should_reselect_data: bool
    reasoning: str

class Synthesise(TypedDict):
    report_generated: bool
    report_path: str
    root_report_path: str
    success: bool
    error: str


class State(InputState, TypedDict):

    select_data_state: SelectDataState #{description, sql_query, dataset_path}
    question: Question #{question, handled or not, spec}
    analysis_plan: AnalysisPlan #{question_text, q_type, primary_attributes, secondary_attributes, transformation, expected_insights, parameters, visualization_types, analysis_focus}
    facts: Facts #{code, stdout, stderr, exit_code}
    insights: list[str]

    follow_up_decision: FollowUpDecision  # Add this field for follow-up decisions
    visualizations: Visualizations
    synthesise: Synthesise  # Add this field for report generation

    dataframe: Optional[pd.DataFrame]
    iteration_history: Optional[List[Dict[str, Any]]] # a list of {question, facts, insights}

class OutputState(TypedDict):
    analysis_tasks: List[Dict[str, Any]]
    final_messages: List[BaseMessage]
    analysis_result: Optional[Dict[str, Any]]

    # figure and report info
    artifacts: Optional[List[Dict[str, Any]]]  # analysis results
    all_figures: Optional[List[Dict[str, Any]]]  # all figure objects
    report_generated: Optional[bool]  # whether report is generated
    report_filename: Optional[str]  # report name
    report_path: Optional[str]  # report path


#
#
# class PrivateState(TypedDict):
#     # Defines any PrivateState types for internal, isolated node communication (especially useful with Send objects).