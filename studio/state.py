from typing_extensions import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import pandas as pd
# the state that LangGraph workflow expects as its initial input
class InputState(TypedDict):
    # dataset_info: str # This key will contain the string with dataset attributes and file name
    # file_path: str
    # dataset_url: str
    # user_query: str
    # messages: List[Dict[str, Any]]
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

# class State(InputState, TypedDict):
#     messages: List[BaseMessage]
#
#     # select data
#     topic: str  # the topic to explore
#     select_data_state: SelectDataState
#
#     # record one question
#     question: Question
#
#     facts: Facts
#     insights: list[str]
#
#     # for controlling the loop
#     iteration_count: int  # Add this field to track iterations
#     max_iterations: int  # Add this field to set max iterations
#     should_continue: bool  # Add this field to control the loop
#
#     dataframe: Optional[pd.DataFrame]
#     data_summary: Optional[Dict[str, Any]]
#     analysis_tasks: Optional[List[Dict[str, Any]]]  # LLM-planned analysis tasks
#
#     # For fig generation and report generation
#     artifacts: Optional[List[Dict[str, Any]]]  # results from data analysis, including figure object, metrics calculated, facts and insights
#     all_figures: Optional[List[Dict[str, Any]]]  # all figure objects
#
#     # For report generation
#     report_generated: Optional[bool]  # whether report is generated
#     report_filename: Optional[str]  # report file name
#     report_path: Optional[str]  # report path
#     report_config: Optional[Dict[str, Any]]  # report configuration
#
#     # for follow-up node
#     follow_up_decision: FollowUpDecision  # Add this field for follow-up decisions
#     # visualizations: Visualizations

class State(InputState, TypedDict):

    select_data_state: SelectDataState #{description, sql_query, dataset_path}
    question: Question #{question, handled or not, spec}
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