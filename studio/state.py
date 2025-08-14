from typing_extensions import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import pandas as pd
# the state that LangGraph workflow expects as its initial input
class InputState(TypedDict):
    dataset_info: str # This key will contain the string with dataset attributes and file name
    file_path: str
    dataset_url: str
    user_query: str
    messages: List[Dict[str, Any]]

class State(InputState, TypedDict):
    messages: List[BaseMessage]
    dataframe: Optional[pd.DataFrame]
    data_summary: Optional[Dict[str, Any]]
    analysis_tasks: Optional[List[Dict[str, Any]]]  # LLM-planned analysis tasks

    # For fig generation and report generation
    artifacts: Optional[List[Dict[str, Any]]]  # results from data analysis, including figure object, metrics calculated, facts and insights
    all_figures: Optional[List[Dict[str, Any]]]  # all figure objects

    # For report generation
    report_generated: Optional[bool]  # whether report is generated
    report_filename: Optional[str]  # report file name
    report_path: Optional[str]  # report path
    report_config: Optional[Dict[str, Any]]  # report configuration

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