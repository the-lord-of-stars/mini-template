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
    task: Optional[Dict[str, Any]]
    dataframe: Optional[pd.DataFrame]
    data_summary: Optional[Dict[str, Any]]
    analysis_tasks: Optional[List[Dict[str, Any]]]  # LLM-planned analysis tasks
    processed_dataframe: Optional[pd.DataFrame]
    processed_summary: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]

class OutputState(TypedDict):
    analysis_tasks: List[Dict[str, Any]]
    final_messages: List[BaseMessage]
    analysis_result: Optional[Dict[str, Any]]


#
#
# class PrivateState(TypedDict):
#     # Defines any PrivateState types for internal, isolated node communication (especially useful with Send objects).