from typing import Dict, Any
import pandas as pd
from langchain_core.messages import AIMessage
from state import State

# --- Data Loading Node ---
def data_loading_node(state: State) -> Dict[str, Any]:
    """
    Loads data from the specified file path into a pandas DataFrame.
    """
    updated_state = state.copy()

    file_path = updated_state["file_path"]

    if "messages" not in updated_state or not isinstance(updated_state["messages"], list):
        updated_state["messages"] = []

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Node: Data Loading - Successfully loaded data from {file_path}. Shape: {df.shape}")

        updated_state["dataframe"] = df
        updated_state["messages"].append(AIMessage(content=f"Data loaded from {file_path}. Shape: {df.shape}."))
        return updated_state
    except FileNotFoundError:
        error_msg = f"Node: Data Loading Error - File not found at {file_path}."
        print(error_msg)
        updated_state["messages"].append(AIMessage(content=error_msg))
        updated_state["dataframe"] = None
        return updated_state
    except Exception as e:
        error_msg = f"Node: Data Loading Error - Could not load data from {file_path}. Error: {e}"
        print(error_msg)
        updated_state["messages"].append(AIMessage(content=error_msg))
        updated_state["dataframe"] = None
        return updated_state
