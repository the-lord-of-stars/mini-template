# e.g., read_csv_file, perform_statistical_test, generate_chart_data, save_to_database
# should be decorated with @tool
# A helper function to collect all tools into a list or dictionary, which can then be passed to your LLM or LangGraph builder.
from typing import Any, List, Optional, Dict

import pandas as pd
import numpy as np
from langchain_core.tools import tool  # Import the tool decorator
import json
import io
# --- LLM Setup ---
from helpers import get_llm

# --- Simplified get_dataframe_summary (No @tool decorator needed if only called internally) ---
def get_dataframe_summary(df: pd.DataFrame) -> dict:
    summary = {}
    summary["shape"] = list(df.shape)

    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True, show_counts=True)
    summary["column_info"] = buffer.getvalue().splitlines()

    missing_values = df.isnull().sum()
    missing_data_info = missing_values[missing_values > 0].to_dict()
    if missing_data_info:
        summary["missing_values"] = {k: int(v) for k, v in missing_data_info.items()}
    else:
        summary["missing_values"] = "No missing values."

    numeric_summary = df.describe(include='number').to_dict()
    for col, stats in numeric_summary.items():
        numeric_summary[col] = {k: (float(v) if pd.notna(v) else None) for k, v in stats.items()}
    summary["numeric_statistics"] = numeric_summary

    summary["sample_data"] = df.head(5).to_dict(orient='records')

    categorical_unique_counts = {}
    for col in df.select_dtypes(include='object').columns:
        if col in df.columns:
            top_values_series = df[col].value_counts().head(5)
            categorical_unique_counts[col] = [{"value": str(k), "count": int(v)} for k, v in top_values_series.items()]

    summary["categorical_unique_values"] = categorical_unique_counts

    return summary

def _get_column_info(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Helper to get column names, types, and sample values."""
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_values = df[col].dropna().unique()

        num_missing = int(df[col].isnull().sum())

        if len(unique_values) < 10:
            converted_sample_values = [convert_numpy_types(val) for val in unique_values.tolist()]
        else:
            converted_sample_values = [convert_numpy_types(val) for val in unique_values.tolist()[:5]]

        column_info.append({
            "name": col,
            "dtype": dtype,
            "num_missing": num_missing,
            "sample_values": converted_sample_values
        })
    return column_info

def convert_numpy_types(obj):
    if isinstance(obj, np.integer): # 捕获所有 NumPy 整数类型 (int64, int32等)
        return int(obj)
    elif isinstance(obj, np.floating): # 捕获所有 NumPy 浮点数类型 (float64, float32等)
        return float(obj)
    elif isinstance(obj, np.ndarray): # 捕获 NumPy 数组
        # 递归地对数组中的每个元素进行转换
        return [convert_numpy_types(elem) for elem in obj.tolist()]
    elif isinstance(obj, dict): # 递归处理字典
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): # 递归处理列表
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj

@tool
def get_dataframe_summary_tool(df_json_string: str) -> str:
    """
    Provides a summary of the pandas DataFrame, including its shape, column names,
    data types, and number of missing values per column.
    Useful for understanding the dataset structure.

    Args:
        df_json_string (str): A JSON string representation of the DataFrame.
                              This is used because tools can't directly take a DataFrame object.
                              The DataFrame should be converted to JSON (e.g., df.to_json(orient='table'))
                              before passing to this tool.
    Returns:
        str: A JSON string containing the summary.
    """
    try:
        # Reconstruct DataFrame from JSON string. Assuming 'table' orient for robustness.
        df_dict = json.loads(df_json_string)
        column_names = [field['name'] for field in df_dict['schema']['fields']]
        df = pd.DataFrame(df_dict['data'], columns=column_names)
    except Exception as e:
        print(f"Error in get_dataframe_summary reconstruction: {type(e).__name__}: {str(e)}")
        return json.dumps({"error": f"Failed to reconstruct DataFrame: {type(e).__name__}: {str(e)}"})

    summary = {
        "shape": list(df.shape),
        "columns": _get_column_info(df),
        "missing_values_overall": int(df.isnull().sum().sum()),
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)


@tool
def list_available_analysis_tools() -> str:
    """
    Lists the types of analysis capabilities currently available in the system.
    This helps the agent understand what kind of tasks it can plan for.

    Returns:
        str: A JSON string listing available analysis types and their typical data requirements.
    """
    available_tools = { # assuming these are available
        "descriptive_statistics": {
            "description": "Calculate mean, median, mode, count, min, max, unique values for columns. Suitable for numerical and categorical data.",
            "data_requirements": "Any column. For numerical stats, numerical column required. For value counts, any column.",
            "example_tasks": ["Analyze distribution of 'Age'", "Count unique values in 'Category'"]
        },
        "time_series_analysis": {
            "description": "Analyze trends over time. Requires a time-based column (e.g., 'Year', 'Date').",
            "data_requirements": "One time-based column and one or more numerical columns.",
            "example_tasks": ["Plot publication trend over 'Year'", "Average 'Downloads' over 'Month'"]
        },
        "correlation_analysis": {
            "description": "Determine relationships between two or more numerical variables.",
            "data_requirements": "At least two numerical columns.",
            "example_tasks": ["Correlate 'Downloads' and 'Citations'", "Relationship between 'PageCount' and 'AbstractLength'"]
        },
        "categorical_analysis": {
            "description": "Analyze patterns within categorical data, e.g., frequency distributions, group comparisons.",
            "data_requirements": "One or more categorical columns.",
            "example_tasks": ["Top conferences by publication count", "Distribution of 'PaperType'"]
        },
        "text_analysis": {
            "description": "Analyze text content, e.g., keyword extraction, sentiment analysis (if a model is available).",
            "data_requirements": "One or more text columns (e.g., 'Abstract', 'AuthorKeywords').",
            "example_tasks": ["Identify most common keywords in 'Abstract'", "Analyze sentiment of 'Title'"]
        },
        "missing_data_analysis": {
            "description": "Identify and quantify missing data patterns.",
            "data_requirements": "Any column.",
            "example_tasks": ["Check missing values in 'Award'", "Summarize completeness of 'AuthorKeywords'"]
        },
        "network_analysis": {
            "description": "Analyse author network.",
            "data_requirements": "Any column.",
            "example_tasks": ["Examine the correlation between authors.'"]
        },
        "topic_analysis": {
            "description": "Identify and quantify main topics.",
            "data_requirements": "Any column.",
            "example_tasks": ["Topic modelling to identify main topics."]
        }
    }
    return json.dumps(available_tools, indent=2, ensure_ascii=False)

@tool
def check_data_for_task_feasibility(
    column_names: List[str],
    df_json_string: str,
    required_dtypes: List[str] = None
) -> str:
    """
    Checks if specified columns exist in the DataFrame and if their data types are suitable
    for a given analysis task.

    Args:
        column_names (List[str]): A list of column names required for the analysis task.
        df_json_string (str): A JSON string representation of the DataFrame (e.g., df.to_json(orient='table')).
        required_dtypes (List[str], optional): A list of required data types for the columns
                                                 (e.g., ["int64", "float64"] for numerical, "object" for text).
                                                 If None, only checks for column existence.

    Returns:
        str: A JSON string indicating 'feasible' (boolean) and a 'reason' message.
    """
    try:
        df_dict = json.loads(df_json_string)
        df = pd.DataFrame(df_dict['data'], columns=df_dict['schema']['fields'])
    except Exception as e:
        return json.dumps({"feasible": False, "reason": f"Failed to reconstruct DataFrame for feasibility check: {e}"})

    missing_cols = [col for col in column_names if col not in df.columns]
    if missing_cols:
        return json.dumps({"feasible": False, "reason": f"Missing required columns: {', '.join(missing_cols)}"})

    if required_dtypes:
        for col in column_names:
            if str(df[col].dtype) not in required_dtypes:
                return json.dumps({
                    "feasible": False,
                    "reason": f"Column '{col}' has type '{str(df[col].dtype)}', but one of {required_dtypes} was required."
                })

    return json.dumps({"feasible": True, "reason": "All required columns exist and types are suitable."})
