import json
import uuid
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel
# from sandbox import run_visualization_in_sandbox
from state import State, Visualization
from helpers import get_llm, get_dataset_info

def fig_line_trend(dff, bucket_col, nums, objective):
    # Year-only line chart
    if not nums:
        agg = dff.groupby(bucket_col, as_index=False).size().rename(columns={"size":"count"})
        fig = px.line(agg, x=bucket_col, y="count", markers=True, title=objective)
    else:
        traces = []
        for ycol in nums:
            agg = dff.groupby(bucket_col, as_index=False)[ycol].mean()
            traces.append(go.Scatter(x=agg[bucket_col], y=agg[ycol], mode="lines+markers", name=f"mean({ycol})"))
        fig = go.Figure(traces)
        fig.update_layout(title=objective, xaxis_title=bucket_col, yaxis_title="mean(value)")
    return fig

def fig_scatter_corr(dff, cols, objective):
    """Scatter plot showing correlation"""
    if len(cols) < 2:
        return None
    x_col, y_col = cols[0], cols[1]
    fig = px.scatter(dff, x=x_col, y=y_col, title=objective)
    return fig

def fig_bar_group(dff, bucket_col, cats, objective):
    """Grouped bar chart"""
    if not cats:
        return None
    cat_col = cats[0]
    agg = dff.groupby([bucket_col, cat_col], as_index=False).size().rename(columns={"size":"count"})
    fig = px.bar(agg, x=bucket_col, y="count", color=cat_col, title=objective)
    return fig

def fig_box_by_category(dff, cols, objective):
    """Box plot grouped by category"""
    if len(cols) < 2:
        return None
    # Assume first is numeric column, second is categorical column
    num_col, cat_col = cols[0], cols[1]
    fig = px.box(dff, x=cat_col, y=num_col, title=objective)
    return fig

def fig_histogram(dff, cols, objective):
    """Histogram"""
    if not cols:
        return None
    col = cols[0]
    fig = px.histogram(dff, x=col, title=objective)
    return fig

def fig_heatmap_xy(dff, cols, bucket_col, objective):
    """Heatmap"""
    if len(cols) < 2:
        return None
    # Create cross table
    pivot_table = pd.crosstab(dff[cols[0]], dff[cols[1]])
    fig = px.imshow(pivot_table, title=objective)
    return fig

def split_cols_by_dtype(df: pd.DataFrame, cols: list, time_col_hint: str | None = None):
    """Function copied from your colleague's code"""
    nums, cats, times = [], [], []
    time_col = None
    for c in cols:
        lc = c.lower()
        if time_col is None and (("year" in lc) or ("date" in lc) or ("time" in lc)):
            time_col = c
        if c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                nums.append(c)
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                times.append(c)
                if time_col is None: time_col = c
            else:
                cats.append(c)
    if time_col is None and time_col_hint and time_col_hint in df.columns:
        time_col = time_col_hint
    return nums, cats, times, time_col


def filter_by_time_scope(df: pd.DataFrame, time_scope: dict, time_col: str):
    """Function copied from your colleague's code"""
    dff = df.copy()
    start, end, interval = time_scope.get("start_year"), time_scope.get("end_year"), time_scope.get("interval_years", 1)
    if start and end and time_col in dff.columns:
        dff = dff[(dff[time_col] >= start) & (dff[time_col] <= end)]
    bucket_col = time_col
    if interval and interval > 1:
        dff[bucket_col] = (dff[time_col] // interval) * interval
    return dff, bucket_col


def get_visualization_tool_schemas():
    """Define function schema for visualization tool selection"""
    return [
        {
            "type": "function",
            "function": {
                "name": "select_visualization",
                "description": "Select the most appropriate visualization tool based on analysis questions and data characteristics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["line_trend", "scatter_corr", "bar_group", "box_by_category", "histogram",
                                     "heatmap_xy"],
                            "description": "Selected visualization operation"
                        },
                        "target_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of target column names"
                        },
                        "chart_type": {
                            "type": "string",
                            "description": "Chart type description"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed reasoning for choosing this visualization"
                        },
                        "objective": {
                            "type": "string",
                            "description": "Specific objective of the visualization"
                        },
                        "time_scope": {
                            "type": "object",
                            "properties": {
                                "start_year": {"type": "integer"},
                                "end_year": {"type": "integer"},
                                "interval_years": {"type": "integer"}
                            },
                            "additionalProperties": False,
                            "description": "Time range settings"
                        }
                    },
                    "required": ["op", "target_columns", "chart_type", "reasoning", "objective"],
                    "additionalProperties": False
                }
            }
        }
    ]


def get_visualization_choice_with_tools(state: State) -> dict:
    """Use function calling to let LLM select the most appropriate visualization tool"""

    # Get current state information
    current_question = state.get("question", {})
    current_facts = state.get("facts", {})
    current_insights = state.get("insights", [])
    topic = state.get("topic", "")

    # Get dataset information
    dataset_path = state.get('select_data_state', {}).get('dataset_path', '')
    dataset_info = get_dataset_info(dataset_path) if dataset_path else "No dataset info available"

    # Analyze available columns
    df = state.get("dataframe")
    if df is None:
        raise ValueError("No dataframe found in state")

    available_columns = list(df.columns)

    # Analyze column data types
    nums, cats, times, time_col = split_cols_by_dtype(df, available_columns)

    system_prompt = f"""
    You are a professional data visualization expert. Based on the current analysis questions, facts, and insights, select the most appropriate visualization tool.

    Dataset information:
    {dataset_info[:500]}...

    Available column information:
    - Numeric columns: {nums}
    - Categorical columns: {cats} 
    - Time columns: {times}
    - Primary time column: {time_col}
    - All columns: {available_columns}

    Available visualization tools and their requirements:
    1. line_trend: Time trend line chart (requires: time column + numeric column or count)
    2. scatter_corr: Scatter plot showing correlation (requires: two numeric columns)
    3. bar_group: Grouped bar chart (requires: time column + categorical column)
    4. box_by_category: Box plot grouped by category (requires: numeric column + categorical column)
    5. histogram: Histogram (requires: numeric column)
    6. heatmap_xy: Heatmap (requires: time/categorical column + categorical column)

    Selection criteria:
    - Choose the most appropriate visualization based on question type
    - Ensure selected columns exist and types match
    - Prioritize chart types that best answer the current question
    - Consider data distribution characteristics and analysis objectives

    Current analysis context:
    - Topic: {topic}
    - Question: {current_question.get('question', 'No question')}
    - Fact analysis results: {current_facts.get('stdout', 'No facts')[:300]}...
    - Existing insights: {current_insights}
    """

    human_prompt = f"""
    Based on the above information, please call the select_visualization function to choose the most appropriate visualization tool to display the analysis results.

    Requirements:
    1. The selected op must be from the available tools
    2. target_columns must exist in the dataset and types must match
    3. Provide clear reasoning for the selection
    4. Set appropriate time range (if applicable)
    5. The chart should best answer the current analysis question

    Please call the select_visualization function.
    """

    llm = get_llm(temperature=0.3, max_tokens=4096)
    bound_llm = llm.bind_tools(get_visualization_tool_schemas())

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    # Call LLM
    response = bound_llm.invoke(messages)

    # Parse tool call results
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "select_visualization":
            return tool_call["args"]

    # If no tool call, use fallback
    return {
        "op": "line_trend",
        "target_columns": [time_col] if time_col else available_columns[:2],
        "chart_type": "Line Chart",
        "reasoning": "Fallback selection due to no tool call",
        "objective": f"Analyze {topic}",
        "time_scope": {}
    }


def draw(state: State) -> State:
    """
    New drawing node: Use function calling to intelligently select visualization tools and generate charts
    """
    current_iteration = state.get("iteration_count", 0)
    print(f"Current iteration: {current_iteration}")
    # max_iterations = state.get("max_iterations", 3)
    # current_insights = state.get("insights", [])
    try:
        # 1. Let LLM select the most appropriate visualization tool
        viz_choice = get_visualization_choice_with_tools(state)

        print(f"LLM selected visualization tool: {viz_choice.get('op')}")
        print(f"Target columns: {viz_choice.get('target_columns')}")
        print(f"Selection reasoning: {viz_choice.get('reasoning')}")

        # 2. Prepare data
        df = state.get("dataframe")
        if df is None:
            raise ValueError("No dataframe available")

        dff = df.copy()
        cols = viz_choice.get("target_columns", [])
        objective = viz_choice.get("objective", f"Analysis of {viz_choice.get('chart_type', 'data')}")

        # 3. Analyze column types
        nums, cats, times, time_col = split_cols_by_dtype(dff, cols)
        bucket_col = time_col

        # 4. Apply time filtering (if any)
        time_scope = viz_choice.get("time_scope", {})
        if time_col and time_scope:
            dff, bucket_col = filter_by_time_scope(dff, time_scope, time_col)

        # 5. Call corresponding plotting function based on selected op
        fig = None
        success = False
        error_msg = ""
        op = viz_choice.get("op")

        try:
            if op == "line_trend" and bucket_col:
                fig = fig_line_trend(dff, bucket_col, [c for c in nums if c != time_col], objective)
            elif op == "scatter_corr":
                fig = fig_scatter_corr(dff, cols, objective)
            elif op == "bar_group" and bucket_col:
                fig = fig_bar_group(dff, bucket_col, [c for c in cats if c != time_col], objective)
            elif op == "box_by_category":
                fig = fig_box_by_category(dff, cols, objective)
            elif op == "histogram":
                fig = fig_histogram(dff, cols, objective)
            elif op == "heatmap_xy":
                fig = fig_heatmap_xy(dff, cols, bucket_col, objective)
            else:
                error_msg = f"Unsupported operation: {op} or missing required columns"

            if fig is not None:
                success = True
                # 统一样式
                fig.update_layout(
                    template="plotly_white",
                    # width=1200,
                    height=600,
                    margin=dict(l=60, r=30, t=60, b=40),
                    autosize=True,
                )

        except Exception as e:
            error_msg = f"Error generating {op} plot: {str(e)}"
            print(f"Drawing error: {error_msg}")

        # 6. Generate HTML for the figure (if successful)
        figure_html = ""
        if success and fig:
            try:
                # Generate HTML snippet for the figure
                figure_html = fig.to_html(
                    full_html=False,           # Don't include full HTML structure
                    include_plotlyjs='cdn'     # Include plotly.js from CDN
                )
                print(f"✅ Figure HTML generated successfully")
            except Exception as e:
                print(f"Failed to generate figure HTML: {e}")
                figure_html = ""

        # 7. Create visualization result
        visualization = Visualization(
            insight=f"Generated {viz_choice.get('chart_type', 'chart')} showing {objective}",
            chart_type=viz_choice.get('chart_type', 'Unknown'),
            altair_code="",  # Using plotly here, not altair
            description=viz_choice.get('reasoning', 'Auto-generated visualization'),
            is_appropriate=success,
            image_path="",  # No image path needed
            success=success,
            figure_object=figure_html,  # Save the HTML content
            code=f"Generated using {op} with columns {cols}"
        )

        # 8. Update state
        new_state = state.copy()

        # Update visualizations - replace with current visualization only
        new_state["visualizations"] = {
            "visualizations": [visualization]
        }


        # Decide whether to continue based on iteration count and insights
        # should_continue = current_iteration < max_iterations and len(current_insights) > 0
        # new_state["should_continue"] = should_continue
        # print(f"Iteration {current_iteration}/{max_iterations}, continuing: {should_continue}")

        print(f"Visualization {'successful' if success else 'failed'}: {op}")
        if not success:
            print(f"Error message: {error_msg}")
        
        # Increment iteration count after visualization is complete
        # new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        return new_state

    except Exception as e:
        print(f"draw_new_node execution failed: {e}")
        # Return error state
        error_visualization = Visualization(
            insight=f"Failed to generate visualization: {str(e)}",
            chart_type="error",
            altair_code="",
            description=f"Error occurred: {str(e)}",
            is_appropriate=False,
            image_path="",
            success=False,
            figure_object=None,
            code=""
        )

        new_state = state.copy()
        new_state["visualizations"] = {
            "visualizations": [error_visualization]
        }

        # Decide whether to continue based on iteration count and insights
        # should_continue = current_iteration < max_iterations and len(current_insights) > 0
        # new_state["should_continue"] = should_continue
        # print(f"Iteration {current_iteration}/{max_iterations}, continuing: {should_continue}")
        # Increment iteration count after visualization is complete
        # new_state["iteration_count"] = state.get("iteration_count", 0) + 1

        return new_state


# Test function
def test_draw():
    """Test the new drawing node"""

    # Create test state
    test_state = {
        "topic": "IEEE VIS research trends",
        "question": {"question": "How has the number of publications changed over time?"},
        "facts": {"stdout": "Publication count shows increasing trend from 2000 to 2020"},
        "insights": ["Research activity has grown significantly", "Peak activity in recent years"],
        "select_data_state": {"dataset_path": "./dataset.csv"},
        "dataframe": pd.read_csv("./dataset.csv") if os.path.exists("./dataset.csv") else None,
        "visualizations": {"visualizations": []}
    }

    if test_state["dataframe"] is not None:
        result = draw(test_state)
        print("Test completed!")
        print(f"Number of generated visualizations: {len(result['visualizations']['visualizations'])}")

        # Print result details
        for i, viz in enumerate(result['visualizations']['visualizations']):
            print(f"Visualization {i + 1}:")
            print(f"  Success: {viz.get('success', False)}")
            print(f"  Type: {viz.get('chart_type', 'Unknown')}")
            print(f"  Description: {viz.get('description', 'No description')[:100]}...")
    else:
        print("Test data file does not exist")


if __name__ == "__main__":
    test_draw()