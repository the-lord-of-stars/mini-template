from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

AnalysisType = Literal[
    "descriptive_statistics",
    "time_series_analysis",
    "correlation_analysis",
    "categorical_analysis",
    "text_analysis",
    "missing_data_analysis",
    "author_analysis",
    "citation_analysis"
]

class AnalysisTask(BaseModel):
    task_id: str = Field(..., description="A unique string identifier for the task (e.g., 'task_1').")
    objective: str = Field(..., description="A concise string describing the goal of the task.")
    analysis_type: AnalysisType = Field(..., description="The type of analysis to perform, must be one of the predefined types.")
    relevant_columns: List[str] = Field(..., description="A list of column names (strings) relevant to this task.")
    details: str = Field(..., description="A more detailed explanation of what needs to be done for this task, including specific calculations or plots.")

class AnalysisTasksOutput(BaseModel):
    tasks: List[AnalysisTask] = Field(..., description="A list of proposed data analysis tasks. Return an empty list if no meaningful tasks are identified.")


# Node 2: analysis_planner_node_with_react's step 2
def task_planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Converting exploration to structured tasks
    """
    print("--- Node: Task Planning - Converting exploration to structured tasks ---")
    updated_state_data = {}

    # Get exploration result from previous node
    exploration_result = state.get("exploration_result", "")
    df = state.get("dataframe")
    messages_for_agent = state.get("messages", [])

    from tools import list_available_analysis_tools
    available_tools_info = list_available_analysis_tools.func()

    # Check if exploration had issues and enhance with basic info if needed
    if not exploration_result or "error" in exploration_result.lower() or "couldn't retrieve" in exploration_result.lower():
        print("Exploration had issues, enhancing with basic DataFrame info...")

        if isinstance(df, pd.DataFrame) and not df.empty:
            # Use basic dataframe info since tools failed
            enhanced_info = f"""
            Based on direct DataFrame inspection:
            - Dataset shape: {df.shape}
            - Available columns: {list(df.columns)}
            - Data types: {df.dtypes.to_dict()}
            - Sample data: {df.head(2).to_dict()}

            Available analysis tools (from previous exploration):
            {available_tools_info}

            This is IEEE VIS publication data from 1990 till now, suitable for publication analysis.
            """

            # Use enhanced info for task planning
            exploration_result = enhanced_info
            print("Enhanced exploration result with basic DataFrame info")
        else:
            error_msg = "Cannot proceed: no valid DataFrame available"
            print(error_msg)
            updated_state_data["analysis_tasks"] = []
            updated_state_data["messages"] = messages_for_agent + [AIMessage(content=error_msg)]
            return updated_state_data

    # Import required modules
    from helpers import get_llm

    # Initialize LLM
    llm = get_llm(temperature=0.8, max_tokens=4096)

    try:
        # Step 2: Convert exploration results to structured tasks using structured LLM
        print("Step 2: Converting exploration to structured tasks...")

        structured_llm = llm.with_structured_output(AnalysisTasksOutput)

        structure_prompt = f"""
        Based on the following data exploration and analysis, create exactly 2 analysis tasks in the required structured format.

        Data Exploration Results:
        {exploration_result}

        Additional Context:
        - Dataset shape: {df.shape if df is not None else 'Unknown'}
        - Available columns: {list(df.columns) if df is not None else 'Unknown'}

        Create 2 AnalysisTask objects that would provide the most insights for IEEE VIS publication analysis.
        Each task should have:
        - task_id: unique identifier
        - objective: clear goal
        - analysis_type: one of the predefined types
        - relevant_columns: specific columns to analyze
        - details: specific steps to take

        Analysis types available: descriptive_statistics, time_series_analysis, correlation_analysis, 
        categorical_analysis, text_analysis, missing_data_analysis, author_analysis, citation_analysis
        """

        structured_output = structured_llm.invoke([HumanMessage(content=structure_prompt)])
        analysis_tasks = structured_output.tasks

        print("-------Task Planning Output---------")
        print(f"Generated {len(analysis_tasks)} tasks")
        for i, task in enumerate(analysis_tasks, 1):
            print(f"Task {i}: {task.objective} ({task.analysis_type})")
            print(f"  Columns: {task.relevant_columns}")

        # Convert Pydantic objects to dictionaries for JSON serialization
        analysis_tasks_dicts = [task.dict() for task in analysis_tasks]

        # Update state with generated tasks
        updated_state_data["analysis_tasks"] = analysis_tasks_dicts
        updated_state_data["messages"] = messages_for_agent + [
            AIMessage(content=f"Generated {len(analysis_tasks)} structured analysis tasks")
        ]

        print(f"Node: Task Planning - Successfully converted exploration to {len(analysis_tasks)} structured tasks.")
        print("------------------------------------")

    except Exception as e:
        error_msg = f"Error in structured task planning: {e}"
        print(error_msg)
        print(f"Exception details: {type(e).__name__}: {str(e)}")

        # Create fallback default tasks if planning fails
        if isinstance(df, pd.DataFrame) and not df.empty:
            default_tasks = [
                {
                    "task_id": "task_1",
                    "objective": "[FALLBACK] Descriptive statistics analysis",
                    "analysis_type": "descriptive_statistics",
                    "relevant_columns": list(df.columns)[:3],
                    "details": "Perform basic statistical analysis on key columns"
                },
                {
                    "task_id": "task_2",
                    "objective": "[FALLBACK] Publication trends over time",
                    "analysis_type": "time_series_analysis",
                    "relevant_columns": [col for col in df.columns if 'year' in col.lower()][:1] or [df.columns[0]],
                    "details": "Analyze publication trends and patterns over time"
                }
            ]
        else:
            default_tasks = []

        updated_state_data["analysis_tasks"] = default_tasks
        updated_state_data["messages"] = messages_for_agent + [AIMessage(content=error_msg)]

    return updated_state_data