from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

# TODO: add the tools that Zefei developed for better research direction suggestion

# Node 1: analysis_planner_node_with_react's step 1
def data_exploration_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 1: Using ReAct agent to explore data
    """
    print("--- Node: Data Exploration - Using ReAct Agent ---")
    updated_state_data = {}

    # Validate dataframe availability
    df = state.get("dataframe")
    if not isinstance(df, pd.DataFrame) or df.empty:
        error_msg = "Error: DataFrame is not loaded correctly or is empty."
        print(error_msg)
        updated_state_data["exploration_result"] = error_msg
        updated_state_data["messages"] = state.get("messages", []) + [AIMessage(content=error_msg)]
        return updated_state_data

    df_json_string = df.head(2).to_json(orient='table', index=False)
    # Prepare message for planning agent
    current_human_message_content = (
        f"Plan the most insightful data analysis tasks for this dataset. "
        f"The dataset's structure, columns, and missing values can be examined using the 'get_dataframe_summary_tool' tool."
    )

    messages_for_agent = state.get("messages", []) + [HumanMessage(content=current_human_message_content)]

    # Import required modules
    from helpers import get_llm
    from tools import get_dataframe_summary_tool, list_available_analysis_tools, check_data_for_task_feasibility

    # Initialize LLM and tools
    llm = get_llm(max_completion_tokens=4096)
    tools = [get_dataframe_summary_tool, list_available_analysis_tools, check_data_for_task_feasibility]

    try:
        # Step 1: Use ReAct agent to explore the dataset
        print("Step 1: Using ReAct agent to explore data...")

        # Create ReAct agent with default settings to avoid parameter conflicts
        from langgraph.prebuilt import create_react_agent

        react_agent = create_react_agent(
            model=llm,
            tools=tools,
            # debug=True
        )

        # Create exploration message to guide agent behavior
        exploration_message = HumanMessage(content=f"""
        You are an expert data analyst analyzing IEEE VIS publication data from 1990 till now.
        The DataFrame JSON string representation is:
        {df_json_string}

        Use provided tools to understand the data structure, columns, and data types, and see what analysis options are available to verify what kinds of analysis are feasible
        Based on your exploration, recommend 2 analysis tasks that would be most insightful. Focus on patterns in publications, authors, research trends, collaborations, etc.
        """)

        # Use exploration message instead of original messages
        exploration_messages = [exploration_message]

        # Invoke ReAct agent with exploration messages
        react_result = react_agent.invoke({
            "messages": exploration_messages,
            "df_json_string": df_json_string
            # "dataframe": df  # Pass dataframe for tools to access
        })

        # Extract agent's exploration results
        exploration_result = react_result["messages"][-1].content
        print("-------ReAct Agent Exploration Result---------")
        print(exploration_result)

        # Save exploration result to state for next node
        updated_state_data["exploration_result"] = exploration_result
        updated_state_data["messages"] = messages_for_agent + [
            AIMessage(content="Data exploration completed using ReAct agent")
        ]

        print("Node: Data Exploration - ReAct agent exploration completed successfully.")
        print("--------------------------------------------")

    except Exception as e:
        error_msg = f"Error in ReAct exploration: {e}"
        print(error_msg)
        print(f"Exception details: {type(e).__name__}: {str(e)}")

        updated_state_data["exploration_result"] = error_msg
        updated_state_data["messages"] = messages_for_agent + [AIMessage(content=error_msg)]

    return updated_state_data