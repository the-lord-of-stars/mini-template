from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json

from helpers import get_llm, get_dataset_info, query_by_sql
from state import State
from memory import shared_memory
from sandbox import run_in_sandbox, run_in_sandbox_with_venv
import time

# Configuration for retry mechanism and timeouts
FACTS_TIMEOUT_CONFIG = {
    "timeout": 30,  # seconds
    "max_retries": 0,
    "retry_delays": [1, 2, 4]  # seconds between retries
}

class ResponseFormatter(BaseModel):
    code: str = Field(description="The python code to get the facts about the dataset")


def execute_facts_with_retry(code: str, config: dict = None):
    """
    Execute facts analysis code with retry mechanism and timeout handling
    
    Args:
        code: Python code to execute
        config: Configuration dictionary with timeout and retry settings
    
    Returns:
        dict: Execution result with stdout, stderr, and exit_code
    """
    if config is None:
        config = FACTS_TIMEOUT_CONFIG
    
    timeout = config["timeout"]
    max_retries = config["max_retries"]
    retry_delays = config["retry_delays"]
    
    for attempt in range(max_retries + 1):
        try:
            print(f"Executing facts analysis... (attempt {attempt + 1}/{max_retries + 1})")
            
            # Execute the code
            result = run_in_sandbox_with_venv(code)
            
            # Check if execution was successful
            if result["exit_code"] == 0:
                print(f"Facts analysis completed successfully!")
                return result
            else:
                print(f"Facts analysis failed with exit code {result['exit_code']}")
                if attempt < max_retries:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Returning last result.")
                    return result
                    
        except Exception as e:
            error_msg = str(e)
            print(f"Facts analysis error (attempt {attempt + 1}): {error_msg}")
            
            # Check if it's a timeout error
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                if attempt < max_retries:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    print(f"Timeout detected. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached after timeout errors.")
                    return {
                        "stdout": "",
                        "stderr": f"Execution timed out after {max_retries + 1} attempts. Last error: {error_msg}",
                        "exit_code": 1
                    }
            else:
                # Non-timeout error, don't retry
                print("Non-timeout error detected. Not retrying.")
                return {
                    "stdout": "",
                    "stderr": f"Execution failed: {error_msg}",
                    "exit_code": 1
                }
    
    # This should never be reached, but just in case
    return {
        "stdout": "",
        "stderr": "Unexpected error in retry mechanism",
        "exit_code": 1
    }


def get_facts(state: State):
    """
    Generate SQL query to select the data (based on the topic)
    """

    fact_term = "facts"

    # Get path of the main program being executed
    dataset_info = get_dataset_info(state['select_data_state']['dataset_path'])

    sys_prompt = f"""
        You are a data analyst who write python script to analyse the dataset by understanding the fundamental statistics.
        CRITICAL: This is a CSV file. Use pd.read_csv(path) with NO separator parameter, or explicitly use delimiter=','.
        DO NOT use sep='\t' or assume it's a tab-separated file.

        The dataset is as follows:
        {dataset_info}

        Please generate a python code to analyse the dataset, the goal is to get {fact_term} about the dataset that can answer the given question.

        You should read the dataset from the following path:
        {state['select_data_state']['dataset_path']}

        Requirements:
        1. only get the most relevant {fact_term}, don't generate too many.
        2. make the code concise
        3. example {fact_term}: statistics, top k papers, trends, most cited authors, etc. you are free to choose the {fact_term} that are most relevant to the question. just keep in mind that you are an expert in data analysis.
        4. the {fact_term} should be relevant to the question
        5. irrelevant or meaningless facts should be avoided
        6. each {fact_term} should be printed as:
            ### Begin of {fact_term}
            <{fact_term}>
            ### End of {fact_term}
        7. feel free to use python libraries to help you analyse the dataset. supported libraries: pandas, numpy, matplotlib, seaborn, networkx.
        8. make sure the code is executable.
        9. if the question is too complex and can not be solved by a short code, just ignore it and do the most basic and simple analysis.
        10. keep the code short and concise.
        
        GOOD EXAMPLES OF FACTS CODE:
        ```python
        import pandas as pd
        import numpy as np
        
        df = pd.read_csv('dataset.csv')
        
        # Calculate citation statistics
        print("### Begin of facts")
        print(f"Total papers: {{len(df)}}")
        print(f"Average citation count: {{df['CitationCount_CrossRef'].mean():.2f}}")
        print(f"Most cited paper: {{df.loc[df['CitationCount_CrossRef'].idxmax(), 'Title']}}")
        print(f"Citation count: {{df['CitationCount_CrossRef'].max()}}")
        print("### End of facts")
        ```
    """

    human_prompt = f"""
    I would like to explore the dataset about the topic of {state['topic']}.
    The current analysis question is: {state['question']['question']}.
    Please generate the python code.
    """

    llm = get_llm(temperature=0, max_tokens=8192)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    # Run the code in sandbox with retry mechanism
    print(f"Starting facts analysis for question: {state['question']['question']}")
    result = execute_facts_with_retry(response.code)
    print(result)

    new_state = state.copy()
    new_state["facts"] = {
        "code": response.code,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "exit_code": result["exit_code"]
    }

    # Save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")

    return new_state
