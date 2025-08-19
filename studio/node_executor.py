from state import State
from memory import shared_memory
import pandas as pd
from typing import Dict, Any
import traceback
from langchain_core.messages import SystemMessage, HumanMessage
from helpers import get_llm
from pydantic import BaseModel, Field


def generate_basic_analysis_params(execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Generate parameters for basic analysis based on specific task"""
    
    class BasicParams(BaseModel):
        top_n: int = Field(default=None, description="Number of top items to show")
        min_frequency: int = Field(default=1, description="Minimum frequency threshold")
        reasoning: str = Field(..., description="Reasoning for parameter choices")
    
    sys_prompt = f"""
    Analyze the basic analysis task and generate appropriate parameters:
    Analysis task: {execution_plan['analysis_text']}
    Expected chart type: {execution_plan['chart_type']}
    Time range: {execution_plan['start_year']}-{execution_plan['end_year']}
    
    Please understand the specific requirements of the task and generate only the necessary parameters:
    - If the task involves "top N", "top N items", or "most active N items", set top_n
    - If the task involves frequency filtering, set min_frequency
    - If the task does not need these parameters, set to None or default value
    
    Provide reasoning for the parameter choices.
    """
    
    human_prompt = "Generate basic analysis parameters"
    
    try:
        llm = get_llm(temperature=0.3, max_tokens=512)
        response = llm.with_structured_output(BasicParams).invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        )
        
        return {
            "top_n": response.top_n,
            "min_frequency": response.min_frequency,
            "reasoning": response.reasoning
        }
    except Exception as e:
        print(f"Warning: Basic analysis parameter generation failed: {e}")
        return {
            "top_n": None,
            "min_frequency": 1,
            "reasoning": "Using default parameters"
        }


def generate_topic_analysis_params(execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Generate parameters for topic analysis based on specific task"""
    
    class TopicParams(BaseModel):
        top_n: int = Field(default=10, description="Number of top topics to show")
        min_frequency: int = Field(default=1, description="Minimum frequency threshold")
        min_cooccurrence: int = Field(default=1, description="Minimum co-occurrence threshold")
        min_year: int = Field(default=2010, description="Minimum year for analysis")
        reasoning: str = Field(..., description="Reasoning for parameter choices")
    
    sys_prompt = f"""
    Analyze the topic analysis task and generate appropriate parameters:
    
    Analysis task: {execution_plan['analysis_text']}
    Expected chart type: {execution_plan['chart_type']}
    Time range: {execution_plan['start_year']}-{execution_plan['end_year']}
    
    Please understand the specific requirements of the task and generate only the necessary parameters:
    - If the task involves "main topics", "popular topics", set top_n
    - If the task involves frequency filtering, set min_frequency
    - If the task involves co-occurrence analysis, set min_cooccurrence
    - Adjust parameter values based on the complexity of the task
    
    Provide reasoning for the parameter choices.
    """
    
    human_prompt = "Generate topic analysis parameters"
    
    try:
        llm = get_llm(temperature=0.3, max_tokens=2048)
        response = llm.with_structured_output(TopicParams).invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        )
        
        return {
            "top_n": response.top_n,
            "min_frequency": response.min_frequency,
            "min_cooccurrence": response.min_cooccurrence,
            "min_year": response.min_year,
            "reasoning": response.reasoning
        }
    except Exception as e:
        print(f"Warning: Topic analysis parameter generation failed: {e}")
        return {
            "top_n": 10,
            "min_frequency": 1,
            "min_cooccurrence": 1,
            "min_year": 2010,
            "reasoning": "Using default parameters"
        }


def generate_author_analysis_params(execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Generate parameters for author analysis based on specific task"""
    
    class AuthorParams(BaseModel):
        top_n: int = Field(default=10, description="Number of top authors to show")
        min_frequency: int = Field(default=1, description="Minimum frequency threshold")
        network_threshold: int = Field(default=1, description="Network connection threshold")
        reasoning: str = Field(..., description="Reasoning for parameter choices")
    
    sys_prompt = f"""
    Analyze the author analysis task and generate appropriate parameters:
    
    Analysis task: {execution_plan['analysis_text']}
    Expected chart type: {execution_plan['chart_type']}
    Time range: {execution_plan['start_year']}-{execution_plan['end_year']}
    
    Please understand the specific requirements of the task and generate only the necessary parameters:
    - If the task involves "most active authors", "top authors", set top_n
    - If the task involves frequency filtering, set min_frequency
    - If the task involves network analysis, set network_threshold
    - Adjust parameter values based on the complexity of the task
    
    Provide reasoning for the parameter choices.
    """
    
    human_prompt = "Generate author analysis parameters"
    
    try:
        llm = get_llm(temperature=0.3, max_tokens=512)
        response = llm.with_structured_output(AuthorParams).invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        )
        
        return {
            "top_n": response.top_n,
            "min_frequency": response.min_frequency,
            "network_threshold": response.network_threshold,
            "reasoning": response.reasoning
        }
    except Exception as e:
        print(f"Warning: Author analysis parameter generation failed: {e}")
        return {
            "top_n": 10,
            "min_frequency": 1,
            "network_threshold": 1,
            "reasoning": "Using default parameters"
        }


def execute_basic_analysis(state: State, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute basic analysis using node_analysis_basics"""
    try:
        from node_analysis_basics import execute_basic_analysis_llm
        from node_analysis_basics import BasicAnalysisParameters
        
        # Generate task-specific parameters
        print("Generating parameters for basic analysis...")
        params = generate_basic_analysis_params(execution_plan)
        print(f"Generated parameters: {params}")
        
        # Create BasicAnalysisParameters from execution plan and generated params
        analysis_params = BasicAnalysisParameters(
            analysis_type="basic_analysis",
            question_text=execution_plan["analysis_text"],
            primary_attributes=execution_plan["primary_attributes"] if "primary_attributes" in execution_plan else [],
            secondary_attributes=execution_plan["secondary_attributes"] if "secondary_attributes" in execution_plan else [],
            chart_type=execution_plan["chart_type"],
            target_columns=(execution_plan["primary_attributes"] if "primary_attributes" in execution_plan else []) + 
                          (execution_plan["secondary_attributes"] if "secondary_attributes" in execution_plan else []),
            time_column="Year",
            top_n=params["top_n"],
            min_frequency=params["min_frequency"],
            time_range={"start_year": execution_plan["start_year"], "end_year": execution_plan["end_year"]}
        )
        
        # Execute the analysis
        result = execute_basic_analysis_llm(state["dataframe"], analysis_params, state["iteration_count"])
        
        return {
            "success": True,
            "result": result,
            "module": "basic_analysis_module"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "module": "basic_analysis_module"
        }


def execute_topic_analysis(state: State, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute topic analysis using node_analyse_topics"""
    try:
        from node_analyse_topics import execute_topic_analysis
        from node_analyse_topics import TopicAnalysisParameters
        
        # Generate task-specific parameters
        print("Generating parameters for topic analysis...")
        params = generate_topic_analysis_params(execution_plan)
        print(f"Generated parameters: {params}")
        
        # Prepare state for topic analysis
        analysis_state = state.copy()
        analysis_state["question"] = {
            "question": execution_plan["analysis_text"],
            "handled": False,
            "spec": ""
        }

        analysis_params = TopicAnalysisParameters(
            analysis_type="topic_analysis",
            question_text=execution_plan["analysis_text"],
            primary_attributes=execution_plan["primary_attributes"] if "primary_attributes" in execution_plan else [],
            secondary_attributes=execution_plan["secondary_attributes"] if "secondary_attributes" in execution_plan else [],
            top_n=params["top_n"],
            time_range="all",
            min_frequency=params["min_frequency"],
            min_cooccurrence=params["min_cooccurrence"],
            min_year=params["min_year"]
        )
        
        # Execute the analysis
        result = execute_topic_analysis(analysis_state, analysis_params)
        
        return {
            "success": True,
            "result": result,
            "module": "topic_analysis_module"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "module": "topic_analysis_module"
        }


def execute_author_analysis(state: State, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute author analysis using node_analyse_authors"""
    try:
        from node_analyse_authors import execute_author_network_analysis
        from node_analyse_authors import AuthorAnalysisParameters
        
        # Generate task-specific parameters
        print("Generating parameters for author analysis...")
        params = generate_author_analysis_params(execution_plan)
        print(f"Generated parameters: {params}")
        
        # Prepare state for author analysis
        analysis_state = state.copy()
        analysis_state["question"] = {
            "question": execution_plan["analysis_text"],
            "handled": False,
            "spec": ""
        }
        
        analysis_params = AuthorAnalysisParameters(
            analysis_type="author_analysis",
            question_text=execution_plan["analysis_text"],
            primary_attributes=execution_plan["primary_attributes"] if "primary_attributes" in execution_plan else [],
            secondary_attributes=execution_plan["secondary_attributes"] if "secondary_attributes" in execution_plan else [],
            time_range="all"
        )
        
        # Execute the analysis
        result = execute_author_network_analysis(analysis_state, analysis_params)
        
        return {
            "success": True,
            "result": result,
            "module": "author_analysis_module"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "module": "author_analysis_module"
        }


def update_analysis_history(state: State, module_name: str, success: bool) -> State:
    """Update analysis history with the current execution"""
    new_state = state.copy()
    
    # Initialize analysis_history if it doesn't exist
    if "analysis_history" not in new_state or new_state["analysis_history"] is None:
        new_state["analysis_history"] = {
            "module_counts": {
                "basic_analysis_module": 0,
                "topic_analysis_module": 0,
                "author_analysis_module": 0
            },
            "recent_analyses": []
        }
    
    # Update module counts
    if module_name in new_state["analysis_history"]["module_counts"]:
        new_state["analysis_history"]["module_counts"][module_name] += 1
    
    # Add to recent analyses
    question_text = ""
    if "question" in new_state and new_state["question"] is not None:
        if isinstance(new_state["question"], dict) and "question" in new_state["question"]:
            question_text = new_state["question"]["question"]
    
    analysis_record = {
        "module": module_name,
        "question": question_text,
        "iteration": new_state["iteration_count"],
        "success": success,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    new_state["analysis_history"]["recent_analyses"].append(analysis_record)
    
    # Keep only last 10 analyses
    if len(new_state["analysis_history"]["recent_analyses"]) > 10:
        new_state["analysis_history"]["recent_analyses"] = new_state["analysis_history"]["recent_analyses"][-10:]
    
    return new_state


def executor(state: State) -> State:
    """
    Execute analysis based on execution plan
    """
    current_iteration = state["iteration_count"]
    result = None
    print(f"PROCESS - Analysis Execution - at iteration: {current_iteration} - START")
    
    # Check if execution plan exists
    if "execution_plan" not in state:
        raise ValueError("No execution_plan found in state")
    
    execution_plan = state["execution_plan"]
    target_function = execution_plan["target_function"]
    module_name = execution_plan["module_name"]
    
    print(f"Executing {target_function} for module: {module_name}")
    print(f"Analysis: {execution_plan['analysis_text']}")
    print(f"Chart type: {execution_plan['chart_type']}")
    
    # Execute based on target function
    if target_function == "analyse_basics":
        execution_result = execute_basic_analysis(state, execution_plan)
    elif target_function == "analyse_topics":
        execution_result = execute_topic_analysis(state, execution_plan)
    elif target_function == "analyse_authors":
        execution_result = execute_author_analysis(state, execution_plan)
    else:
        raise ValueError(f"Unknown target function: {target_function}")
    
    # Update state with execution results
    new_state = state.copy()
    if "question" in state and state["question"] is not None:
        new_state["question"] = state["question"]
    
    if execution_result["success"]:
        result = execution_result["result"]
        
        # Debug result structure
        print(f"DEBUG: result type: {type(result)}")
        print(f"DEBUG: result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        # Update state with analysis results
        if isinstance(result, dict):
            if "visualizations" in result:
                # new_state["visualizations"] = result["visualizations"]

                new_state["visualizations"] = {
                            "visualizations": result["visualizations"]
                        }
                
            if "facts" in result:
                new_state["facts"] = result["facts"]
            if "insights" in result:
                new_state["insights"] = result["insights"]
            
            print(f"Analysis executed successfully:")
            try:
                visualizations = result['visualizations'] if 'visualizations' in result and result['visualizations'] is not None else []
                viz_count = len(visualizations)
                print(f"   Visualizations: {viz_count}")
            except Exception as e:
                print(f"   Error getting visualizations count: {e}")
            try:
                facts = result['facts'] if 'facts' in result and result['facts'] is not None else {}
                facts_count = len(facts)
                print(f"   Facts: {facts_count}")
            except Exception as e:
                print(f"   Error getting facts count: {e}")
            try:
                insights = result['insights'] if 'insights' in result and result['insights'] is not None else []
                insights_count = len(insights)
                print(f"   Insights: {insights_count}")
            except Exception as e:
                print(f"   Error getting insights count: {e}")
        else:
            print(f"Warning: result is not a dict, type: {type(result)}")
            # Set empty results if result is not a dict
            new_state["visualizations"] = {"visualizations": []}
            new_state["facts"] = {}
            new_state["insights"] = []
        
    else:
        print(f"Analysis execution failed: {execution_result['error']}")
        # Set empty results on failure
        new_state["visualizations"] = {"visualizations": []}
        new_state["facts"] = {}
        new_state["insights"] = []
    
    # Update analysis history
    new_state = update_analysis_history(new_state, module_name, execution_result["success"])
    
    # Print analysis history
    if "analysis_history" in new_state and new_state["analysis_history"] is not None:
        counts = new_state["analysis_history"]["module_counts"]
        print(f"Analysis History:")
        print(f"   Basic: {counts['basic_analysis_module'] if 'basic_analysis_module' in counts else 0}")
        print(f"   Topic: {counts['topic_analysis_module'] if 'topic_analysis_module' in counts else 0}")
        print(f"   Author: {counts['author_analysis_module'] if 'author_analysis_module' in counts else 0}")
    
    # Save state to memory
    if result is not None:
        iteration_history = state["iteration_history"] if "iteration_history" in state and state["iteration_history"] is not None else []
        new_state["iteration_history"] = iteration_history + [result]
    else:
        iteration_history = state["iteration_history"] if "iteration_history" in state and state["iteration_history"] is not None else []
        new_state["iteration_history"] = iteration_history
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")
    
    print(f"PROCESS - Analysis Execution - at iteration: {current_iteration} - DONE")
    
    return new_state


def test_executor():
    """
    Test function for executor
    """
    # Create a mock state with execution plan
    mock_state = {
        "iteration_count": 0,
        "execution_plan": {
            "analysis_text": "Analyze research themes using topic modeling",
            "analysis_type": "topic_analysis_module",
            "chart_type": "topic_clustering",
            "start_year": 1990,
            "end_year": 2024,
            "reasoning": "Content analysis for research themes",
            "target_function": "analyse_topics",
            "function_params": {},
            "module_name": "topic_analysis_module",
            "expected_outputs": ["visualizations", "facts", "insights"],
            "validation_rules": []
        },
        "dataframe": pd.DataFrame({
            "Title": ["Test Paper 1", "Test Paper 2"],
            "Abstract": ["Test abstract 1", "Test abstract 2"],
            "Year": [2020, 2021]
        })
    }
    
    print("=== Testing Executor ===")
    print(f"Target function: {mock_state['execution_plan']['target_function']}")
    print(f"Module: {mock_state['execution_plan']['module_name']}")
    print("=" * 40)
    
    try:
        result_state = executor(mock_state)
        
        print(f"\nExecution completed")
        visualizations = result_state['visualizations'] if 'visualizations' in result_state and result_state['visualizations'] is not None else []
        print(f"Visualizations: {len(visualizations)}")
        facts = result_state['facts'] if 'facts' in result_state and result_state['facts'] is not None else {}
        print(f"Facts: {len(facts)}")
        insights = result_state['insights'] if 'insights' in result_state and result_state['insights'] is not None else []
        print(f"Insights: {len(insights)}")
        
        if "analysis_history" in result_state:
            counts = result_state["analysis_history"]["module_counts"]
            print(f"Analysis history updated: {counts}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()
    
    return result_state


if __name__ == "__main__":
    test_executor()
