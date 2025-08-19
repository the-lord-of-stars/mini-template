from agents.star.state import State
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from agents.star.memory import shared_memory


class ExecutionPlan(BaseModel):
    """Execution plan for analysis modules"""
    analysis_text: str = Field(..., description="The analysis text to be executed")
    analysis_type: str = Field(..., description="The type of analysis to be executed")
    chart_type: str = Field(..., description="The type of chart to be created")
    start_year: int = Field(..., description="The start year of the analysis")
    end_year: int = Field(..., description="The end year of the analysis")
    reasoning: str = Field(..., description="The reasoning for the analysis")
    target_function: str = Field(..., description="Name of the function to call")
    function_params: Dict[str, Any] = Field(..., description="Parameters for the function")
    module_name: str = Field(..., description="Name of the analysis module")
    expected_outputs: List[str] = Field(..., description="Expected output types")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules for the plan")


def generate_execution_plan(decision: Dict[str, Any]) -> ExecutionPlan:
    """Generate execution plan based on analysis decision"""
    
    module_mapping = {
        "basic_analysis_module": {
            "function": "analyse_basics",
            "outputs": ["visualizations", "facts", "insights"]
        },
        "topic_analysis_module": {
            "function": "analyse_topics",
            "outputs": ["visualizations", "facts", "insights"]
        },
        "author_analysis_module": {
            "function": "analyse_authors",
            "outputs": ["visualizations", "facts", "insights"]
        }
    }
    
    suggested_module = decision["suggested_module"]
    
    if suggested_module not in module_mapping:
        raise ValueError(f"Unknown module: {suggested_module}")
    
    module_config = module_mapping[suggested_module]
    
    return ExecutionPlan(
        analysis_text=decision["analysis_text"],
        analysis_type=suggested_module,
        chart_type=decision["suggested_chart_type"],
        start_year=decision["time_range"]["start_year"],
        end_year=decision["time_range"]["end_year"],
        reasoning=decision["reasoning"],
        target_function=module_config["function"],
        function_params={},  # Parameters will be generated in executor
        module_name=suggested_module,
        expected_outputs=module_config["outputs"],
        validation_rules=[
            "Check if dataframe exists in state",
            "Validate time range parameters",
            "Ensure required columns exist in dataset"
        ]
    )


def planner(state: State) -> State:
    """
    Generate execution plan based on analysis decision
    """
    current_iteration = state["iteration_count"]
    print(f"PROCESS - Analysis Planning - at iteration: {current_iteration} - START")
    
    # Check if analysis decision exists
    if "analysis_decision" not in state:
        raise ValueError("No analysis_decision found in state")
    
    decision = state["analysis_decision"]
    print(f"Processing decision for module: {decision['suggested_module']}")
    
    try:
        # Generate execution plan
        execution_plan = generate_execution_plan(decision)
        
        # Update state with execution plan
        new_state = state.copy()
        new_state["execution_plan"] = {
            "analysis_text": execution_plan.analysis_text,
            "analysis_type": execution_plan.analysis_type,
            "chart_type": execution_plan.chart_type,
            "start_year": execution_plan.start_year,
            "end_year": execution_plan.end_year,
            "reasoning": execution_plan.reasoning,
            "target_function": execution_plan.target_function,
            "function_params": execution_plan.function_params,
            "module_name": execution_plan.module_name,
            "expected_outputs": execution_plan.expected_outputs,
            "validation_rules": execution_plan.validation_rules
        }
        
        # Print execution plan details
        print(f"✅ Execution plan generated:")
        print(f"   Analysis text: {execution_plan.analysis_text}")
        print(f"   Analysis type: {execution_plan.analysis_type}")
        print(f"   Chart type: {execution_plan.chart_type}")
        print(f"   Time range: {execution_plan.start_year}-{execution_plan.end_year}")
        print(f"   Target function: {execution_plan.target_function}")
        print(f"   Module: {execution_plan.module_name}")
        print(f"   Expected outputs: {execution_plan.expected_outputs}")
        print(f"   Reasoning: {execution_plan.reasoning}")
        
        # Save state to memory
        shared_memory.save_state(new_state)
        print(f"state saved to memory for thread {shared_memory.thread_id}")
        
        print(f"PROCESS - Analysis Planning - at iteration: {current_iteration} - DONE")
        
        return new_state
        
    except Exception as e:
        print(f"❌ Error in analysis planning: {e}")
        raise


def test_analysis_planner():
    """
    Test function for analysis planner
    """
    # Create a mock state with analysis decision
    mock_state = {
        "iteration_count": 0,
        "analysis_decision": {
            "question_text": "What are the main research themes in IEEE VIS publications?",
            "analysis_text": "Analyze research themes using topic modeling",
            "suggested_module": "topic_analysis_module",
            "primary_attributes": ["Abstract", "Title", "AuthorKeywords"],
            "secondary_attributes": ["AuthorNames-Deduped"],
            "suggested_chart_type": "topic_clustering",
            "time_range": {"start_year": 1990, "end_year": 2024},
            "reasoning": "Content analysis for research themes"
        }
    }
    
    print("=== Testing Analysis Planner ===")
    print(f"Input decision: {mock_state['analysis_decision']['suggested_module']}")
    print("=" * 40)
    
    try:
        result_state = planner(mock_state)
        
        if "execution_plan" in result_state:
            plan = result_state["execution_plan"]
            print(f"\n✅ Execution Plan Generated:")
            print(f"Analysis Text: {plan['analysis_text']}")
            print(f"Analysis Type: {plan['analysis_type']}")
            print(f"Chart Type: {plan['chart_type']}")
            print(f"Time Range: {plan['start_year']}-{plan['end_year']}")
            print(f"Target Function: {plan['target_function']}")
            print(f"Module: {plan['module_name']}")
            print(f"Expected Outputs: {plan['expected_outputs']}")
            print(f"Reasoning: {plan['reasoning']}")
        else:
            print("❌ No execution_plan found in result_state")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    return result_state


if __name__ == "__main__":
    test_analysis_planner()
