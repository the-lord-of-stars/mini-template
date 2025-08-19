from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json

from agents.star.helpers import get_llm, get_dataset_info, query_by_sql
from agents.star.state import State
from agents.star.memory import shared_memory


class ResponseFormatter(BaseModel):
    should_reselect_data: bool = Field(description="Whether to re-select data for the next analysis")
    should_continue_analysis: bool = Field(description="Whether to continue with more iterations based on analysis quality and value")
    analysis_direction: str = Field(description="Suggested direction for next analysis: 'continue_same_direction', 'explore_new_perspective', 'switch_to_different_module', or 'focus_on_emerging_topics'")
    reasoning: str = Field(description="Reasoning for the decision")


def follow_up_decision(state: State):
    """
    Decide on follow-up question and whether to re-select data based on insights
    """

    print(f"DEBUG: follow_up_decision received state keys: {list(state.keys())}")
    print(f"DEBUG: Does state have 'insights'? {'insights' in state}")
    print(f"DEBUG: Does state have 'topic_analysis_result'? {'topic_analysis_result' in state}")
    print(f"DEBUG: Does state have 'question'? {'question' in state}")
    if 'question' in state:
        print(f"DEBUG: question value: {state['question']}")
    if 'topic_analysis_result' in state:
        print(f"DEBUG: topic_analysis_result keys: {list(state['topic_analysis_result'].keys())}")
        print(f"DEBUG: topic_analysis_result has insights: {'insights' in state['topic_analysis_result']}")
    
    # Debug insights value
    if 'insights' in state:
        print(f"DEBUG: insights type: {type(state['insights'])}")
        print(f"DEBUG: insights value: {state['insights']}")

    # Get current iteration info
    current_iteration = state["iteration_count"]
    max_iterations = state["max_iterations"]

    iteration_history = state["iteration_history"]
    print(f"DEBUG: iteration_history type: {type(iteration_history)}")
    if iteration_history:
        print(f"DEBUG: last iteration type: {type(iteration_history[-1])}")
        print(f"DEBUG: last iteration content: {iteration_history[-1]}")
    current_insights = iteration_history[-1]["insights"]

    # Get dataset info
    dataset_info = get_dataset_info(state['select_data_state']['dataset_path'])

    # Get previous questions and insights from iteration history
    previous_questions = []
    previous_insights = []

    if state.get("iteration_history"):
        for iteration in state["iteration_history"]:
            if "question" in iteration and iteration["question"]:
                previous_questions.append(iteration["question"]["question"])
            if "insights" in iteration and iteration["insights"]:
                # Handle both list and dict formats for insights
                insights = iteration["insights"]
                if isinstance(insights, list):
                    previous_insights.extend(insights)
                elif isinstance(insights, dict):
                    # Convert dict to list format
                    for key, value in insights.items():
                        previous_insights.append(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    previous_insights.append(str(insights))

    print("DEBUG: About to create questions_text")
    questions_text = "\n".join([f"- {q}" for q in previous_questions])
    print("DEBUG: questions_text created successfully")
    
    print("DEBUG: About to create insights_text")
    print(f"DEBUG: previous_insights type: {type(previous_insights)}")
    print(f"DEBUG: previous_insights length: {len(previous_insights)}")
    for i, insight in enumerate(previous_insights):
        print(f"DEBUG: previous_insights[{i}] type: {type(insight)}, value: {insight}")
    insights_text = "\n".join([f"- {i}" for i in previous_insights])
    print("DEBUG: insights_text created successfully")
    
    # Handle case where insights might be None
    print("DEBUG: About to get current_insights from iteration_history")
    current_insights = iteration_history[-1]["insights"]
    print(f"DEBUG: current_insights type: {type(current_insights)}")
    print(f"DEBUG: current_insights value: {current_insights}")
    if current_insights is None:
        current_insights = []
    print("DEBUG: current_insights processing completed")
    print("DEBUG: About to create sys_prompt")
    print(f"DEBUG: dataset_info type: {type(dataset_info)}")
    print(f"DEBUG: state['topic'] type: {type(state['topic'])}")
    print(f"DEBUG: current_iteration type: {type(current_iteration)}")
    print(f"DEBUG: max_iterations type: {type(max_iterations)}")
    print(f"DEBUG: questions_text type: {type(questions_text)}")
    print(f"DEBUG: insights_text type: {type(insights_text)}")
    print(f"DEBUG: current_insights type: {type(current_insights)}")

    sys_prompt = f"""
        You are a data analyst who decides on follow-up questions and data selection strategy based on insights.

        The dataset is as follows:
        {dataset_info}

        Current topic: {state['topic']}
        Current iteration: {current_iteration}/{max_iterations}

        Previous questions asked:
        {questions_text}

        Previous insights generated:
        {insights_text}

        Latest insights from current analysis:
        {current_insights}

        Your task is to:
        1. Decide whether the current dataset is sufficient for the next analysis or if new data selection is needed
        2. Decide whether to continue with more iterations based on analysis quality and value
        3. Suggest the direction for the next analysis (even if you think we should stop, suggest a direction in case we continue)

        Rules for data selection decision:
        - If the next analysis requires different data than what's currently selected, choose to re-select
        - If the current dataset can support the next analysis, don't re-select
        - Consider whether the current SQL query and dataset are sufficient for future questions

        Rules for continuing analysis decision:
        - Continue if the current analysis provided valuable insights and there are clear follow-up questions
        - Continue if the analysis revealed new patterns or trends that need deeper investigation
        - Continue if the current dataset has more unexplored potential
        - STOP if the insights are repetitive or not adding significant value
        - STOP if the analysis has reached a natural conclusion
        - STOP if the follow-up questions would be too similar to previous ones
        - STOP if the current iteration limit is reached ({current_iteration}/{max_iterations})
        - Consider the quality and novelty of insights when deciding to continue

        Rules for analysis direction suggestion:
        - 'continue_same_direction': If the current analysis path is fruitful and needs deeper exploration
        - 'explore_new_perspective': If we should look at the same topic from a different angle
        - 'switch_to_different_module': If we should use a different analysis module (author/topic/basic)
        - 'focus_on_emerging_topics': If we should focus on the emerging topics identified

        IMPORTANT: You should make an intelligent decision about whether to continue, considering both the quality of insights and the iteration limit.
    """
    print("DEBUG: sys_prompt created successfully")

    human_prompt = f"""
    Based on the insights from the current analysis, please decide on:
    1. Whether to re-select data for the next analysis
    2. Whether to continue with more iterations (considering analysis quality and iteration limit)
    3. What direction the next analysis should take (even if you think we should stop, suggest a direction)
    
    The current analysis question was: {state['question']['question']}.
    """

    print("DEBUG: About to create LLM")
    llm = get_llm(temperature=0.8, max_tokens=8192)
    print("DEBUG: LLM created successfully")

    print("DEBUG: About to call LLM with structured output")
    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )
    print("DEBUG: LLM call completed successfully")
    print("DEBUG: About to access response attributes")
    print(f"DEBUG: response type: {type(response)}")
    print(f"DEBUG: response attributes: {dir(response)}")
    
    new_state = state.copy()

    # Add decision info to state
    print("DEBUG: About to access response.should_reselect_data")
    should_reselect_data = response.should_reselect_data
    print("DEBUG: About to access response.reasoning")
    reasoning = response.reasoning
    print("DEBUG: About to access response.should_continue_analysis")
    should_continue_analysis = response.should_continue_analysis
    print("DEBUG: About to access response.analysis_direction")
    analysis_direction = response.analysis_direction
    
    new_state["follow_up_decision"] = {
        "should_reselect_data": should_reselect_data,
        "reasoning": reasoning
    }

    # Use LLM's intelligent decision about whether to continue
    llm_should_continue = should_continue_analysis
    
    # NEW LOGIC: Always continue if we haven't reached max iterations
    # But pass the LLM's decision and suggested direction to the question node
    should_continue = current_iteration < max_iterations
    
    print(f"LLM decision to continue: {llm_should_continue}")
    print(f"Iteration limit check: {current_iteration} < {max_iterations}")
    print(f"Final decision to continue: {should_continue}")
    print(f"Analysis direction suggested: {analysis_direction}")

    if should_continue:
        # Increment iteration count after completing this iteration
        new_state["should_continue"] = True
        new_state["iteration_count"] = current_iteration + 1

        # Pass LLM's decision and direction to the question node
        new_state["follow_up_decision"] = {
            "should_reselect_data": should_reselect_data,
            "llm_suggested_stop": not llm_should_continue,  # True if LLM wanted to stop
            "analysis_direction": analysis_direction,
            "reasoning": reasoning
        }
        print(f"state saved to memory for thread {shared_memory.thread_id}")
        print(f"Should re-select data: {should_reselect_data}")
        print(f"Should continue analysis: {should_continue_analysis}")
        print(f"Analysis direction: {analysis_direction}")
        print(f"Reasoning: {reasoning}")
        print(f"Iteration {new_state['iteration_count']}/{max_iterations}, continuing: {new_state['should_continue']}")
        
        if llm_should_continue:
            print("🔄 Continuing to next iteration based on intelligent decision")
        else:
            print("🔄 Continuing to next iteration (max not reached) - will try new direction")
    else:
        # Stop iterations - we've reached the limit
        new_state["should_continue"] = False
        print("🛑 Stopping: Reached maximum iteration limit")

    
    # 统一保存状态
    shared_memory.save_state(new_state)

    return new_state
