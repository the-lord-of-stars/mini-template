from state import State
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from helpers import get_llm
from memory import shared_memory


class RoutingDecision(BaseModel):
    """Structured routing decision schema"""
    analysis_module: str = Field(..., description="The analysis module to route to: analyse_topics, analyse_authors, or node_facts")
    routing_reason: str = Field(..., description="Detailed explanation of why this module was chosen")


def route(state: State):
    """
    Route to the appropriate analysis node based on the analysis plan using LLM
    """
    
    # Get the analysis plan from state
    analysis_plan = state.get("analysis_plan", {})
    
    # Extract routing information
    question_type = analysis_plan.get("q_type", "")
    primary_attributes = analysis_plan.get("primary_attributes", [])
    secondary_attributes = analysis_plan.get("secondary_attributes", [])
    question_text = analysis_plan.get("question_text", "")
    analysis_focus = analysis_plan.get("analysis_focus", "")
    transformation = analysis_plan.get("transformation", "")
    expected_insights = analysis_plan.get("expected_insights", [])
    visualization_types = analysis_plan.get("visualization_types", [])
    reasoning = analysis_plan.get("reasoning", "")
    
    # Create context for LLM routing
    context = f"""
    Analysis Plan Information:
    - Question: {question_text}
    - Question Type: {question_type}
    - Primary Attributes: {primary_attributes}
    - Secondary Attributes: {secondary_attributes}
    - Analysis Focus: {analysis_focus}
    - Transformation: {transformation}
    - Expected Insights: {expected_insights}
    - Visualization Types: {visualization_types}
    - Reasoning: {reasoning}
    """
    
    sys_prompt = """
    You are an intelligent routing system that determines which analysis module should handle a given analysis plan.
    
    Available analysis modules:
    1. analyse_topics - Specialized for topic analysis, trend analysis, subject classification, theme evolution, and content categorization
    2. analyse_authors - Specialized for author analysis, collaboration networks, researcher profiles, citation analysis, and publication patterns
    3. node_facts - General analysis module for basic data exploration, statistical summaries, and other general analytical tasks on numerical data
    
    Routing rules:
    - Choose analyse_topics if the analysis focuses on topics, themes, subjects, categories, or content classification
    - Choose analyse_authors if the analysis focuses on authors, researchers, collaborations, networks, or publication patterns
    - Choose node_facts for general data exploration, statistical analysis, or when the focus is not clearly topics or authors
    
    Provide a structured routing decision with high confidence when the analysis type is clear, and lower confidence when it's ambiguous.
    """
    
    human_prompt = f"""
    Based on the following analysis plan, determine the most appropriate analysis module:
    
    {context}
    
    Please provide a routing decision.
    """
    
    llm = get_llm(temperature=0, max_tokens=1024)
    
    response = llm.with_structured_output(RoutingDecision).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )
    
    next_node = response.analysis_module
    routing_reason = response.routing_reason
    
    # Create new state with routing information
    new_state = state.copy()
    
    new_state["routing"] = {
        "analysis_module": next_node,
        "routing_reason": routing_reason,
    }
    
    # save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")
    
    # Print routing decision for debugging
    print(f"=== Routing Decision ===")
    print(f"Analysis Module: {next_node}")
    print(f"Reason: {routing_reason}")
    print("========================")
    
    return new_state


def test_route():
    """
    Test function for the routing logic
    """
    # Test case 1: Topic analysis
    test_state_1 = {
        "analysis_plan": {
            "question_text": "What are the most common topics in IEEE VIS publications from 1990 to 2024, and how have these topics evolved over time?",
            "q_type": "QuestionType.TOPIC_ANALYSIS",
            "primary_attributes": ["PrimaryAttribute.TOPICS"],
            "secondary_attributes": ["SecondaryAttribute.YEAR"],
            "transformation": "TransformationType.TIME_SERIES_ANALYSIS",
            "expected_insights": ["InsightType.TREND_ANALYSIS", "InsightType.STATISTICAL_SUMMARY"],
            "parameters": {
                "top_n": 10,
                "min_papers": None,
                "time_period": "1990-2024",
                "min_collaborations": None,
                "include_self_citations": None,
                "network_threshold": None,
                "clustering_method": None
            },
            "visualization_types": ["VisualizationType.LINE_CHART", "VisualizationType.BAR_CHART"],
            "analysis_focus": "evolution of topics in IEEE VIS publications",
            "reasoning": "Understanding the evolution of topics over time can provide insights into the trends and shifts in research focus within the IEEE VIS community, helping to identify emerging areas of interest and the historical context of current research directions."
        }
    }
    
    # Test case 2: Author analysis
    # test_state_2 = {
    #     "analysis_plan": {
    #         "q_type": "AUTHOR_ANALYSIS",
    #         "primary_attributes": ["AUTHORS"],
    #         "question_text": "Who are the most cited authors?",
    #         "analysis_focus": "author collaboration"
    #     }
    # }
    
    # # Test case 3: General analysis
    # test_state_3 = {
    #     "analysis_plan": {
    #         "q_type": "GENERAL_ANALYSIS",
    #         "primary_attributes": ["YEAR"],
    #         "question_text": "How has the field evolved?",
    #         "analysis_focus": "general trends"
    #     }
    # }
    
    print("Testing routing logic...")
    print("\nTest 1 - Topic Analysis:")
    result_1 = route(test_state_1)
    
    # print("\nTest 2 - Author Analysis:")
    # result_2 = route(test_state_2)
    
    # print("\nTest 3 - General Analysis:")
    # result_3 = route(test_state_3)


if __name__ == "__main__":
    test_route()