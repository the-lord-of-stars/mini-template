from state import State
from helpers import get_dataset_info
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from helpers import get_llm
from memory import shared_memory
from typing import List, Optional
from schema import QuestionType, PrimaryAttribute, SecondaryAttribute, TransformationType, InsightType, VisualizationType, AnalysisParameters
from pydantic import Field

# class ResponseFormatter(BaseModel):
#     question: str

class AnalysisDecision(BaseModel):
    """Analysis decision with analysis module selection"""
    question_text: str = Field(..., description="A specific analytical question based on the user question")
    analysis_text: str = Field(..., description="A short description of the analysis to do based on the question")
    analysis_type: str = Field(..., description="The type of analysis to do based on the question")
    suggested_module: str = Field(..., description="Suggested analysis tool: node_analyse_authors, node_analyse_topics, or node_analysis_basics")
    primary_attributes: List[str] = Field(..., description="Primary data columns to analyze")
    secondary_attributes: List[str] = Field(..., description="Secondary columns to analyze")
    suggested_chart_type: str = Field(..., description="Suggested type of chart to create")
    start_year: int = Field(..., description="Start year for analysis")
    end_year: int = Field(..., description="End year for analysis")
    reasoning: str = Field(..., description="Reasoning for tool selection")


def question(state: State):
    """
    Generate a question based on the topic and selected dataset
    """
    current_iteration = state["iteration_count"]
    print(f"PROCESS - Question Generation - at iteration: {current_iteration} - START")

    selected_dataset_path = state["select_data_state"]["dataset_path"]
    dataset_info = get_dataset_info(selected_dataset_path)

    author_analysis = {
        "analysis_type": ["author_collaboration_networks"],
        "chart_type": ["network_graph"]
    }

    topic_analysis = {
        "analysis_type": ["top keywords", "topic temporal evolution", "keyword cooccurrence"],
        "chart_type": ["topic_evolution_plot","top_keywords_plot", "cooccurrence_matrix"]
    }

    basic_analysis = {
        "analysis_type": ["publication_trends", "citation_analysis", "download_patterns", "conference_comparisons", "time_series_analysis"],
        "chart_type": ["line_chart", "bar_chart", "scatter_plot", "box_plot", "histogram", "heatmap"]
    }

    questions, _ = shared_memory.export_questions_and_insights()

    # Get analysis history for module diversity guidance
    analysis_history_info = ""
    if "analysis_history" in state and state["analysis_history"]:
        history = state["analysis_history"]
        analysis_history_info = f"""
        ANALYSIS HISTORY (module usage count):
        - Basic analysis module: {history.get('Basic', 0)} times
        - Topic analysis module: {history.get('Topic', 0)} times  
        - Author analysis module: {history.get('Author', 0)} times
        
        MODULE DIVERSITY GUIDANCE:
        - Consider using modules that have been used less frequently
        - If one module has been used significantly more than others, consider exploring other modules
        - Aim for a balanced exploration across different analysis types
        """

    context = ""
    if state["iteration_count"] > 1:
        # Check if we have follow_up_decision information
        follow_up_info = ""
        if "follow_up_decision" in state:
            follow_up_decision = state["follow_up_decision"]
            if "llm_suggested_stop" in follow_up_decision and follow_up_decision["llm_suggested_stop"]:
                follow_up_info = f"""
        IMPORTANT: The previous analysis suggested stopping, but we're continuing to explore new directions.
        Suggested analysis direction: {follow_up_decision.get('analysis_direction', 'explore_new_perspective')}
        Reasoning: {follow_up_decision.get('reasoning', '')}
        
        Please generate a question that explores a DIFFERENT aspect or perspective to avoid repetition.
        """
            else:
                follow_up_info = f"""
        Suggested analysis direction: {follow_up_decision.get('analysis_direction', 'continue_same_direction')}
        Reasoning: {follow_up_decision.get('reasoning', '')}
        """
        
        context = f"""
        Here are the previous questions:
        {questions}

        Here are the insights generated in the last iteration:
        {state["insights"]}
        {follow_up_info}
        {analysis_history_info}

        Please pick a follow-up question based on the previous questions and insights.
        The follow-up question should be focused and operationalizable with not very complex code.
        For example, it can be a further analysis of specific insights that you find interesting.
    """

    sys_prompt = f"""
        You are a talented assistant (MBTI: ISTJ). Your task is to generate analysis questions and suggest the appropriate analysis modules

        The dataset has been selected based on the topic of {state["topic"]}.
        The query to select the dataset is to {state['select_data_state']['description']}.

        Here are the information of the selected dataset:
        {dataset_info}

        Please generate ONLY ONE atomic question based on the interests of the user, you and the available analysis modules, and select the most appropriate analysis module.

        IMPORTANT: You must choose between THREE analysis modules:

        1. author_analysis_module: Specialized in author-centric analysis
           - CAPABILITIES: Author networks, collaboration patterns, author influence ranking, co-authorship analysis
           - DATA FIELDS: AuthorNames, AuthorNames-Deduped, AuthorAffiliation
           - ANALYSIS TYPES: {", ".join(author_analysis["analysis_type"])}
           - AVAILABLE CHART TYPES: {", ".join(author_analysis["chart_type"])}
           - EXAMPLES: "Who are the most influential authors?", "How do researchers collaborate?", "Which authors have the strongest networks?"

        2. topic_analysis_module: Specialized in content and theme analysis
           - CAPABILITIES: Topic modeling, keyword analysis, theme evolution, text mining, semantic analysis
           - DATA FIELDS: Abstract, Title, AuthorKeywords, AuthorNames-Deduped
           - ANALYSIS TYPES: {", ".join(topic_analysis["analysis_type"])}
           - AVAILABLE CHART TYPES: {", ".join(topic_analysis["chart_type"])}
           - EXAMPLES: "What are the main research themes?", "How have topics evolved over time?", "Which keywords are most connected?"

        3. basic_analysis_module: Specialized in statistical and trend analysis on the publication metrics
           - CAPABILITIES: Publication trends, citation analysis, download patterns, conference comparisons, time series analysis
           - DATA FIELDS: Year, Conference, Downloads_Xplore, CitationCount_CrossRef, PaperType
           - ANALYSIS TYPES: {", ".join(basic_analysis["analysis_type"])}
           - AVAILABLE CHART TYPES: {", ".join(basic_analysis["chart_type"])}
           - EXAMPLES: "What are the publication trends?", "How do citations vary by conference?", "Do awarded papers get more downloads?"

        SELECTION RULES:
        1. Choose author_analysis_module when the question focuses on PEOPLE (authors, researchers, collaboration)
        2. Choose topic_analysis_module when the question focuses on CONTENT (topics, themes, keywords, text)
        3. Choose basic_analysis_module when the question focuses on METRICS (trends, statistics, comparisons, patterns)
        4. The question should be focused and operationalizable
        5. Suggest appropriate chart_type based on the analysis type
        6. Consider module diversity - prefer modules that have been used less frequently
        7. If one module has been heavily used, consider exploring other analysis perspectives

        {context}
    """

    human_prompt = f"Please generate the question."

    llm = get_llm(temperature=0.8, max_tokens=4096)

    for i in range(5):

        response = llm.with_structured_output(AnalysisDecision).invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        )
        valid_modules = ["author_analysis_module", "topic_analysis_module", "basic_analysis_module"]
        if response.suggested_module in valid_modules:
            break
        else:
            print(f"Invalid module: {response.suggested_module}, expected one of: {valid_modules}")
            print(f"Retrying... {i+1}/3")

    new_state = state.copy()

    new_state["analysis_decision"] = {
        "question_text": response.question_text,
        "analysis_text": response.analysis_text,
        "analysis_type": response.analysis_type,
        "suggested_module": response.suggested_module,
        "primary_attributes": response.primary_attributes,
        "secondary_attributes": response.secondary_attributes,
        "suggested_chart_type": response.suggested_chart_type,
        "time_range": {"start_year": response.start_year, "end_year": response.end_year},
        "reasoning": response.reasoning
    }

    new_state["question"] = {
        "question": response.question_text,
        "handled": False,
        "spec": response.suggested_module
    }

    print(f"Question generated: {response.question_text}")
    print(f"Analysis text: {response.analysis_text}")
    print(f"Suggested module: {response.suggested_module}")
    print(f"Suggested chart type: {response.suggested_chart_type}")
    print(f"Start year: {response.start_year}")
    print(f"End year: {response.end_year}")
    print(f"Reasoning: {response.reasoning}")

    # save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")
    print(f"PROCESS - Question Generation - at iteration: {current_iteration} - DONE")

    return new_state


def test_question_structured():
    """
    Test function for node_question_structured with a hypothetical user query topic
    """
    # Create a mock state with a hypothetical user query topic
    mock_state = {
        "topic": "evolution in sensemaking research",
        "select_data_state": {
            "dataset_path": "dataset.csv",
            "description": "The dataset contains information about the IEEE VIS publication record from 1990 to 2024."
        },
        "iteration_count": 0,
        "insights": []
    }
    
    print("=== Testing node_question_structured ===")
    print(f"User Topic: {mock_state['topic']}")
    print(f"Dataset: {mock_state['select_data_state']['description']}")
    print("=" * 50)

    
    try:
        # Call the question function
        result_state = question(mock_state)
        
        # Print the analysis decision
        if "analysis_decision" in result_state:
            decision = result_state["analysis_decision"]
                
        else:
            print("❌ No analysis_decision found in result_state")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    return result_state


if __name__ == "__main__":
    test_question_structured()


