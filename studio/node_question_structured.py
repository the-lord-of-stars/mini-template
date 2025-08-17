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

class AnalysisPlan(BaseModel):
    """Structured analysis plan schema"""
    question_text: str = Field(..., description="The original user question")
    q_type: QuestionType = Field(..., description="Type of question/analysis")
    primary_attributes: List[PrimaryAttribute] = Field(..., description="Primary data column to analyze")
    secondary_attributes: List[SecondaryAttribute] = Field(..., description="Secondary columns to analyze")
    transformation: TransformationType = Field(..., description="Data transformation method")
    expected_insights: List[InsightType] = Field(..., description="Types of insights expected")
    parameters: AnalysisParameters = Field(default_factory=AnalysisParameters, description="Analysis parameters")
    visualization_types: List[VisualizationType] = Field(..., description="Types of visualizations to create")
    analysis_focus: Optional[str] = Field(default="general", description="Focus area of the analysis")
    reasoning: Optional[str] = Field(default="", description="Reasoning for this analysis plan")


def question(state: State):
    """
    Generate a question based on the topic and selected dataset
    """

    selected_dataset_path = state["select_data_state"]["dataset_path"]
    dataset_info = get_dataset_info(selected_dataset_path)

    questions, _ = shared_memory.export_questions_and_insights()

    context = ""
    if state["iteration_count"] > 1:
        context = f"""
        Here are the previous questions:
        {questions}

        Here are the insights generated in the last iteration:
        {state["insights"]}

        Please pick a follow-up question based on the previous questions and insights.
        The follow-up question should be focused and operationalizable with not very complex code.
        For example, it can be a further analysis of specific insights that you find interesting.
    """

    sys_prompt = f"""
        You are a helpful assistant that generate analysis questions to explore the dataset.

        In previous analysis, the dataset has been selected based on the topic of {state["topic"]}.
        The query to select the dataset is to {state['select_data_state']['description']}.

        Here are the information of the selected dataset:
        {dataset_info}

        Please generate one atomic question that is the most relevant to start explore the dataset.

        IMPORTANT: You must choose between only TWO types of analysis:

        1. TOPIC_ANALYSIS: Use this when the question are primarily using publication or research keywords
           - What topics are most common/popular
           - How topics relate to each other
           - Topic evolution over time
           - Examples: "What are the most common research topics?", "How have topics evolved?"

        2. GENERAL_ANALYSIS: Use this when the question focuses on:
           - Publication trends, citation trends, collaboration patterns
           - Statistical overviews, productivity patterns
           - Non-topic, non-author related analysis
           - Examples: "What are the publication trends?", "How do citations vary?"

        Rules:
        1. the question should be focused and relevant
        2. the question should be one task that is operationalizable
        3. Choose the appropriate analysis type based on the question content

        {context}
    """

    human_prompt = f"Please generate the question."

    llm = get_llm(temperature=0, max_tokens=4096)

    response = llm.with_structured_output(AnalysisPlan).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    new_state = state.copy()

    new_state["analysis_plan"] = {
        "question_text": response.question_text,
        "q_type": response.q_type,
        "primary_attributes": response.primary_attributes,
        "secondary_attributes": response.secondary_attributes,
        "transformation": response.transformation,
        "expected_insights": response.expected_insights,
        "parameters": response.parameters,
        "visualization_types": response.visualization_types,
        "analysis_focus": response.analysis_focus,
        "reasoning": response.reasoning
    }

    new_state["question"] = {
        "question": response.question_text,
        "handled": False,
        "spec": ""
    }

    # save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")

    return new_state


def test_analysis_plan():
    """
    Simple test function to print the analysis_plan from new_state and route to appropriate analysis
    """
    # Create a mock state for testing
    mock_state = {
        "topic": "major topics in sensemaking research",
        "select_data_state": {
            "dataset_path": "dataset.csv",
            "description": "The dataset contains information about the IEEE VIS publication record from 1990 to 2024."
        },
        "iteration_count": 0,
        "insights": []
    }
    
    # Call the question function
    result_state = question(mock_state)
    
    # Print the analysis_plan
    if "analysis_plan" in result_state:
        print("=== Analysis Plan ===")
        print(f"Question Text: {result_state['analysis_plan']['question_text']}")
        print(f"Question Type: {result_state['analysis_plan']['q_type']}")
        print(f"Primary Attributes: {result_state['analysis_plan']['primary_attributes']}")
        print(f"Secondary Attributes: {result_state['analysis_plan']['secondary_attributes']}")
        print(f"Transformation: {result_state['analysis_plan']['transformation']}")
        print(f"Expected Insights: {result_state['analysis_plan']['expected_insights']}")
        print(f"Parameters: {result_state['analysis_plan']['parameters']}")
        print(f"Visualization Types: {result_state['analysis_plan']['visualization_types']}")
        print(f"Analysis Focus: {result_state['analysis_plan']['analysis_focus']}")
        print(f"Reasoning: {result_state['analysis_plan']['reasoning']}")
        print("===================")
        
        # Route based on question type
        q_type = result_state['analysis_plan']['q_type']
        print(f"\n=== Routing Decision ===")
        print(f"Question type: {q_type}")
        
        # Simple routing logic based on question type
        if q_type == "topic_analysis":
            print("✅ Routing to node_analyse_topics")
            # Import and call the topic analysis function
            from node_analyse_topics import analyse_topics
            import pandas as pd
            
            # Prepare state for topic analysis
            analysis_state = result_state.copy()
            analysis_state["dataframe"] = pd.read_csv("dataset.csv")
            
            # Call topic analysis
            topic_result = analyse_topics(analysis_state)
            print("✅ Topic analysis completed!")
            
            # Update result_state with topic analysis results
            result_state.update(topic_result)
            
        elif q_type == "author_ranking":
            print("📋 Would route to author_ranking analysis (not implemented yet)")
            # TODO: Implement author_ranking analysis
            # from node_analyse_authors import analyse_authors
            # author_result = analyse_authors(result_state)
            # result_state.update(author_result)
            
        elif q_type == "collaboration_analysis":
            print("📋 Would route to collaboration_analysis (not implemented yet)")
            # TODO: Implement collaboration analysis
            
        elif q_type == "statistical_overview":
            print("📋 Would route to statistical_overview (not implemented yet)")
            # TODO: Implement statistical overview
            
        elif q_type == "network_structure":
            print("📋 Would route to network_structure (not implemented yet)")
            # TODO: Implement network structure analysis
            
        elif q_type == "trend_analysis":
            print("📋 Would route to trend_analysis (not implemented yet)")
            # TODO: Implement trend analysis
            
        elif q_type == "comparative_analysis":
            print("📋 Would route to comparative_analysis (not implemented yet)")
            # TODO: Implement comparative analysis
            
        else:
            print(f"❌ Unknown question type: {q_type}")
            print(f"Available types: topic_analysis, author_ranking, collaboration_analysis, statistical_overview, network_structure, trend_analysis, comparative_analysis")
            
    else:
        print("No analysis_plan found in result_state")
    
    return result_state


def test_workflow_node():
    """
    Test the workflow node function
    """
    # Create a mock state for testing
    mock_state = {
        "topic": "major topics in sensemaking research",
        "select_data_state": {
            "dataset_path": "dataset.csv",
            "description": "The dataset contains information about the IEEE VIS publication record from 1990 to 2024."
        },
        "iteration_count": 0,
        "insights": []
    }
    
    print("=== Testing Workflow Node ===")
    print("Input state keys:", list(mock_state.keys()))
    
    # Call the workflow node function
    result_state = question_structured(mock_state)
    
    print(f"\n=== Workflow Node Result ===")
    print("Output state keys:", list(result_state.keys()))
    
    if "analysis_plan" in result_state:
        print(f"✅ Analysis plan generated: {result_state['analysis_plan']['question_text']}")
    
    if "topic_analysis_result" in result_state:
        print("✅ Topic analysis results included")
    
    if "visualizations" in result_state:
        print("✅ Visualizations included")
    
    if "facts" in result_state:
        print("✅ Facts included")
    
    if "insights" in result_state:
        print("✅ Insights included")
    
    return result_state


if __name__ == "__main__":
    # test_analysis_plan()  # Original test
    test_workflow_node()    # New workflow node test
