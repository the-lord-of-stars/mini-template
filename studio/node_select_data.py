from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers import get_llm, get_dataset_info, query_by_sql
from state import State
from memory import shared_memory

class ResponseFormatter(BaseModel):
    description: str = Field(description="The description of the SQL query")
    sql_query: str = Field(description="The SQL query to select the data from the dataset")

def get_automated_visualization_domain_knowledge():
    """提供深入的自动化可视化领域知识"""
    
    return """
    AUTOMATED VISUALIZATION DOMAIN EXPERTISE:
    
    === FIELD DEFINITION ===
    Automated visualization is the research area focused on creating systems that automatically generate, recommend, or assist in creating data visualizations with minimal human intervention.
    
    === KEY RESEARCH AREAS ===
    
    1. VISUALIZATION RECOMMENDATION SYSTEMS:
    - Chart recommendation based on data characteristics
    - Visualization suggestion engines
    - Systems: Voyager, Draco, VizML, SeeDB
    - Keywords: "visualization recommendation", "chart recommendation", "vis recommendation"
    
    2. GRAMMAR-BASED APPROACHES:
    - Declarative visualization languages
    - Grammar of Graphics implementations
    - Tools: Vega-Lite, ggplot2, Grammar of Graphics
    - Keywords: "grammar of graphics", "declarative visualization", "vega-lite"
    
    3. NATURAL LANGUAGE INTERFACES:
    - Text-to-visualization systems
    - Conversational visualization interfaces
    - Systems: nl4dv, DataTone, Articulate
    - Keywords: "nl2vis", "natural language visualization", "text to visualization"
    
    4. MACHINE LEARNING APPROACHES:
    - Learning-based chart generation
    - AI-driven visualization design
    - Automated design optimization
    - Keywords: "ml-driven visualization", "learning-based visualization", "ai visualization"
    
    5. AUTOMATIC CHART GENERATION:
    - Data-to-visualization pipelines
    - Automated encoding selection
    - Layout and aesthetic optimization
    - Keywords: "automatic chart generation", "chart synthesis", "visualization generation"
    
    === IMPORTANT TOOLS & SYSTEMS ===
    - Vega-Lite: The most important declarative visualization grammar
    - Draco: Constraint-based visualization recommendation
    - Voyager: Faceted browsing for exploratory visualization
    - D3.js: While not automated itself, often used in automated systems
    - Tableau: Has some automation features
    - VizML: Machine learning approach to visualization
    
    === RESEARCH VENUES ===
    - CHI: Human-Computer Interaction conference
    - VIS/InfoVis: IEEE Visualization conferences
    - SIGMOD/VLDB: Database conferences (data-driven viz)
    - IUI: Intelligent User Interfaces
    
    === COMMON TERMINOLOGY PATTERNS ===
    Authors typically use:
    - "automated" vs "automatic" (both common)
    - "visualization" vs "visualisation" (US vs UK spelling)
    - "chart" vs "graph" vs "plot" (often interchangeable)
    - "recommendation" vs "suggestion" vs "selection"
    - "generation" vs "synthesis" vs "creation"
    
    === SEARCH STRATEGY FOR ACADEMIC PAPERS ===
    Most relevant papers will contain:
    1. Core automation terms: automated, automatic, intelligent, smart
    2. Core visualization terms: visualization, chart, graph, visual
    3. Method terms: recommendation, generation, synthesis, design
    4. Tool names: vega-lite, draco, voyager, grammar
    """
def select_data(state: State):
    """
    Generate SQL query to select the data (based on the topic)
    """
    print(f"PROCESSING - SELECT DATA - START")

    # Get path of the main program being executed

    new_state = state.copy()
    dataset_info = get_dataset_info("dataset.csv")
    # domain_knowledge = get_automated_visualization_domain_knowledge()

    sys_prompt = f"""
        You are a data scientist with deep domain knowledge in data visualization. Your task is to generates SQL queries to select the data from the dataset.
        The dataset is as follows:
        {dataset_info}

        Please generate a SQL query to select the data from the dataset to support the analysis of the topic given by the user.

        Rules:
        1. Always use 'FROM Papers' (not FROM dataset or any other table name)
        2. Use standard SQL syntax compatible with pandasql
        3. Make sure column names match exactly with the dataset headers
        4. Use double quotes for column names that contain special characters
        5. IMPORTANT: Always include all the columns in your SELECT statement. These columns are required for further analysis
        6. Add WHERE conditions to filter data based on the topic
        7. Use ORDER BY Year ASC to sort by year

        Please generate a SQL query that finds papers relevant to this topic by:
        1. Identifying the main concepts in the topic
        2. For each concept, include relevant synonyms and variations
        3. Use logical operators to ensure papers contain all key concepts
        4. Avoid overly broad terms that might match irrelevant content

        For topic analysis:
        - Use domain knowledge to generate the keywords
        - Break down compound topics into key concepts
        - Use AND logic to ensure papers contain ALL relevant concepts
        - Use OR logic within each concept to include synonyms and variations
        - Be precise with keyword matching to avoid false positives
        - Consider both exact phrases and related terms
        - Search primarily in Abstract, then Title, then AuthorKeywords to ensure comprehensive coverage

    """

    human_prompt = f"{state['topic']}"

    llm = get_llm(temperature=0, max_tokens=4096)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    print(f"---------------------------------sql query: {response.sql_query}")

    # test sql query
    dataset = query_by_sql(response.sql_query)
    print(f"---------------------------------dataset size: {dataset.shape}")
    thread_dir = shared_memory._get_thread_dir()
    dataset_path = f"{thread_dir}/dataset_selected.csv"
    dataset.to_csv(dataset_path, index=False)


    new_state["select_data_state"] = {
        "description": response.description,
        "sql_query": response.sql_query,
        "dataset_path": dataset_path
    }

    new_state["dataframe"] = dataset

    iteration_count = new_state["iteration_count"] if "iteration_count" in new_state else 0
    new_state["iteration_count"] = iteration_count + 1

    # Save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")

    print(f"PROCESSING - SELECT DATA - END")

    return new_state
