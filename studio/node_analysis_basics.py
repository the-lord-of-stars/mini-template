#!/usr/bin/env python3
"""
Basic analysis tools for data exploration and visualization.
These are general-purpose statistical analysis tools that can be used for various data exploration tasks.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from state import State, Visualization
from memory import shared_memory
from helpers import update_state
from schema_new import BaseAnalysisParameters
import json
import os

class BasicAnalysisParameters(BaseAnalysisParameters):
    """Basic analysis specific parameters"""
    chart_type: str = Field(..., description="Type of chart: line_trend, scatter_corr, bar_group, box_by_category, histogram, heatmap_xy")
    target_columns: List[str] = Field(..., description="Target columns for analysis")
    time_column: Optional[str] = Field(default=None, description="Time column for temporal analysis")
    top_n: Optional[int] = Field(default=None, description="Number of top items to show (not needed for line_trend)")
    min_frequency: int = Field(default=1, description="Minimum frequency for inclusion")
    time_range: Optional[Dict[str, int]] = Field(default=None, description="Time range settings")

def analyse_basics(state: State, analysis_params: BasicAnalysisParameters = None):
    """
    Basic analysis using standardized parameters
    """
    #-------Data filtering-------
    # get the filters
    file_path = state['select_data_state']['dataset_path']
    # response = llm_filter(task, file_path)
    response = llm_filter_validation(task, file_path, max_iterations=3)
    filters = response.filters
    print('filters: ', filters)
    filtered_df = df.query(filters['query'])

    #-------Analysis parameters-------
    if analysis_params is None:
        analysis_plan = state.get("analysis_plan", {})
        analysis_params = BasicAnalysisParameters(
            analysis_type="basic_analysis",
            question_text=analysis_plan.get("question_text", ""),
            primary_attributes=analysis_plan.get("primary_attributes", []),
            secondary_attributes=analysis_plan.get("secondary_attributes", []),
            chart_type=analysis_plan.get("parameters", {}).get("chart_type", "line_trend"),
            target_columns=analysis_plan.get("parameters", {}).get("target_columns", []),
            time_column=analysis_plan.get("parameters", {}).get("time_column", None),
            top_n=analysis_plan.get("parameters", {}).get("top_n", 10),
            min_frequency=analysis_plan.get("parameters", {}).get("min_frequency", 1),
            time_range=analysis_plan.get("parameters", {}).get("time_range", None)
        )
    
    print(f"DEBUG: analysis_params: {analysis_params}")

    #-------Data Aggregation & verification-------
    #...

    #-------Execute analysis: facts, insights, visualizations-------
    result = execute_basic_analysis_llm(state, analysis_params)
    updated_state = update_state(state, result)
    return updated_state

def execute_basic_analysis(state: State, analysis_params: BasicAnalysisParameters):
    """
    Execute basic analysis based on chart type
    """
    print("=== ENTERING execute_basic_analysis FUNCTION ===")
    current_iteration = state["iteration_count"] if "iteration_count" in state else 0
    print(f"Current iteration: {current_iteration}")
    
    df = state["dataframe"] if "dataframe" in state else None
    if df is None:
        return {"error": "No dataframe found in state"}
    
    chart_type = analysis_params.chart_type
    target_columns = analysis_params.target_columns
    time_column = analysis_params.time_column
    
    print(f"Executing {chart_type} analysis with columns: {target_columns}")
    
    try:
        return execute_line_trend_llm(df, analysis_params, current_iteration)
        # if chart_type == "line_trend":
        #     return execute_line_trend(df, analysis_params, current_iteration)
        # elif chart_type == "scatter_corr":
        #     return execute_scatter_corr(df, analysis_params, current_iteration)
        # elif chart_type == "bar_group":
        #     return execute_bar_group(df, analysis_params, current_iteration)
        # elif chart_type == "box_by_category":
        #     return execute_box_by_category(df, analysis_params, current_iteration)
        # elif chart_type == "histogram":
        #     return execute_histogram(df, analysis_params, current_iteration)
        # elif chart_type == "heatmap_xy":
        #     return execute_heatmap_xy(df, analysis_params, current_iteration)
        # else:
        #     return {"error": f"Unsupported chart type: {chart_type}"}
    except Exception as e:
        print(f"Error in basic analysis: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

def execute_line_trend(df: pd.DataFrame, params: BasicAnalysisParameters, iteration: int) -> Dict[str, Any]:
    """Execute line trend analysis"""
    print("=== Executing Line Trend Analysis ===")
    
    time_col = params.time_column
    target_cols = [col for col in params.target_columns if col != time_col and col in df.columns]
    
    if not time_col or time_col not in df.columns:
        return {"error": f"Time column {time_col} not found in dataframe"}
    
    # Filter by time range if specified
    dff = df.copy()
    if params.time_range:
        start_year = params.time_range.get("start_year")
        end_year = params.time_range.get("end_year")
        if start_year and end_year:
            dff = dff[(dff[time_col] >= start_year) & (dff[time_col] <= end_year)]
    
    # Create line chart
    if not target_cols:
        # Count by time
        agg = dff.groupby(time_col, as_index=False).size().rename(columns={"size": "count"})
        fig = px.line(agg, x=time_col, y="count", markers=True, 
                     title=f"Trend Analysis: {params.question_text}")
    else:
        # Mean by time for numeric columns
        traces = []
        for col in target_cols:
            if col in dff.columns and pd.api.types.is_numeric_dtype(dff[col]):
                agg = dff.groupby(time_col, as_index=False)[col].mean()
                traces.append(go.Scatter(x=agg[time_col], y=agg[col], 
                                       mode="lines+markers", name=f"mean({col})"))
        fig = go.Figure(traces)
        fig.update_layout(title=f"Trend Analysis: {params.question_text}", 
                         xaxis_title=time_col, yaxis_title="mean(value)")
    
    return create_visualization_result(fig, "line_trend", params.question_text, iteration)

def execute_scatter_corr(df: pd.DataFrame, params: BasicAnalysisParameters, iteration: int) -> Dict[str, Any]:
    """Execute scatter correlation analysis"""
    print("=== Executing Scatter Correlation Analysis ===")
    
    target_cols = [col for col in params.target_columns if col in df.columns]
    
    if len(target_cols) < 2:
        return {"error": "Need at least 2 numeric columns for correlation analysis"}
    
    # Select first two numeric columns
    x_col, y_col = target_cols[0], target_cols[1]
    
    if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
        return {"error": "Both columns must be numeric for correlation analysis"}
    
    # Create scatter plot
    fig = px.scatter(df, x=x_col, y=y_col, 
                    title=f"Correlation Analysis: {params.question_text}",
                    labels={x_col: x_col, y_col: y_col})
    
    # Add trend line
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col].rolling(window=10).mean(),
                            mode='lines', name='Trend', line=dict(color='red')))
    
    return create_visualization_result(fig, "scatter_corr", params.question_text, iteration)

def execute_bar_group(df: pd.DataFrame, params: BasicAnalysisParameters, iteration: int) -> Dict[str, Any]:
    """Execute grouped bar chart analysis"""
    print("=== Executing Grouped Bar Chart Analysis ===")
    
    time_col = params.time_column
    target_cols = [col for col in params.target_columns if col != time_col and col in df.columns]
    
    if not time_col or time_col not in df.columns:
        return {"error": f"Time column {time_col} not found in dataframe"}
    
    if not target_cols:
        return {"error": "No categorical columns specified for grouping"}
    
    # Filter by time range if specified
    dff = df.copy()
    if params.time_range:
        start_year = params.time_range.get("start_year")
        end_year = params.time_range.get("end_year")
        if start_year and end_year:
            dff = dff[(dff[time_col] >= start_year) & (dff[time_col] <= end_year)]
    
    # Create grouped bar chart
    cat_col = target_cols[0]
    if cat_col not in dff.columns:
        return {"error": f"Categorical column {cat_col} not found in dataframe"}
    
    # Count by time and category
    agg = dff.groupby([time_col, cat_col], as_index=False).size().rename(columns={"size": "count"})
    
    fig = px.bar(agg, x=time_col, y="count", color=cat_col,
                title=f"Grouped Bar Analysis: {params.question_text}",
                barmode='group')
    
    return create_visualization_result(fig, "bar_group", params.question_text, iteration)

def execute_box_by_category(df: pd.DataFrame, params: BasicAnalysisParameters, iteration: int) -> Dict[str, Any]:
    """Execute box plot by category analysis"""
    print("=== Executing Box Plot by Category Analysis ===")
    
    target_cols = [col for col in params.target_columns if col in df.columns]
    
    if len(target_cols) < 2:
        return {"error": "Need at least 1 numeric column and 1 categorical column"}
    
    # Find numeric and categorical columns
    numeric_cols = [col for col in target_cols if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols = [col for col in target_cols if not pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols or not cat_cols:
        return {"error": "Need both numeric and categorical columns for box plot"}
    
    numeric_col, cat_col = numeric_cols[0], cat_cols[0]
    
    # Create box plot
    fig = px.box(df, x=cat_col, y=numeric_col,
                title=f"Box Plot Analysis: {params.question_text}")
    
    return create_visualization_result(fig, "box_by_category", params.question_text, iteration)

def execute_histogram(df: pd.DataFrame, params: BasicAnalysisParameters, iteration: int) -> Dict[str, Any]:
    """Execute histogram analysis"""
    print("=== Executing Histogram Analysis ===")
    
    target_cols = [col for col in params.target_columns if col in df.columns]
    
    if not target_cols:
        return {"error": "No target columns specified for histogram"}
    
    numeric_cols = [col for col in target_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return {"error": "Need numeric column for histogram"}
    
    numeric_col = numeric_cols[0]
    
    # Create histogram
    nbins = params.top_n if params.top_n is not None else 20
    fig = px.histogram(df, x=numeric_col, nbins=nbins,
                      title=f"Distribution Analysis: {params.question_text}")
    
    return create_visualization_result(fig, "histogram", params.question_text, iteration)

def execute_heatmap_xy(df: pd.DataFrame, params: BasicAnalysisParameters, iteration: int) -> Dict[str, Any]:
    """Execute heatmap analysis"""
    print("=== Executing Heatmap Analysis ===")
    
    target_cols = [col for col in params.target_columns if col in df.columns]
    
    if len(target_cols) < 2:
        return {"error": "Need at least 2 columns for heatmap"}
    
    # Use first two columns
    x_col, y_col = target_cols[0], target_cols[1]
    
    # Create cross-tabulation
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        # For numeric columns, create bins
        bins_count = params.top_n if params.top_n is not None else 10
        x_bins = pd.cut(df[x_col], bins=min(10, bins_count))
        y_bins = pd.cut(df[y_col], bins=min(10, bins_count))
        pivot_table = pd.crosstab(x_bins, y_bins)
    else:
        # For categorical columns, direct cross-tabulation
        pivot_table = pd.crosstab(df[x_col], df[y_col])
    
    # Create heatmap
    fig = px.imshow(pivot_table, 
                    title=f"Heatmap Analysis: {params.question_text}",
                    labels=dict(x=x_col, y=y_col, color="Count"))
    
    return create_visualization_result(fig, "heatmap_xy", params.question_text, iteration)

def create_visualization_result(fig, chart_type: str, question_text: str, iteration: int) -> Dict[str, Any]:
    """Create standardized visualization result"""
    try:
        # Save figure
        thread_dir = shared_memory._get_thread_dir()
        fig_path = f'{thread_dir}/basic_analysis_{chart_type}_iteration_{iteration}.html'
        
        # Save as HTML
        fig.write_html(fig_path)
        
        # Generate HTML snippet
        figure_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Create visualization object
        visualization = Visualization(
            insight=f"Basic {chart_type} analysis",
            chart_type=chart_type,
            altair_code="",  # Using plotly here
            description=f"Basic {chart_type} analysis for: {question_text}",
            is_appropriate=True,
            image_path=fig_path,
            success=True,
            figure_object=figure_html,
            code=""
        )
        
        # Generate basic facts
        facts = {
            "code": "",
            "stdout": f"Successfully created {chart_type} visualization",
            "stderr": "",
            "exit_code": 0
        }
        
        return {
            "visualizations": {"visualizations": [visualization]},
            "facts": facts,
            "insights": [f"Generated {chart_type} analysis visualization"]
        }
        
    except Exception as e:
        print(f"Error creating visualization result: {e}")
        return {
            "error": f"Failed to create visualization: {str(e)}",
            "facts": {"code": "", "stdout": "", "stderr": str(e), "exit_code": 1}
        }


if __name__ == "__main__":
    # Test main function for basic analysis tools
    state = State()
    state["dataframe"] = pd.read_csv("dataset.csv")
    
    # Create a mock analysis plan for testing
    state["analysis_plan"] = {
        "primary_attributes": ["AuthorKeywords"],
        "secondary_attributes": ["Year", "Conference"],
        "parameters": {
            "top_n": 10,
            "time_range": "1990-2024",
            "conference_filter": True
        },
        # "question_text": "What are correlations between frequent topics?"
        "question_text": "analyse the temporal evolution of the most frequent topics?"
    }
    
    state["iteration_count"] = 0
    
    print("=== Testing Basic Analysis Tools ===")
    print(f"Dataset shape: {state['dataframe'].shape}")
    print(f"Dataset columns: {list(state['dataframe'].columns)}")
    
    # Test 1: Line Trend Analysis
    print("\n--- Test 1: Line Trend Analysis ---")
    line_trend_params = BasicAnalysisParameters(
        analysis_type="basic_analysis",
        question_text="How has the number of publications changed over time?",
        primary_attributes=["Year"],
        secondary_attributes=["Conference"],
        chart_type="line_trend",
        target_columns=["Year"],
        time_column="Year",
        time_range={"start_year": 2010, "end_year": 2024}
    )
    
    try:
        result1 = analyse_basics(state, line_trend_params)
        print("✅ Line trend analysis completed")
        if "visualizations" in result1:
            print(f"Generated visualization: {result1['visualizations']['visualizations'][0].chart_type}")
    except Exception as e:
        print(f"❌ Line trend analysis failed: {e}")
    
    # Test 2: Scatter Correlation Analysis
    print("\n--- Test 2: Scatter Correlation Analysis ---")
    df = pd.read_csv("dataset.csv")
    #calculate the correlation between download count and citation count differ from conference
    df_new = df[["Year", "Downloads_Xplore", "CitationCount_CrossRef", "Conference"]]
    df_new['CitationCount_CrossRef'] = df_new['CitationCount_CrossRef'].fillna(0)
    df_new = df_new.groupby("Year").sum()
    df_new = df_new.reset_index()
    df_new = df_new.sort_values(by="Year", ascending=True)
    df_new = df_new.reset_index(drop=True)
    df_new = df_new.sort_values(by="Year", ascending=True)

    scatter_params = BasicAnalysisParameters(
        analysis_type="basic_analysis",
        question_text=f"What is the relationship between download count and citation count differ from conference?",
        primary_attributes=["Year", "Conference"],
        secondary_attributes=["Downloads_Xplore", "CitationCount_CrossRef"],
        chart_type="scatter_corr",
        target_columns=["Year", "Conference", "Downloads_Xplore", "CitationCount_CrossRef"]
    )
    
    state["dataframe"] = df_new
    try:
        result2 = analyse_basics(state, scatter_params)
        print("✅ Scatter correlation analysis completed")
        if "visualizations" in result2:
            print(f"Generated visualization: {result2['visualizations']['visualizations'][0].chart_type}")
    except Exception as e:
        print(f"❌ Scatter correlation analysis failed: {e}")
    
    # Test 3: Bar Group Analysis
    print("\n--- Test 3: Bar Group Analysis ---")

    # which conference has the most papers published over time?
    df_new = df[["Year", "Conference", "Downloads_Xplore"]]
    df_new = df_new.groupby(["Year", "Conference"]).size().reset_index(name="count")
    df_new = df_new.sort_values(by=["Year", "count"], ascending=[True, False])
    df_new = df_new.reset_index(drop=True)
    df_new = df_new.sort_values(by=["Year", "count"], ascending=[True, False])

    bar_group_params = BasicAnalysisParameters(
        analysis_type="basic_analysis",
        question_text="Which conference has the most papers published over time?",
        primary_attributes=["Year", "Conference"],
        secondary_attributes=["count"],
        chart_type="bar_group",
        target_columns=["Year", "Conference", "count"],
        time_column="Year",
        top_n=10,
        time_range={"start_year": 2015, "end_year": 2024}
    )
    state["dataframe"] = df_new
    
    try:
        result3 = analyse_basics(state, bar_group_params)
        print("✅ Bar group analysis completed")
        if "visualizations" in result3:
            print(f"Generated visualization: {result3['visualizations']['visualizations'][0].chart_type}")
    except Exception as e:
        print(f"❌ Bar group analysis failed: {e}")
    
    # Test 4: Box Plot Analysis
    print("\n--- Test 4: Box Plot Analysis ---")

    # which paper type has the most citations?
    df_new = df[["Year", "Conference", "CitationCount_CrossRef", "PaperType"]]
    df_new = df_new.groupby(["Year", "Conference", "PaperType"]).sum().reset_index()
    df_new = df_new.sort_values(by=["Year", "CitationCount_CrossRef"], ascending=[True, False])
    df_new = df_new.reset_index(drop=True)
    df_new = df_new.sort_values(by=["Year", "CitationCount_CrossRef"], ascending=[True, False])
    
    state["dataframe"] = df_new

    box_params = BasicAnalysisParameters(
        analysis_type="basic_analysis",
        question_text="Which paper type has the most citations?",
        primary_attributes=["Year", "Conference", "PaperType"],
        secondary_attributes=["CitationCount_CrossRef"],
        chart_type="box_by_category",
        target_columns=["Year", "Conference", "PaperType", "CitationCount_CrossRef"]
    )

    try:
        result4 = analyse_basics(state, box_params)
        print("✅ Box plot analysis completed")
        if "visualizations" in result4:
            print(f"Generated visualization: {result4['visualizations']['visualizations'][0].chart_type}")
    except Exception as e:
        print(f"❌ Box plot analysis failed: {e}")
    
    # Test 5: Histogram Analysis
    print("\n--- Test 5: Histogram Analysis ---")

    # paper length distribution
    df_new = df[["Year", "FirstPage", "LastPage"]]
    df_new['PaperLength'] = df_new['LastPage'] - df_new['FirstPage'] + 1
    df_new = df_new.groupby("Year").mean().reset_index()
    df_new = df_new.sort_values(by="Year", ascending=True)
    df_new = df_new.reset_index(drop=True)
    df_new = df_new.sort_values(by="Year", ascending=True)
    state["dataframe"] = df_new

    histogram_params = BasicAnalysisParameters(
        analysis_type="basic_analysis",
        question_text="What is the distribution of paper length?",
        primary_attributes=["Year"],
        secondary_attributes=["PaperLength"],
        chart_type="histogram",
        target_columns=["Year", "PaperLength"]
    )

    
    try:
        result5 = analyse_basics(state, histogram_params)
        print("✅ Histogram analysis completed")
        if "visualizations" in result5:
            print(f"Generated visualization: {result5['visualizations']['visualizations'][0].chart_type}")
    except Exception as e:
        print(f"❌ Histogram analysis failed: {e}")
    
    # Test 6: Heatmap Analysis
    print("\n--- Test 6: Heatmap Analysis ---")
    # each year, the number of papers published differ from conference
    df_new = df[["Year", "Conference", "Downloads_Xplore"]]
    df_new = df_new.groupby(["Year", "Conference"]).size().reset_index(name="count")
    df_new = df_new.sort_values(by=["Year", "count"], ascending=[True, False])
    df_new = df_new.reset_index(drop=True)
    df_new = df_new.sort_values(by=["Year", "count"], ascending=[True, False])
    state["dataframe"] = df_new
    
    heatmap_params = BasicAnalysisParameters(
        analysis_type="basic_analysis",
        question_text="What is the relationship between year and conference?",
        primary_attributes=["Year", "Conference"],
        secondary_attributes=["count"],
        chart_type="heatmap_xy",
        target_columns=["Year", "Conference", "count"]
    )

    try:
        result6 = analyse_basics(state, heatmap_params)
        print("✅ Heatmap analysis completed")
        if "visualizations" in result6:
            print(f"Generated visualization: {result6['visualizations']['visualizations'][0].chart_type}")
    except Exception as e:
        print(f"❌ Heatmap analysis failed: {e}")
    
    print("\n=== Test Summary ===")
    print("All basic analysis tools have been tested!")
    print("Check the output directory for generated visualizations.")


# Import sandbox function for safe code execution
from sandbox import run_in_sandbox_with_venv

def execute_basic_analysis_llm(df: pd.DataFrame, analysis_params: BasicAnalysisParameters, iteration: int) -> Dict[str, Any]:
    """
    Execute line trend analysis using LLM for intelligent data processing and visualization
    """
    print("=== Executing LLM-Powered Analysis ===")
    
    # Define question variable to match other analysis functions
    question = analysis_params.question_text
    success = False
    
    try:
        # Prepare context for LLM
        data_sample = df.head(5).to_dict('records')  # Sample data for LLM
        column_info = {
            col: str(df[col].dtype) for col in df.columns
        }
        
        # Get LLM response using structured output
        from helpers import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage
        from pydantic import BaseModel, Field
        
        class ResponseFormatter(BaseModel):
            code: str = Field(..., description="Complete Python Plotly code as a string")
            insights: str = Field(..., description="Descriptive insights about the trends as a JSON string")
        
        llm = get_llm(temperature=0.1, max_tokens=4096)
        
        system_message = SystemMessage(content=f"""
        Please generate Python Plotly code to visualize insights from the dataset, output should be graphs and narrative.
        
        IMPORTANT: The data is already loaded in a variable called 'df'. Do NOT use pd.read_csv().
        
        The dataset has these column names: {list(df.columns)}
        Analysis question: {analysis_params.question_text}
        Time range: {analysis_params.time_range}
        
        Requirements:
            1. Generate Python code using Plotly to create the visualization
            2. Use the existing 'df' variable (do not load data from file)
            3. Create appropriate chart type (line, bar, scatter, or box plot.) based on the question
            4. Include proper titles, labels, and styling
            5. Provide descriptive narrative/insights about the data trends
            6. Focus on answering the analysis question
            7. Use Plotly features only
            8. Create only ONE chart (do not use multiple fig.show() calls)
        """)
        
        human_message = HumanMessage(content="Generate a response.")
        
        try:
            structured_llm = llm.with_structured_output(ResponseFormatter)
            response = structured_llm.invoke([system_message, human_message])
            
            # Parse the insights
            import json
            generated_code = response.code
            insights = json.loads(response.insights)
            
        except Exception as e:
            print(f"!!!!!!!Structured output failed, using fallback: {e}")
            # Fallback to basic Python Plotly code
            generated_code = f"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Filter data by time range
start_year = {analysis_params.time_range.get('start_year', 1990)}
end_year = {analysis_params.time_range.get('end_year', 2024)}
filtered_df = df[(df['{analysis_params.time_column or "Year"}'] >= start_year) & (df['{analysis_params.time_column or "Year"}'] <= end_year)]

# Aggregate data by year
agg_data = filtered_df.groupby('{analysis_params.time_column or "Year"}').agg({{
    'Downloads_Xplore': 'mean',
    'CitationCount_CrossRef': 'mean'
}}).reset_index()

# Create subplots
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Downloads Trend', 'Citations Trend'),
                    vertical_spacing=0.1)

# Add downloads line
fig.add_trace(
    go.Scatter(x=agg_data['{analysis_params.time_column or "Year"}'], 
               y=agg_data['Downloads_Xplore'],
               mode='lines+markers',
               name='Downloads',
               line=dict(color='blue')),
    row=1, col=1
)

# Add citations line
fig.add_trace(
    go.Scatter(x=agg_data['{analysis_params.time_column or "Year"}'], 
               y=agg_data['CitationCount_CrossRef'],
               mode='lines+markers',
               name='Citations',
               line=dict(color='orange')),
    row=2, col=1
)

# Update layout
fig.update_layout(
    title='{analysis_params.question_text}',
    height=600,
    showlegend=True
)

fig.show()
"""
            insights = ["Basic trend analysis completed"]
        
        # Execute Python Plotly code to generate visualization
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import json
            
            # Execute the generated code
            exec_globals = {'df': df, 'pd': pd, 'px': px, 'go': go, 'make_subplots': make_subplots}
            exec(generated_code, exec_globals)
            
            # Get the figure from executed code
            fig = exec_globals.get('fig')
            
            if fig is None:
                # If no figure was created, create a basic one
                fig = go.Figure()
                fig.add_annotation(text="No visualization generated", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Convert to HTML
            html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Add narrative to HTML
            if isinstance(insights, dict):
                narrative_html = "<div style='margin: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;'>"
                narrative_html += "<h3>Analysis Insights</h3>"
                for key, value in insights.items():
                    narrative_html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
                narrative_html += "</div>"
                
                # Create complete HTML
                complete_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Plotly Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
    </style>
</head>
<body>
    {narrative_html}
    {html_content}
</body>
</html>
"""
            else:
                complete_html = html_content
            
            # Save to file
            import os
            outputs_dir = f'outputs/simple_iteration/{shared_memory.thread_id}'
            os.makedirs(outputs_dir, exist_ok=True)
            
            html_path = os.path.join(outputs_dir, f'basic_analysis_llm_iteration_{iteration}.html')
            with open(html_path, 'w') as f:
                f.write(complete_html)
            print(f"✅ LLM-generated Plotly visualization saved to: {html_path}")
            success = True
            
        except Exception as e:
            print(f"Warning: Failed to create Plotly visualization: {e}")
            html_content = f"<div>Plotly visualization failed: {e}</div>"
            html_path = ""
            success = False
       
        visualization = Visualization(
            insight=f"Generated exploration analysis",
            chart_type=analysis_params.chart_type,
            altair_code=generated_code,
            description=f"Basic analysis",
            is_appropriate=success,
            image_path=html_path if 'html_path' in locals() else "",
            success=success,
            figure_object=html_content if success else "",
            code=""
        )
        
        facts = {
                "code": generated_code,
                "stdout": f"LLM Analysis Results:\nInsights: {insights}",
                "stderr": "",
                "exit_code": 0
            }
        # Convert insights from dict to list format for consistency
        if isinstance(insights, dict):
            insights_list = [f"{key.replace('_', ' ').title()}: {value}" for key, value in insights.items()]
        else:
            insights_list = insights if isinstance(insights, list) else [str(insights)]

        question_ = {
            "question": question,
            "handled": True,
            "spec": ""
        }
        current_iteration_data = {
            "question": question_,
            "facts": facts,
            "insights": insights_list,
            "visualizations": [visualization]
        }
        # print("Visualisation html:")
        # print(current_iteration_data['visualizations']['visualization'].figure_object)
        
        return current_iteration_data

        
        
        
    except Exception as e:
        print(f"Error in LLM line trend analysis: {e}")
        # Fallback to traditional method
        return execute_line_trend(df, analysis_params, iteration)
    
    

    




    
