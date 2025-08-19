"""
Simple analysis schema based on the unified design principle.
Schema serves as both input (planning) and output (vis/insight generation) format.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class QuestionType(str, Enum):
    """Question types for analysis"""
    AUTHOR_RANKING = "author_ranking"
    TOPIC_ANALYSIS = "topic_analysis"
    COLLABORATION_ANALYSIS = "collaboration_analysis"
    STATISTICAL_OVERVIEW = "statistical_overview"
    NETWORK_STRUCTURE = "network_structure"
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"

class PrimaryAttribute(str, Enum):
    """Primary attributes for analysis"""
    AUTHORS = "authors"
    PAPERS = "papers"
    COLLABORATIONS = "collaborations"
    TOPICS = "topics"
    VENUES = "venues"
    YEARS = "years"

class SecondaryAttribute(str, Enum):
    """Secondary attributes for analysis"""
    PUBLICATION_COUNT = "publication_count"
    CITATION_COUNT = "citation_count"
    COLLABORATION_COUNT = "collaboration_count"
    YEAR = "year"
    VENUE = "venue"
    TOPIC = "topic"
    AUTHOR_NAME = "author_name"
    PAPER_TITLE = "paper_title"

class TransformationType(str, Enum):
    """Data transformation types"""
    AGGREGATION_WITH_FILTERING = "aggregation_with_filtering"
    RANKING_AND_SORTING = "ranking_and_sorting"
    NETWORK_ANALYSIS = "network_analysis"
    STATISTICAL_COMPUTATION = "statistical_computation"
    CLUSTERING = "clustering"
    TIME_SERIES_ANALYSIS = "time_series_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"

class InsightType(str, Enum):
    """Expected insight types"""
    TOP_CONTRIBUTORS = "top_contributors"
    PRODUCTIVITY_PATTERNS = "productivity_patterns"
    COLLABORATION_PATTERNS = "collaboration_patterns"
    NETWORK_STRUCTURE = "network_structure"
    TREND_ANALYSIS = "trend_analysis"
    STATISTICAL_SUMMARY = "statistical_summary"
    COMPARATIVE_INSIGHTS = "comparative_insights"
    OUTLIER_DETECTION = "outlier_detection"

class VisualizationType(str, Enum):
    """Visualization types"""
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    NETWORK_GRAPH = "network_graph"
    MATRIX = "matrix"
    LINE_CHART = "line_chart"
    HEATMAP = "heatmap"
    PIE_CHART = "pie_chart"
    HISTOGRAM = "histogram"

class AnalysisParameters(BaseModel):
    """Parameters for analysis"""
    top_n: Optional[int] = Field(default=10, description="Number of top items to show")
    min_papers: Optional[int] = Field(default=1, description="Minimum number of papers for inclusion")
    time_period: Optional[str] = Field(default="all", description="Time period for analysis")
    min_collaborations: Optional[int] = Field(default=0, description="Minimum number of collaborations")
    include_self_citations: Optional[bool] = Field(default=False, description="Include self-citations")
    network_threshold: Optional[int] = Field(default=1, description="Minimum collaboration strength for network")
    clustering_method: Optional[str] = Field(default="kmeans", description="Clustering method if applicable")

class AnalysisPlan(BaseModel):
    """Structured analysis plan schema"""
    question_text: str = Field(..., description="The original user question")
    q_type: QuestionType = Field(..., description="Type of question/analysis")
    primary_attributes: List[PrimaryAttribute] = Field(..., description="Primary attributes to analyze")
    secondary_attributes: List[SecondaryAttribute] = Field(..., description="Secondary attributes to analyze")
    transformation: TransformationType = Field(..., description="Data transformation method")
    expected_insights: List[InsightType] = Field(..., description="Types of insights expected")
    parameters: AnalysisParameters = Field(default_factory=AnalysisParameters, description="Analysis parameters")
    visualization_types: List[VisualizationType] = Field(..., description="Types of visualizations to create")
    analysis_focus: Optional[str] = Field(default="general", description="Focus area of the analysis")
    reasoning: Optional[str] = Field(default="", description="Reasoning for this analysis plan")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "question_text": "who are the main authors in sensemaking",
                "q_type": "author_ranking",
                "primary_attributes": ["authors"],
                "secondary_attributes": ["publication_count", "collaboration_count"],
                "transformation": "aggregation_with_filtering",
                "expected_insights": ["top_contributors", "productivity_patterns"],
                "parameters": {
                    "top_n": 10,
                    "min_papers": 2,
                    "time_period": "all"
                },
                "visualization_types": ["bar_chart", "scatter_plot"],
                "analysis_focus": "productivity_ranking",
                "reasoning": "Query seeks to identify the most productive authors in the sensemaking field"
            }
        }

# Convenience functions for creating analysis plans
def create_author_ranking_plan(question: str, top_n: int = 10, min_papers: int = 1) -> AnalysisPlan:
    """Create an author ranking analysis plan"""
    return AnalysisPlan(
        question_text=question,
        q_type=QuestionType.AUTHOR_RANKING,
        primary_attributes=[PrimaryAttribute.AUTHORS],
        secondary_attributes=[SecondaryAttribute.PUBLICATION_COUNT, SecondaryAttribute.COLLABORATION_COUNT],
        transformation=TransformationType.AGGREGATION_WITH_FILTERING,
        expected_insights=[InsightType.TOP_CONTRIBUTORS, InsightType.PRODUCTIVITY_PATTERNS],
        parameters=AnalysisParameters(top_n=top_n, min_papers=min_papers),
        visualization_types=[VisualizationType.BAR_CHART, VisualizationType.SCATTER_PLOT],
        analysis_focus="productivity_ranking"
    )

def create_collaboration_analysis_plan(question: str, network_threshold: int = 1) -> AnalysisPlan:
    """Create a collaboration analysis plan"""
    return AnalysisPlan(
        question_text=question,
        q_type=QuestionType.COLLABORATION_ANALYSIS,
        primary_attributes=[PrimaryAttribute.AUTHORS, PrimaryAttribute.COLLABORATIONS],
        secondary_attributes=[SecondaryAttribute.COLLABORATION_COUNT, SecondaryAttribute.PUBLICATION_COUNT],
        transformation=TransformationType.NETWORK_ANALYSIS,
        expected_insights=[InsightType.COLLABORATION_PATTERNS, InsightType.NETWORK_STRUCTURE],
        parameters=AnalysisParameters(network_threshold=network_threshold),
        visualization_types=[VisualizationType.NETWORK_GRAPH, VisualizationType.MATRIX],
        analysis_focus="collaboration_network"
    )

def create_statistical_overview_plan(question: str) -> AnalysisPlan:
    """Create a statistical overview analysis plan"""
    return AnalysisPlan(
        question_text=question,
        q_type=QuestionType.STATISTICAL_OVERVIEW,
        primary_attributes=[PrimaryAttribute.AUTHORS, PrimaryAttribute.PAPERS],
        secondary_attributes=[SecondaryAttribute.PUBLICATION_COUNT, SecondaryAttribute.YEAR],
        transformation=TransformationType.STATISTICAL_COMPUTATION,
        expected_insights=[InsightType.STATISTICAL_SUMMARY],
        parameters=AnalysisParameters(),
        visualization_types=[VisualizationType.BAR_CHART, VisualizationType.HISTOGRAM],
        analysis_focus="dataset_overview"
    )