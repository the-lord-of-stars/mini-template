from pydantic import BaseModel, Field
from typing import List

# 所有工具都应该有这些基础参数
class BaseAnalysisParameters(BaseModel):
    analysis_type: str = Field(..., description="Type of analysis")
    question_text: str = Field(..., description="The question to analyze")
    primary_attributes: List[str] = Field(..., description="Primary attributes for analysis")
    secondary_attributes: List[str] = Field(..., description="Secondary attributes for analysis")

class ToolDecision(BaseModel):
    """Schema for LLM tool selection decision"""
    tool_name: str = Field(..., description="Name of the tool to execute: top_keywords, temporal_evolution, or cooccurrence_matrix")
    reasoning: str = Field(..., description="Explanation for why this tool was selected")
