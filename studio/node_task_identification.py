import json
import re
from datetime import datetime
from typing import Dict, Any
from langchain_core.messages import AIMessage
from state import State
from helpers import get_llm
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage


def task_identification_node(state: State) -> Dict[str, Any]:
    """
    Extracts task information from user query using LLM.
    Identifies domain, time range, and analysis type.
    """
    updated_state = state.copy()

    if "messages" not in updated_state or not isinstance(updated_state["messages"], list):
        updated_state["messages"] = []

    try:
        user_query = updated_state.get("user_query", "")
        # dataset_info = updated_state.get("dataset_info", "")

        if not user_query:
            error_msg = "Node: Task Identification Error - No user query provided."
            print(error_msg)
            updated_state["messages"].append(AIMessage(content=error_msg))
            return updated_state

        # Construct prompt for LLM
        prompt = f"""
User Query: "{user_query}"

Extract the following information and return as valid JSON:
1. domain: The specific research area/topic to analyze (e.g., "sensemaking"). Can also be "none" to indicate the request to analyse across the whole domain.
2. time_from: Start year for analysis (integer, e.g., 1990)
3. time_to: End year for analysis (integer, use current year if "now" or "present")

Examples:
- "analyse the development of visualisation for sensemaking" → domain: "visualisation for sensemaking"
- "from 1990 till now" → time_from: 1990, time_to: {datetime.now().year}

Return only valid JSON format:
{{
    "domain": "extracted domain",
    "time_from": start_year,
    "time_to": end_year,
}}
"""

        # Call LLM (assuming you have an LLM instance available)
        llm = get_llm(max_completion_tokens=4096)
        llm_response = llm.invoke([
            SystemMessage(
                content="You are analyzing a publication record analysis query. Extract key information for data analysis."),
            HumanMessage(content=prompt)
        ])

        # Parse LLM response
        try:
            # Clean the response to extract JSON
            json_match = re.search(r'\{.*\}', llm_response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                extracted_task = json.loads(json_str)
            else:
                raise ValueError("No JSON found in LLM response")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Node: Task Identification Warning - JSON parsing failed: {e}")
            # Fallback extraction using regex
            extracted_task = fallback_extraction(user_query)

        # Validate and set defaults
        task = {
            "domain": extracted_task.get("domain", "").strip(),
            "time_from": int(extracted_task.get("time_from", 1990)),
            "time_to": int(extracted_task.get("time_to", datetime.now().year)),
        }

        # Basic validation
        if task["time_from"] > task["time_to"]:
            task["time_from"], task["time_to"] = task["time_to"], task["time_from"]

        if not task["domain"]:
            # Try to extract domain from query as fallback
            task["domain"] = extract_domain_fallback(user_query)

        # Update state
        updated_state["task_domain"] = task["domain"]
        updated_state["task"] = task

        success_msg = f"Node: Task Identification - Successfully extracted task. Domain: '{task['domain']}', Time range: {task['time_from']}-{task['time_to']}"
        print(success_msg)
        updated_state["messages"].append(AIMessage(content=success_msg))

        return updated_state

    except Exception as e:
        error_msg = f"Node: Task Identification Error - Could not extract task information. Error: {e}"
        print(error_msg)
        updated_state["messages"].append(AIMessage(content=error_msg))

        # Set default task information
        default_task = {
            "domain": "visualization",
            "time_from": 1990,
            "time_to": datetime.now().year,
        }
        # updated_state["task"] = default_task["domain"]
        updated_state["task"] = default_task

        return updated_state


def fallback_extraction(user_query: str) -> Dict[str, Any]:
    """
    Fallback method to extract task information using regex patterns.
    """
    task = {}

    # Extract domain using common patterns
    domain_patterns = [
        r'development of (.+?)(?:\s+from|\s+in|\s+over|$)',
        r'analyse (?:the )?(.+?)(?:\s+from|\s+in|\s+over|$)',
        r'analyze (?:the )?(.+?)(?:\s+from|\s+in|\s+over|$)',
    ]

    for pattern in domain_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            task["domain"] = match.group(1).strip()
            break

    # Extract years
    year_pattern = r'(\d{4})'
    years = re.findall(year_pattern, user_query)
    if years:
        years = [int(y) for y in years if 1900 <= int(y) <= datetime.now().year]
        if len(years) >= 2:
            task["time_from"] = min(years)
            task["time_to"] = max(years)
        elif len(years) == 1:
            task["time_from"] = years[0]
            if "till now" in user_query.lower() or "to now" in user_query.lower():
                task["time_to"] = datetime.now().year

    # Extract analysis type
    if any(word in user_query.lower() for word in ["development", "develop", "evolution", "evolve"]):
        task["analysis_type"] = "development"
    elif any(word in user_query.lower() for word in ["trend", "trending", "pattern"]):
        task["analysis_type"] = "trends"
    else:
        task["analysis_type"] = "development"

    return task


def extract_domain_fallback(user_query: str) -> str:
    """
    Simple fallback to extract domain when LLM fails.
    """
    # Look for common visualization terms
    viz_terms = ["visualization", "visualisation", "visual analytics", "information visualization", "sensemaking"]

    for term in viz_terms:
        if term in user_query.lower():
            # Try to find context around the term
            words = user_query.lower().split()
            if term in words:
                idx = words.index(term)
                # Get surrounding context
                start = max(0, idx - 2)
                end = min(len(words), idx + 3)
                context = " ".join(words[start:end])
                return context

    return "sensemaking"
