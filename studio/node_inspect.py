# node_task_discovery.py
import json
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from studio.helpers import get_llm


# ---------- base functions ----------
def _hash_str(s: str) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:8]

def _safe_topk_categorical(series: pd.Series, k: int = 10) -> List[Dict[str, Any]]:
    vc = series.astype(str).value_counts(dropna=False).head(k)
    return [{"hash": _hash_str(idx), "count": int(cnt)} for idx, cnt in vc.items()]

def _numeric_hist(series: pd.Series, bins: int = 20) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(clean, bins=bins)
    return {"bins": [float(x) for x in edges.tolist()], "counts": [int(x) for x in counts.tolist()]}

def _quantiles(series: pd.Series) -> Dict[str, float]:
    q = series.quantile([0, 0.25, 0.5, 0.75, 1.0], interpolation="linear")
    return {str(k): float(v) for k, v in q.items()}

def _infer_time_granularity(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    valid = s.dropna()
    if valid.empty:
        if series.dtype.kind in ("i", "u") and series.between(1800, 2100).mean() > 0.8:
            return {"is_time_like": True, "granularity": "year", "coverage": None}
        return {"is_time_like": False}
    diffs = valid.sort_values().diff().dropna()
    if diffs.empty:
        gran = "unknown"
    else:
        md = diffs.dt.total_seconds().median()
        if md <= 60: gran = "minute/second"
        elif md <= 3600: gran = "hour"
        elif md <= 86400: gran = "day"
        elif md <= 86400 * 31: gran = "month"
        else: gran = "year+"
    return {
        "is_time_like": True,
        "granularity": gran,
        "coverage": {"start": str(valid.min()), "end": str(valid.max()), "n": int(len(valid))}
    }

# --------- meta-functions -----------
def list_columns_impl(df: pd.DataFrame) -> Dict[str, Any]:
    return {"columns": list(df.columns), "dtypes": {c: str(df[c].dtype) for c in df.columns}}

def column_profile_impl(df: pd.DataFrame, columns: List[str], bins: int = 20, topk: int = 10) -> Dict[str, Any]:
    out = {}
    for col in columns:
        if col not in df.columns:
            out[col] = {"error": "column_not_found"}
            continue
        s = df[col]
        info = {
            "dtype": str(s.dtype),
            "count": int(s.size),
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=False))
        }
        if pd.api.types.is_numeric_dtype(s):
            info["histogram"] = _numeric_hist(s, bins=bins)
            info["quantiles"] = _quantiles(pd.to_numeric(s, errors="coerce").dropna())
        else:
            info["topk_hashed"] = _safe_topk_categorical(s, k=topk)
        out[col] = info
    return out

def relationship_probe_impl(df: pd.DataFrame, x: str, y: str, op: str, target: Optional[str]) -> Dict[str, Any]:
    if x not in df.columns or y not in df.columns:
        return {"error": "column_not_found"}
    if op == "corr":
        xn = pd.to_numeric(df[x], errors="coerce")
        yn = pd.to_numeric(df[y], errors="coerce")
        mask = ~(xn.isna() | yn.isna())
        if mask.sum() < 3:
            return {"corr": None, "n": int(mask.sum())}
        corr = float(np.corrcoef(xn[mask], yn[mask])[0, 1])
        return {"corr": corr, "n": int(mask.sum())}
    if op == "groupby_count":
        vc = df[[x, y]].astype(str).value_counts().head(20)
        return [{"x": _hash_str(ix[0]), "y": _hash_str(ix[1]), "count": int(cnt)} for ix, cnt in vc.items()]
    if op == "groupby_sum":
        if not target or target not in df.columns:
            return {"error": "target_required"}
        g = df.groupby([x, y], dropna=False)[target].sum().sort_values(ascending=False).head(20)
        return [{"x": _hash_str(ix[0]), "y": _hash_str(ix[1]), "sum": float(val)} for ix, val in g.items()]
    return {"error": "unknown_op"}

def time_granularity_impl(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if col not in df.columns:
        return {"error": "column_not_found"}
    return _infer_time_granularity(df[col])

# ---------- schema ----------
def tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": "list_columns", "description": "List dataframe columns and dtypes. Use this first.",
                         "parameters": {"type": "object", "properties": {}, "required": []}}
        },
        {
            "type": "function",
            "function": {"name": "column_profile", "description": "Return safe profiles for selected columns (no raw values).",
                         "parameters": {"type": "object",
                                        "properties": {
                                            "columns": {"type": "array", "items": {"type": "string"}},
                                            "bins": {"type": "integer", "minimum": 5, "maximum": 100, "default": 20},
                                            "topk": {"type": "integer", "minimum": 3, "maximum": 50, "default": 10}
                                        },
                                        "required": ["columns"]}}
        },
        {
            "type": "function",
            "function": {"name": "relationship_probe", "description": "Probe relationship between two columns via aggregates.",
                         "parameters": {"type": "object",
                                        "properties": {
                                            "x": {"type": "string"},
                                            "y": {"type": "string"},
                                            "op": {"type": "string", "enum": ["corr", "groupby_count", "groupby_sum"]},
                                            "target": {"type": "string", "nullable": True}
                                        },
                                        "required": ["x", "y", "op"]}}
        },
        {
            "type": "function",
            "function": {"name": "time_granularity", "description": "Detect time-likeness and granularity of a column.",
                         "parameters": {"type": "object",
                                        "properties": {"col": {"type": "string"}},
                                        "required": ["col"]}}
        },
        {
            "type": "function",
            "function": {"name": "finalize_tasks", "description": "Submit final list of promising analysis tasks.",
                         "parameters": {"type": "object",
                                        "properties": {
                                            "analysis_tasks": {"type": "array", "items": {
                                                "type": "object",
                                                "properties": {
                                                    "objective": {"type": "string"},
                                                    "target_columns": {"type": "array", "items": {"type": "string"}},
                                                    "reason": {"type": "string"},
                                                    "priority": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                                                    "suggested_ops": {"type": "array", "items": {"type": "string"}},
                                                    "description": {"type": "string"},
                                                },
                                                "required": ["objective", "target_columns", "reason", "priority", "suggested_ops", "description"],
                                            }}
                                        },
                                        "required": ["analysis_tasks"]}}
        },
    ]

# ---------- The Main Loop ----------
def discover_tasks_with_function_calls(state: Dict[str, Any], llm, max_rounds: int = 6) -> Dict[str, Any]:
    df: pd.DataFrame = state["dataframe"]

    system = SystemMessage(content=(
        "You are a senior data analyst. Your mission is to discover analysis tasks focusing on "
        "IMPORTANT TOPICS and PEOPLE over TIME (\"important topic and person over time\").\n"
        "\n"
        "Privacy rule: you never receive raw values. Tools only return aggregates or hashed keys.\n"
        "\n"
        "Workflow:\n"
        "  (1) Always start with list_columns.\n"
        "  (2) Explore with column_profile / time_granularity / relationship_probe in multiple rounds.\n"
        "  (3) When READY, call finalize_tasks with 5–8 promising tasks "
        "(each has objective, target_columns, reason, priority, suggested_ops and description of the current trend or distribution).\n"
        "\n"
        "Topic focus:\n"
        "  • TIME: If any time-like column exists (e.g., 'Year', 'Date', 'Time', 'Timestamp'), use time_granularity on it.\n"
        "  • TOPICS: Use categorical/text-ish columns (e.g., 'AuthorKeywords', 'PaperType') as topic proxies.\n"
        "  • PEOPLE: Use author/affiliation/award columns (e.g., 'AuthorNames', 'AuthorAffiliation', 'Award') as person proxies.\n"
        "\n"
        "Required evidence BEFORE finalize (do not finalize until all satisfied):\n"
        "  • Run column_profile for at least 5 distinct columns combining time, numeric (e.g., citations/downloads), and categorical.\n"
        "  • Run time_granularity on at least one time-like column (if present).\n"
        "  • Run at least 3 relationship_probe calls covering:\n"
        "      - corr(time, numeric) such as corr(Year, CitationCount_* or Downloads_*),\n"
        "      - corr(numeric, numeric) e.g., corr(Downloads_*, CitationCount_*),\n"
        "      - groupby_count(time, categorical) e.g., groupby_count(Year, AuthorKeywords) or (Year, AuthorNames).\n"
        "\n"
        "Selection heuristics:\n"
        "  • Prefer columns whose profiles show variation (wide quantiles/histograms) or high frequency categories in top-k.\n"
        "  • Prefer relationships showing strong signals (non-trivial corr, skewed counts across years).\n"
        "  • If multiple time-like or candidate topic/person columns exist, sample more than one.\n"
        "\n"
        "Final task list MUST be diverse and theme-aligned:\n"
        "  • Include at least: (a) 2+ time-trend tasks (topics or people over time),\n"
        "    (b) 1+ numeric correlation task (e.g., downloads vs citations),\n"
        "    (c) 1+ category-over-time task (e.g., keywords or authors over years),\n"
        "    (d) 1 task explicitly about PEOPLE (authors/affiliations/awards) over time.\n"
        "\n"
        "Use only these values for suggested_ops: "
        "[\"line_trend\",\"scatter_corr\",\"bar_group\",\"box_by_category\",\"histogram\",\"heatmap_xy\"].\n"
        "Map your intent to the closest allowed op(s). Avoid vague labels like 'visualization' or 'trends'.\n"
        "\n"
        "If evidence is insufficient, keep exploring with tools instead of finalizing.\n"
    ))

    user = HumanMessage(content="Discover which aspects are worth investigating for this dataset. Use tools and finish by calling finalize_tasks.")
    messages: List[Any] = [system, user]

    bound = llm.bind_tools(tool_schemas())
    for _ in range(max_rounds):
        ai: AIMessage = bound.invoke(messages)
        messages.append(ai)

        print("\n--- AIMessage ---")
        print(ai.content)
        print("--- tool_calls ---")
        print(ai.tool_calls)


        if not ai.tool_calls:
            print("No tool used")
            continue

        for tc in ai.tool_calls:
            name = tc["name"]

            raw_args = tc.get("args") or {}
            if isinstance(raw_args, str):
                args = json.loads(raw_args)
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            if name == "list_columns":
                result = list_columns_impl(df)
            elif name == "column_profile":
                result = column_profile_impl(df, args.get("columns", []),
                                             bins=args.get("bins", 20),
                                             topk=args.get("topk", 10))
            elif name == "relationship_probe":
                result = relationship_probe_impl(df, args.get("x"), args.get("y"),
                                                 args.get("op"), args.get("target"))
            elif name == "time_granularity":
                result = time_granularity_impl(df, args.get("col"))
            elif name == "finalize_tasks":
                tasks = args.get("analysis_tasks", [])
                return {
                    **state,
                    "analysis_tasks": tasks,
                    "messages": state.get("messages", []) + [
                        AIMessage(content="Task discovery completed via function calls.")]
                }
            else:
                result = {"error": "unknown_tool"}

            messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                name=name,
                tool_call_id=tc["id"]
            ))

    return {
        **state,
        "analysis_tasks": [],
        "messages": state.get("messages", []) + [AIMessage(content="No finalize_tasks call within step limit.")]
    }

# ---------- LangGraph Node----------
def task_discovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return discover_tasks_with_function_calls(state, llm=get_llm(), max_rounds=6)

if __name__ == "__main__":
    df = pd.read_csv("./dataset.csv", encoding='utf-8')

    result = discover_tasks_with_function_calls(
        {"dataframe": df, "messages": []},
        llm=get_llm(),
        max_rounds=6
    )

    print(result["analysis_tasks"])
