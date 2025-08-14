import json
import hashlib
from typing import Any, Dict, List, Optional

import math
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

def _nice_step(value: float, allowed=(1, 2, 5, 10)) -> float:
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    frac = value / base
    for a in allowed:
        if frac <= a:
            return a * base
    return 10 * base

def _infer_time_granularity(
    series: pd.Series,
    max_periods: int = 8,
    allowed_steps=(1, 2, 5, 10),
) -> Dict[str, Any]:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    valid = s.dropna()

    if valid.empty:
        if series.dtype.kind in ("i", "u") and series.between(1800, 2100).mean() > 0.8:
            return {
                "is_time_like": True,
                "granularity": "year",
                "year_interval": 1.0,
                "year_interval_mean": 1.0,
                "effective_interval_years": 1.0,
                "suggested_periods": None,
                "coverage": None,
            }
        return {"is_time_like": False}

    valid = valid.sort_values()
    diffs = valid.diff().dropna()

    # calculate year interval
    if diffs.empty:
        median_years = None
        mean_years = None
    else:
        year_diffs = diffs.dt.days / 365.25
        median_years = float(round(year_diffs.median(), 2))
        mean_years = float(round(year_diffs.mean(), 2))

    cov_start = valid.min()
    cov_end = valid.max()
    span_years = max(1e-6, (cov_end - cov_start).days / 365.25)  # 防0

    # target_interval
    target_interval = span_years / max(1, max_periods)
    nice_target_interval = _nice_step(target_interval, allowed=allowed_steps)

    # some heuristics to choose the effective interval
    if median_years is None:
        effective_interval = nice_target_interval
    else:
        effective_interval = max(median_years, nice_target_interval)

    # suggested_periods
    suggested_periods = int(math.ceil(span_years / max(effective_interval, 1e-6)))

    # generate granularity string
    if effective_interval < 0.5:
        gran = "month"
    elif effective_interval < 1.5:
        gran = "year"
    else:
        gran = f"{int(round(effective_interval))}-year"

    return {
        "is_time_like": True,
        "granularity": gran,
        "year_interval": median_years,
        "year_interval_mean": mean_years,
        "effective_interval_years": float(effective_interval),
        "suggested_periods": suggested_periods,
        "coverage": {
            "start": str(cov_start),
            "start_year": int(cov_start.year),
            "end": str(cov_end),
            "end_year": int(cov_end.year),
            "span_years": float(round(span_years, 2)),
            "n": int(len(valid))
        }
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
            "function": {
                "name": "finalize_tasks",
                "description": "Submit final list of promising analysis tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_tasks": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "objective": {"type": "string"},
                                    "target_columns": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "reason": {"type": "string"},
                                    "priority": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 10,
                                        "default": 5
                                    },
                                    "suggested_ops": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": [
                                                "line_trend",
                                                "scatter_corr",
                                                "bar_group",
                                                "box_by_category",
                                                "histogram",
                                                "heatmap_xy",
                                            ]
                                        }
                                    },
                                    "description": {"type": "string"},
                                    "time_scope": {
                                        "type": "object",
                                        "properties": {
                                            "start_year": {"type": "integer"},
                                            "end_year": {"type": "integer"},
                                            "interval_years": {"type": "number"}
                                        }
                                    }
                                },
                                "required": ["objective", "target_columns", "reason", "priority", "suggested_ops",
                                             "description", "time_scope"]
                            }
                        }
                    },
                    "required": ["analysis_tasks"]
                }
            }
        }

    ]

# ---------- The Main Loop ----------
def discover_tasks_with_function_calls(state: Dict[str, Any]) -> Dict[str, Any]:
    max_rounds: int = 20
    llm = get_llm(temperature=0.5)

    df: pd.DataFrame = state["dataframe"]

    system = SystemMessage(content=(
        "You are a senior data analyst tasked with discovering analysis tasks about autor network and themes changing you can at most call 19 times tool functions."
        "IMPORTANT INSIGHTS must be focusing on the author's network of relationships and themes change over time.(All the insights must be under one big context)\n"
        "\n"
        "Privacy: Never receive raw values. Tools return only aggregates or hashed keys.\n"
        "\n"
        "Workflow:\n"
        "  1. Start with list_columns.\n"
        "  2. Explore via column_profile / time_granularity / relationship_probe in multiple rounds.\n"
        "  3. Finalize with 8 tasks (objective, target_columns, reason, priority, suggested_ops, "
        "description ≥50 words) when evidence is sufficient. Each of them must have different suggested ops\n"
        "\n"
        "Focus:\n"
        "  • TIME: Use time_granularity for time-like columns (e.g., Year, Date).\n"
        "  • TOPICS: Categorical/text columns (e.g., AuthorKeywords, PaperType).\n"
        "  • PEOPLE: Author/affiliation/award columns (e.g., AuthorNames, AuthorAffiliation, Award).\n"
        "\n"
        "Required before finalize:\n"
        "  • column_profile on ≥5 distinct columns (time, numeric, categorical).\n"
        "  • time_granularity on ≥1 time-like column.\n"
        "  • ≥3 relationship_probe calls: corr(time, numeric), corr(numeric, numeric), groupby_count(time, categorical).\n"
        "\n"
        "Selection:\n"
        "  • Prefer varied distributions (wide quantiles, skewed counts).\n"
        "  • Prefer strong relationships (non-trivial corr, skewed counts over time).\n"
        "  • Use multiple time-like/topic/person columns if available.\n"
        "\n"
        "Final tasks must:\n"
        "  • Include: ≥2 time-trend, ≥1 numeric correlation, ≥1 category-over-time, ≥1 people-over-time.\n"
        "  • Very Important:Each suggested_op [lne_trend, scatter_corr, bar_group, box_by_category, histogram, heatmap_xy] must be used at least once.\n"
        "  • For each suggested_op, ensure target_columns strictly match the required data types:\n"
        "      - line_trend: time, numeric (+optional category)\n"
        "      - scatter_corr: numeric, numeric (+optional category)\n"
        "      - bar_group: time, category\n"
        "      - box_by_category: numeric, category\n"
        "      - histogram: numeric (+optional category)\n"
        "      - heatmap_xy: time/category, category (counts or binned numeric)\n"
        "  • suggested_op must be from [lne_trend, scatter_corr, bar_group, box_by_category, histogram, heatmap_xy] it must not be groupby_count.\n"
        "  • You must include time_scope {start_year, end_year, interval_years} from evidence if there's a trend to make your analysis actionable and the interval_years should be reasonable.\n"
        "  • Objects must be measurable with clear metrics and exact year cycle.\n"
        "  • Descriptions must analyze trends/distributions in detail (≥50 words), "
        "with clear reasoning on importance.\n"
        "\n"
        "Map intent to closest allowed op(s). Avoid vague labels.\n"
        "\n"
        "If evidence is insufficient, keep exploring before finalize.\n"
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
    return discover_tasks_with_function_calls(state)

# ---------- test ------------
if __name__ == "__main__":
    df = pd.read_csv("./dataset.csv", encoding='utf-8')

    result = discover_tasks_with_function_calls(
        {"dataframe": df, "messages": []}
    )

    for task in result["analysis_tasks"]:
        print(f"task: {task["objective"]}, priority: {task["priority"]}, suggested_ops: {task["suggested_ops"]}")

    # output result["analysis_tasks"] to a json file
    with open("analysis_tasks.json", "w", encoding="utf-8") as f:
        json.dump(result["analysis_tasks"], f, ensure_ascii=False, indent=4)