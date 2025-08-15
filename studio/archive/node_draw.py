import json
import shutil
from typing import Any

import numpy as np

# from studio.state import State
from state import State
import os, uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def filter_by_time_scope(df: pd.DataFrame, time_scope: dict, time_col: str) -> tuple[Any, str]:
    dff = df.copy()
    start, end, interval = time_scope["start_year"], time_scope["end_year"], time_scope["interval_years"]
    dff = dff[(dff[time_col] >= start) & (dff[time_col] <= end)]
    bucket_col = time_col
    if interval and interval > 1:
        # bucket by interval
        dff[bucket_col] = (dff[time_col] // interval) * interval
    return dff, bucket_col

def split_cols_by_dtype(df: pd.DataFrame, cols: list, time_col_hint: str | None = None):
    nums, cats, times = [], [], []
    time_col = None
    for c in cols:
        lc = c.lower()
        if time_col is None and (("year" in lc) or ("date" in lc) or ("time" in lc)):
            time_col = c
        if c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                nums.append(c)
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                times.append(c)
                if time_col is None: time_col = c
            else:
                cats.append(c)
    if time_col is None and time_col_hint and time_col_hint in df.columns:
        time_col = time_col_hint
    return nums, cats, times, time_col

# --------- draw functions ---------

def fig_line_trend(dff, bucket_col, nums, objective):
    # Year-only line chart
    if not nums:
        agg = dff.groupby(bucket_col, as_index=False).size().rename(columns={"size":"count"})
        fig = px.line(agg, x=bucket_col, y="count", markers=True, title=objective)
    else:
        traces = []
        for ycol in nums:
            agg = dff.groupby(bucket_col, as_index=False)[ycol].mean()
            traces.append(go.Scatter(x=agg[bucket_col], y=agg[ycol], mode="lines+markers", name=f"mean({ycol})"))
        fig = go.Figure(traces)
        fig.update_layout(title=objective, xaxis_title=bucket_col, yaxis_title="mean(value)")
    return fig

def fig_scatter_corr(dff, cols, objective, color_col=None):
    # choose two numeric columns to plot
    cand = [c for c in cols if pd.api.types.is_numeric_dtype(dff[c])]
    if len(cand) < 2: return None
    xcol, ycol = cand[:2]
    fig = px.scatter(dff, x=xcol, y=ycol, color=color_col, opacity=0.7, title=objective)
    return fig


def fig_bar_group(dff, bucket_col, cats, objective,
                  width=1200, height=600,
                  bargap=0.15, bargroupgap=0.05):
    if not cats or bucket_col not in dff.columns:
        return None
    ccol = cats[0]
    if ccol not in dff.columns:
        return None

    agg = dff.groupby([bucket_col, ccol], as_index=False).size()

    # ensure x-axis order
    if pd.api.types.is_numeric_dtype(agg[bucket_col]):
        x_order = sorted(agg[bucket_col].unique())
    else:
        try:
            x_order = sorted(agg[bucket_col].unique(), key=lambda x: pd.to_datetime(x))
        except Exception:
            x_order = agg[bucket_col].unique().tolist()

    fig = px.bar(
        agg,
        x=bucket_col,
        y="size",
        color=ccol,
        barmode="group",
        title=objective,
        category_orders={bucket_col: x_order}
    )

    fig.update_layout(
        width=width,
        height=height,
        bargap=bargap,
        bargroupgap=bargroupgap,
        xaxis=dict(tickangle=-30, automargin=True),
        yaxis=dict(automargin=True),
        template="plotly_dark",
        margin=dict(l=60, r=30, t=60, b=40),
    )
    return fig


def fig_box_by_category(dff, cols, objective):
    nums = [c for c in cols if pd.api.types.is_numeric_dtype(dff[c])]
    cats = [c for c in cols if not pd.api.types.is_numeric_dtype(dff[c])]
    if not nums or not cats: return None
    fig = px.box(dff, x=cats[0], y=nums[0], points="suspectedoutliers", title=objective)
    return fig

def fig_histogram(dff, cols, objective):
    nums = [c for c in cols if pd.api.types.is_numeric_dtype(dff[c])]
    if not nums: return None
    fig = px.histogram(dff, x=nums[0], nbins=40, title=objective)
    return fig

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)

def fig_heatmap_xy(
    dff: pd.DataFrame,
    cols: list,
    bucket_col: str,
    objective: str,
    y_top_k: int = 30,
    normalize: str | None = None,     # None | "row" | "col"
    fallback_bin_numeric: bool = True,
    bins: int = 10,
    zmax_percentile: float = 99.0
):
    if bucket_col not in dff.columns or dff.empty:
        return None

    # choose y attribute
    cat_candidates = [c for c in cols if (c in dff.columns and c != bucket_col and not _is_numeric(dff[c]))]
    ycol = None
    if cat_candidates:
        ycol = cat_candidates[0]
    else:
        if fallback_bin_numeric:
            num_candidates = [c for c in cols if (c in dff.columns and _is_numeric(dff[c]) and c != bucket_col)]
            if num_candidates:
                base = num_candidates[0]
                ycol = "_binned_" + base
                s = pd.to_numeric(dff[base], errors="coerce")
                q = int(np.clip(bins, 1, max(1, s.notna().sum())))
                try:
                    cats = pd.qcut(s, q=q, duplicates="drop")
                except Exception:
                    # if qcut fails, try cut instead
                    try:
                        cats = pd.cut(s, bins=min(bins, max(1, s.nunique())), include_lowest=True)
                    except Exception:
                        cats = pd.Series(np.where(s.notna(), "valid", "NaN"), index=s.index)
                dff = dff.assign(**{ycol: cats.astype(str)})
        if ycol is None:
            return None

    # pick x attribute
    df2 = dff[[bucket_col, ycol]].copy().dropna(subset=[bucket_col, ycol])
    if df2.empty:
        return None

    # count
    pair = df2.groupby([bucket_col, ycol], dropna=False).size().reset_index(name="count")

    # Top-k
    if y_top_k and pair[ycol].nunique() > y_top_k:
        totals = pair.groupby(ycol)["count"].sum().sort_values(ascending=False)
        keep_y = totals.head(y_top_k).index
        pair = pair[pair[ycol].isin(keep_y)].copy()

    if pair.empty:
        return None

    # x plan order
    x_ser = pair[bucket_col]
    if _is_datetime(x_ser):
        # dt/period
        if pd.api.types.is_period_dtype(x_ser):
            x_dt = x_ser.astype('period[M]').dt.to_timestamp() if x_ser.dtype.freq else x_ser.astype(str)
        else:
            x_dt = x_ser
        x_order = pair.assign(_x_dt=pd.to_datetime(x_ser, errors="coerce")) \
                      .sort_values("_x_dt") \
                      [bucket_col].drop_duplicates().tolist()
    else:
        # transform to string
        try_dt = pd.to_datetime(x_ser, errors="coerce")
        if try_dt.notna().any():
            x_order = pair.assign(_x_dt=try_dt).sort_values("_x_dt")[bucket_col].drop_duplicates().tolist()
        elif _is_numeric(x_ser):
            x_order = sorted(x_ser.unique().tolist())
        else:
            x_order = x_ser.drop_duplicates().tolist()

    # y plan order
    y_order = pair.groupby(ycol)["count"].sum().sort_values(ascending=False).index.tolist()

    # pivot table
    tab = pair.pivot_table(index=ycol, columns=bucket_col, values="count", fill_value=0, aggfunc="sum")
    tab = tab.reindex(index=y_order, columns=x_order, fill_value=0)
    Z = tab.values.astype(float)

    # normalize
    ztitle = "count"
    if normalize == "row":
        row_sum = Z.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        Z = Z / row_sum
        ztitle = "row share"
    elif normalize == "col":
        col_sum = Z.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        Z = Z / col_sum
        ztitle = "col share"

    # z range
    zmin = 0.0
    if ztitle == "count":
        zpos = Z[Z > 0]
        zmax = float(np.percentile(zpos, zmax_percentile)) if zpos.size > 0 else None
    else:
        zmax = 1.0

    heatmap_kwargs = dict(z=Z, x=x_order, y=y_order, colorbar=dict(title=ztitle), zmin=zmin)
    if zmax is not None:
        heatmap_kwargs["zmax"] = zmax

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))
    fig.update_layout(
        title=objective,
        xaxis_title=bucket_col,
        yaxis_title=ycol,
        margin=dict(l=60, r=30, t=60, b=40),
        width=1200,
        height=600
    )
    return fig

# --------- fact generation -------
def generate_line_trend_metrics(dff, bucket_col, nums, objective):
    """
    Generate enhanced analysis metrics for line chart with trend pattern analysis
    """
    metrics_text = ""

    if not nums:
        agg = dff.groupby(bucket_col, as_index=False).size().rename(columns={"size": "count"})
        if len(agg) < 2:
            return "No enough results for trend analysis."

        data_col = "count"
        var_name = "Record Count"

        # Basic metrics
        start_val = agg['count'].iloc[0]
        end_val = agg['count'].iloc[-1]
        growth_rate = ((end_val - start_val) / start_val) * 100 if start_val > 0 else 0
        max_val = agg['count'].max()
        max_time = agg.loc[agg['count'] == max_val, bucket_col].iloc[0]
        min_val = agg['count'].min()
        min_time = agg.loc[agg['count'] == min_val, bucket_col].iloc[0]

        # Enhanced pattern analysis
        max_index = agg['count'].idxmax()
        min_index = agg['count'].idxmin()
        total_points = len(agg)
        max_position = (max_index / (total_points - 1)) * 100

        # Calculate phase-specific growth rates
        if max_index > 0:
            pre_peak_growth = ((max_val - start_val) / start_val) * 100 if start_val > 0 else 0
        else:
            pre_peak_growth = 0

        if max_index < total_points - 1:
            post_peak_decline = ((end_val - max_val) / max_val) * 100 if max_val > 0 else 0
        else:
            post_peak_decline = 0

        # Determine trend pattern
        if max_position < 25:
            pattern = "early peak then decline"
        elif max_position > 75:
            pattern = "gradual rise to late peak"
        elif 25 <= max_position <= 75:
            pattern = "rise then decline"
        else:
            pattern = "complex trend"

        years_after_peak = total_points - 1 - max_index

        metrics_text = f"""Chart type: line chart (counting)
Analysis objective: {objective}
Variable: {var_name}
Time range: {agg[bucket_col].iloc[0]} to {agg[bucket_col].iloc[-1]}
Trend pattern: {pattern}
Start value: {start_val}
End value: {end_val}
Overall growth rate: {growth_rate:.1f}%
Peak value: {max_val} (Year: {max_time})
Peak timing: {max_position:.0f}% through period
Pre-peak growth: {pre_peak_growth:.1f}%
Post-peak change: {post_peak_decline:.1f}%
Years after peak: {years_after_peak}
Minimum: {min_val} (Year: {min_time})"""

    else:
        ycol = nums[0]
        agg = dff.groupby(bucket_col, as_index=False)[ycol].mean()
        if len(agg) < 2:
            return f"{ycol}: No enough results for trend analysis."

        # Basic metrics
        start_val = agg[ycol].iloc[0]
        end_val = agg[ycol].iloc[-1]
        growth_rate = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
        max_val = agg[ycol].max()
        max_time = agg.loc[agg[ycol] == max_val, bucket_col].iloc[0]
        min_val = agg[ycol].min()
        min_time = agg.loc[agg[ycol] == min_val, bucket_col].iloc[0]

        # Enhanced pattern analysis
        max_index = agg[ycol].idxmax()
        min_index = agg[ycol].idxmin()
        total_points = len(agg)
        max_position = (max_index / (total_points - 1)) * 100

        # Calculate phase-specific growth rates
        if max_index > 0:
            pre_peak_growth = ((max_val - start_val) / start_val) * 100 if start_val != 0 else 0
        else:
            pre_peak_growth = 0

        if max_index < total_points - 1:
            post_peak_decline = ((end_val - max_val) / max_val) * 100 if max_val != 0 else 0
        else:
            post_peak_decline = 0

        # Determine trend pattern
        if max_position < 25:
            pattern = "early peak then decline"
        elif max_position > 75:
            pattern = "gradual rise to late peak"
        elif 25 <= max_position <= 75:
            pattern = "rise then decline"
        else:
            pattern = "complex trend"

        years_after_peak = total_points - 1 - max_index

        metrics_text = f"""Chart type: line chart (numerical)
Analysis objective: {objective}
Variable: {ycol}
Time range: {agg[bucket_col].iloc[0]} to {agg[bucket_col].iloc[-1]}
Trend pattern: {pattern}
Start value: {start_val:.2f}
End value: {end_val:.2f}
Overall growth rate: {growth_rate:.1f}%
Peak value: {max_val:.2f} (Year: {max_time})
Peak timing: {max_position:.0f}% through period
Pre-peak growth: {pre_peak_growth:.1f}%
Post-peak change: {post_peak_decline:.1f}%
Years after peak: {years_after_peak}
Minimum: {min_val:.2f} (Year: {min_time})
Average: {agg[ycol].mean():.2f}"""

    return metrics_text

def generate_line_trend_fact(dff, bucket_col, nums, objective, metrics_text):
    fact = ""
    insight = ""

    # LLM reasoning - from metrics to facts
    prompt_fact = f"""You are a professional data scientist. Given a IEEE VIS publication record data, here are some data analysis results. Based on the following metrics, generate 1-3 SHORT, atomic facts.

    {metrics_text}

    Requirements:
    - Each fact should be ONE clear, specific observation
    - Focus on the most important finding
    - Be direct and concise
    - No intro text like "The analysis shows". """

    prompt_insight = f"""You are a professional data scientist. Given a IEEE VIS publication record data, here are some data analysis results and identified facts. Provide 2-3 ANALYTICAL INSIGHTS.

    {metrics_text}, {fact}

    Requirements:
    - Focus on what this reveals about the field's evolution and maturity
    - Explain the research community's attention patterns
    - Describe the field's life cycle stage
    - Be specific about implications for researchers in this area
    - Use bullet points
    - Avoid business jargon"""

    try:
        from helpers import get_llm
        llm = get_llm(temperature = 0.2)
        response_fact = llm.invoke(prompt_fact)
        fact = response_fact.content.strip()
        response_insight = llm.invoke(prompt_insight)
        insight = response_insight.content.strip()
    except Exception as e:
        print(f"LLM fact generation failed: {e}")
        return f"LLM fact generation failed: {e}"

    return fact, insight
# --------- reander functions ---------

def render_task_with_plotly(task: dict, df: pd.DataFrame, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    cols = task["target_columns"]
    ops  = task.get("suggested_ops", [])
    time_scope = task.get("time_scope") or {}
    objective = task.get("objective", "Chart")

    # attribute split
    nums, cats, times, time_col = split_cols_by_dtype(df, cols)
    dff = df.copy()
    bucket_col = time_col
    color_col = None

    # bucket
    if time_col and time_scope:
        dff, bucket_col = filter_by_time_scope(df, time_scope, time_col)
        color_col = bucket_col

    saved = []
    figures = []

    for op in ops:
        fig = None
        fact = ""
        insight = ""

        print(f"render {op}...")
        try:
            if op == "line_trend" and bucket_col:
                fig = fig_line_trend(dff, bucket_col, [c for c in nums if c != time_col], objective)
                metrics_text = generate_line_trend_metrics(dff, bucket_col, [c for c in nums if c != time_col], objective)
                fact,insight = generate_line_trend_fact(dff, bucket_col, [c for c in nums if c != time_col], objective,metrics_text)
            elif op == "scatter_corr":
                fig = fig_scatter_corr(dff, cols, objective, color_col=color_col)
            elif op == "bar_group" and bucket_col:
                fig = fig_bar_group(dff, bucket_col, [c for c in cats if c != time_col], objective)
            elif op == "box_by_category":
                fig = fig_box_by_category(dff, cols, objective)
            elif op == "histogram":
                fig = fig_histogram(dff, cols, objective)
            elif op == "heatmap_xy":
                fig = fig_heatmap_xy(dff, cols, bucket_col, objective)
            else:
                reason = "unsupported_op"
        except Exception as e:
            reason = f"exception: {e}"
            fact = f"Error when generating {op} plot: {str(e)}"
            insight = f"Error when generating {op} plot: {str(e)}"

        if fig:
            # unify style
            fig.update_layout(
                template="plotly_white",
                width=1200,
                height=600,
                margin=dict(l=60, r=30, t=60, b=40),
            )
            fig.update_xaxes(scaleanchor=None)

            figures.append({
                'figure': fig,
                'op': op,
                'objective': objective,
                'fact': fact,
                'insight': insight
            })

            fname = f"{uuid.uuid4().hex[:8]}_{op}.png"
            path = os.path.join(out_dir, fname)
            fig.write_image(path, scale=2.0)  # 需要安装 kaleido
            saved.append(path)
    return saved, figures


def react_analysis_node(state: dict) -> dict:
    # clear old charts
    if os.path.exists("./charts"):
        shutil.rmtree("./charts")

    df: pd.DataFrame = state["dataframe"]

    tasks = state.get("analysis_tasks", [])
    tasks = [t for t in tasks if t.get("suggested_ops") != ["groupby_count"]]
    tasks = sorted(tasks, key=lambda x: x["priority"], reverse=True)[:5]

    artifacts = []
    all_figures = []
    for task in tasks:
        out_files, figures = render_task_with_plotly(task, df, out_dir=state.get("out_dir", "./charts"))
        artifacts.append({
            "objective": task.get("objective"),
            "files": out_files,
            "ops": task.get("suggested_ops", []),
            "figures": figures
        })
        all_figures.extend(figures)

    state["artifacts"] = artifacts
    state["all_figures"] = all_figures
    return state





# test
if __name__ == "__main__":
    state = State(messages=[])

    # read ./analysis_tasks.json file and set it to state
    with open("analysis_tasks.json", "r", encoding="utf-8") as f:
        analysis_tasks = json.load(f)
    state.update({"analysis_tasks":analysis_tasks})

    # read ./dataset.csv file and set it to states
    df = pd.read_csv("./dataset.csv", encoding='utf-8')
    state.update({"dataframe":df})

    # react_analysis_node(state)
    # test
    updated_state = react_analysis_node(state)
    print(f"Number of figs generated: {len(updated_state.get('all_figures', []))}")
    print(f"for {len(updated_state.get('artifacts', []))} sets of analysis")

    # generate html report - need to move to the last step in the main workflow
    html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Data Analysis Report - brief</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    background-color: #f5f5f5;
                }
                .task-section { 
                    background-color: white;
                    margin: 20px 0; 
                    padding: 20px; 
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .task-title { 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #333;
                    margin-bottom: 10px;
                }
                .task-info { 
                    background-color: #f8f9fa;
                    padding: 10px; 
                    margin: 10px 0;
                    border-left: 4px solid #007bff;
                }
                .chart-container { 
                    margin: 20px 0; 
                }
                .chart-title {
                    font-size: 18px;
                    font-weight: bold;
                    color: #555;
                    margin-bottom: 10px;
                }.fact-box {
                    background-color: #e8f4fd;
                    border-left: 4px solid #1f77b4;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                    font-style: italic;
                    color: #2c3e50;
                    line-height: 1.6;
                }
                .fact-label {
                    font-weight: bold;
                    font-style: normal;
                    color: #1f77b4;
                    margin-bottom: 8px;
                }.insight-box {
                    background-color: #e8f4fd;
                    border-left: 4px solid #1f77b4;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                    font-style: italic;
                    color: #2c3e50;
                    line-height: 1.6;
                }
                .insight-label {
                    font-weight: bold;
                    font-style: normal;
                    color: #1f77b4;
                    margin-bottom: 8px;
                }
                h1 { 
                    text-align: center; 
                    color: #333;
                }
            </style>
        </head>
        <body>
            <h1>Data analysis report</h1>
        """

    # Task section
    artifacts = updated_state.get("artifacts", [])

    for i, artifact in enumerate(artifacts):
        task_num = i + 1
        objective = artifact.get("objective", f"task {task_num}")
        ops = artifact.get("ops", [])
        figures = artifact.get("figures", [])

        html_content += f"""
            <div class="task-section">
                <div class="task-title">Task {task_num}: {objective}</div>
                <div class="task-info">
                    <strong>Operation:</strong> {', '.join(ops)}<br>
                    <strong>Number of gigures generated:</strong> {len(figures)}
                </div>
            """

        for j, fig_data in enumerate(figures):
            figure = fig_data["figure"]
            op = fig_data["op"]
            fact = fig_data.get("fact", "None")
            insight = fig_data.get("insight", "None")

            div_id = f"chart_{task_num}_{j + 1}"

            html_content += f"""
                <div class="chart-container">
                    <div class="chart-title">Figure: {op}</div>
                    <div id="{div_id}"></div>
                    <div class="fact-box">
                        <div class="fact-label">Facts:</div>
                        {fact}
                    </div>
                    <div class="insight-box">
                        <div class="insight-label">Insights:</div>
                        {insight}
                    </div>
                </div>
                """

        html_content += "</div>"

    html_content += """
        <script>
        """

    for i, artifact in enumerate(artifacts):
        figures = artifact.get("figures", [])
        task_num = i + 1

        for j, fig_data in enumerate(figures):
            figure = fig_data["figure"]
            div_id = f"chart_{task_num}_{j + 1}"

            fig_json = figure.to_json()

            html_content += f"""
                Plotly.newPlot('{div_id}', {fig_json});
                """

    html_content += """
        </script>
        </body>
        </html>
        """

    with open("newoutput.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("HTML saved: output.html")
    print(f"Number of analysis tasks: {len(artifacts)}")
    print(f"Number of figures: {sum(len(artifact.get('figures', [])) for artifact in artifacts)}")

