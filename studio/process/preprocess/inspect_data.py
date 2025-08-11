from pathlib import Path
import pandas as pd

def inspect_data(path: str, sample_size: int = 5) -> list[dict]:
    analysis_plan = []
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        null_count = col_data.isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_vals = col_data.dropna().unique()
        unique_count = len(unique_vals)

        sample_vals = col_data.dropna().astype(str).unique()[:sample_size].tolist()

        avg_len = col_data.dropna().astype(str).apply(len).mean()

        # Infer the strategy
        if pd.api.types.is_numeric_dtype(col_data):
            if unique_count < 15:
                suggestion = "📊 Small-range numeric → categorical bar chart"
            elif unique_count < 100:
                suggestion = "📈 Continuous numeric → histogram / boxplot"
            else:
                suggestion = "📉 Long-range numeric → correlation, trend, binning"
        elif pd.api.types.is_string_dtype(col_data):
            if unique_count < 30:
                suggestion = "📊 Categorical text → frequency analysis"
            elif unique_count < 300:
                suggestion = "🤔 High-cardinality text → needs dimensionality reduction"
            elif avg_len > 30:
                suggestion = "🧠 Long text → NLP topic modeling"
            else:
                suggestion = "🔍 Possibly noisy / IDs / hash"
        else:
            suggestion = "❓ Unrecognized type"

        analysis_plan.append({
            "column": col,
            "dtype": dtype,
            "null_count": int(null_count),
            "null_pct": float(round(null_pct, 1)),
            "unique_count": int(unique_count),
            "sample_values": sample_vals,
            "suggested_analysis": suggestion
        })

    return analysis_plan
