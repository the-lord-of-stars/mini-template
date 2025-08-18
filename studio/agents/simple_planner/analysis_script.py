#!/usr/bin/env python3
"""
Script: Identify exemplar papers per automated-visualization subtype
Reads ../../dataset.csv, searches Title+Abstract+AuthorKeywords for keywords
mapping to subtypes: recommendation, generation, mixed-initiative, agents,
and pipeline automation. Ranks matches by CitationCount_CrossRef (fallback
AminerCitationCount). Prints a short ranked list per subtype, distribution
counts, and papers matching multiple subtypes (outliers).

Requires: pandas, numpy
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_dataset(path):
    # Try to infer separator; be permissive
    try:
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', low_memory=False)
    except Exception:
        # fallback to comma
        df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    return df


def to_float_safe(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    path = Path('../../dataset.csv')
    if not path.exists():
        print(f"Dataset not found at {path.resolve()}")
        sys.exit(1)

    df = read_dataset(path)

    # Ensure required columns exist
    for col in ['Title', 'Abstract', 'AuthorKeywords', 'CitationCount_CrossRef', 'AminerCitationCount', 'Year', 'Conference']:
        if col not in df.columns:
            df[col] = ''

    # Prepare searchable text
    df['Title'] = df['Title'].fillna('').astype(str)
    df['Abstract'] = df['Abstract'].fillna('').astype(str)
    df['AuthorKeywords'] = df['AuthorKeywords'].fillna('').astype(str)
    df['search_text'] = (df['Title'] + ' ' + df['Abstract'] + ' ' + df['AuthorKeywords']).str.lower()

    # Define subtype keyword patterns (case-insensitive via lowercasing above)
    subtype_patterns = {
        'recommendation': [r'visualization recommendation', r'visualization-recommendation', r'\brecommendation\b'],
        'generation': [r'visualization generation', r'vis generation', r'\bautomatic vis\b', r'\bautomated vis\b', r'\bvisualization generation\b', r'\bgenerate visualization\b', r'\bgenerate vis\b'],
        'mixed-initiative': [r'mixed[- ]initiative', r'\bmixed initiative\b'],
        'agents': [r'\bagent\b', r'\bagents\b', r'agent-based', r'agent based'],
        'pipeline automation': [r'\bpipeline\b', r'\bautomation\b', r'pipeline automation', r'automated pipeline']
    }

    # Compile combined regex per subtype
    compiled = {}
    for k, pats in subtype_patterns.items():
        combined = '|'.join(f'(?:{p})' for p in pats)
        compiled[k] = re.compile(combined, flags=re.IGNORECASE)

    # Detect matches
    for subtype, cre in compiled.items():
        df[f'match__{subtype}'] = df['search_text'].apply(lambda s: bool(cre.search(s)))

    # Collect matched subtype lists
    def collect_matches(row):
        matches = [s for s in subtype_patterns.keys() if row.get(f'match__{s}', False)]
        return matches

    df['matched_subtypes'] = df.apply(collect_matches, axis=1)
    df['matched_any'] = df['matched_subtypes'].apply(lambda x: len(x) > 0)

    # Convert citation columns to numeric and build score
    df['CitationCount_CrossRef'] = pd.to_numeric(df['CitationCount_CrossRef'], errors='coerce')
    df['AminerCitationCount'] = pd.to_numeric(df['AminerCitationCount'], errors='coerce')
    # Score precedence: CitationCount_CrossRef > AminerCitationCount > 0
    df['score'] = df['CitationCount_CrossRef'].fillna(df['AminerCitationCount']).fillna(0)

    matched = df[df['matched_any']].copy()

    # Distribution
    distribution = {s: int(matched[f'match__{s}'].sum()) for s in subtype_patterns.keys()}

    # Print summary
    print('\nAutomated-visualization subtype search summary')
    print('Dataset rows:', len(df))
    print('Total matched rows:', len(matched))
    print('\nDistribution (count of matched papers per subtype):')
    for s, c in distribution.items():
        print(f' - {s}: {c}')

    # Ranked lists per subtype (top 5)
    TOP_K = 5
    print('\nTop exemplars per subtype (ranked by CitationCount_CrossRef then AminerCitationCount):')
    for subtype in subtype_patterns.keys():
        sub_df = matched[matched[f'match__{subtype}']].copy()
        if sub_df.empty:
            print(f"\n{subtype}: (no matches)")
            continue
        sub_df = sub_df.sort_values(by='score', ascending=False)
        print(f"\n{subtype} (top {min(TOP_K, len(sub_df))}):")
        for i, row in enumerate(sub_df.head(TOP_K).itertuples(), start=1):
            title = getattr(row, 'Title', '')
            year = getattr(row, 'Year', '')
            conf = getattr(row, 'Conference', '')
            score = getattr(row, 'score', 0)
            matched_list = getattr(row, 'matched_subtypes', [])
            print(f" {i}. {title} ({conf}, {year}) — score={int(score) if not np.isnan(score) else 0} — matches={matched_list}")

    # Outliers: papers matching multiple subtypes
    multi = matched[matched['matched_subtypes'].apply(lambda x: len(x) > 1)].copy()
    print('\nPapers matching multiple subtypes (outliers):')
    if multi.empty:
        print(' None found')
    else:
        for i, row in enumerate(multi.sort_values(by='score', ascending=False).itertuples(), start=1):
            title = getattr(row, 'Title', '')
            year = getattr(row, 'Year', '')
            conf = getattr(row, 'Conference', '')
            score = getattr(row, 'score', 0)
            matches = getattr(row, 'matched_subtypes', [])
            print(f" {i}. {title} ({conf}, {year}) — score={int(score) if not np.isnan(score) else 0} — matches={matches}")

    # Optionally: write matched results to CSV for inspection
    out_csv = Path('matched_autovis_papers.csv')
    matched_cols = ['Title', 'Year', 'Conference', 'score', 'matched_subtypes', 'Abstract']
    try:
        matched[matched_cols].to_csv(out_csv, index=False)
        print(f"\nWrote matched papers to {out_csv.resolve()}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
