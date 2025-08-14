import pandas as pd
import json
import re
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from state import State
from helpers import get_llm


class KeywordExtractionAgent:
    """React Agent for extracting optimal keywords from AuthorKeywords data"""

    def __init__(self, df: pd.DataFrame, domain: str):
        self.df = df
        self.domain = domain
        self.llm = get_llm(max_completion_tokens=2048)
        self.max_iterations = 5
        self.target_range = (20, 500)  # Target number of matching papers

    def sample_keywords_tool(self, n_samples: int = 150) -> List[str]:
        """Sample AuthorKeywords from the dataset"""
        if 'AuthorKeywords' not in self.df.columns:
            return []

        keywords_series = self.df['AuthorKeywords'].dropna()
        if len(keywords_series) == 0:
            return []

        sample_size = min(n_samples, len(keywords_series))
        sampled = keywords_series.sample(sample_size).tolist()
        return sampled

    def test_keyword_tool(self, keyword: str) -> Dict[str, Any]:
        """Test how many papers match a specific keyword"""
        if 'AuthorKeywords' not in self.df.columns:
            return {"keyword": keyword, "matches": 0, "sample_matches": []}

        keyword_lower = keyword.lower()
        matches = []
        sample_matches = []

        for idx, author_keywords in self.df['AuthorKeywords'].items():
            if pd.notna(author_keywords) and isinstance(author_keywords, str):
                if keyword_lower in author_keywords.lower():
                    matches.append(idx)
                    if len(sample_matches) < 5:  # Keep first 5 as samples
                        sample_matches.append(author_keywords)

        return {
            "keyword": keyword,
            "matches": len(matches),
            "sample_matches": sample_matches
        }

    def analyze_keyword_distribution(self, keywords_sample: List[str]) -> Dict[str, Any]:
        """Analyze the distribution of keywords to understand patterns"""
        all_words = []
        for kw_string in keywords_sample:
            if isinstance(kw_string, str):
                # Split by common delimiters
                words = re.split(r'[;,\-\s]+', kw_string.lower())
                words = [w.strip() for w in words if w.strip() and len(w.strip()) > 2]
                all_words.extend(words)

        # Count word frequency
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "total_words": len(all_words),
            "unique_words": len(word_counts),
            "top_words": top_words
        }

    def run(self) -> List[str]:
        """Run the React Agent to extract optimal keywords"""
        print(f"KeywordAgent: Starting keyword extraction for domain '{self.domain}'")

        # Step 1: Sample and analyze data
        keywords_sample = self.sample_keywords_tool()
        if not keywords_sample:
            print("KeywordAgent: No AuthorKeywords data found, using fallback")
            return self._fallback_keywords()

        word_analysis = self.analyze_keyword_distribution(keywords_sample)

        # Step 2: LLM analysis and initial keyword generation
        initial_keywords = self._llm_analyze_and_generate(keywords_sample, word_analysis)

        # Step 3: React loop - test and refine keywords
        final_keywords = self._react_optimization_loop(initial_keywords)

        print(f"KeywordAgent: Final keywords extracted: {final_keywords}")
        return final_keywords

    def _llm_analyze_and_generate(self, keywords_sample: List[str], word_analysis: Dict) -> List[str]:
        """Use LLM to analyze sample data and generate initial keywords"""

        # Limit sample size for prompt
        sample_for_prompt = keywords_sample[:50]
        top_words = word_analysis["top_words"][:15]

        prompt = f"""
You are analyzing academic publication keywords to find papers about "{self.domain}".

Domain to analyze: "{self.domain}". If it is "none", it indicates the request is to analyse across the whole domain.

Sample AuthorKeywords from the dataset:
{sample_for_prompt}

Most frequent words: {top_words}

CRITICAL INSTRUCTIONS:
1. Focus on the CORE CONCEPT of "{self.domain}" - what is the main subject matter?
2. For "visualisation for sensemaking" → the key concept is "sensemaking", not just "visual"
3. For "narrative visualization" → focus on "narrative" or "storytelling"
4. Look for SPECIFIC terms that researchers use for this exact topic
5. Avoid generic visualization terms that match too many irrelevant papers

PRIORITIZE:
- Specific domain concepts (e.g., "sensemaking", "storytelling", "narrative")
- Exact terminology from the research field
- Terms that appear in the sample data related to your domain

AVOID generic terms like:
- "visualization" (too broad)
- "visual analytics" (too broad) 
- "interactive" (too generic)
- "visual" alone (matches everything)

Extract 2-3 SPECIFIC keywords that directly relate to the core concept:

Return ONLY a JSON list of precise keywords:
["keyword1", "keyword2"].
If the domain is none, return [""].
"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert at analyzing academic publication data."),
                HumanMessage(content=prompt)
            ])

            # Extract JSON from response
            json_match = re.search(r'\[.*?\]', response.content, re.DOTALL)
            if json_match:
                keywords = json.loads(json_match.group())
                return [kw.strip().lower() for kw in keywords if isinstance(kw, str)]
            else:
                print("KeywordAgent: Failed to parse LLM response, using fallback")
                return self._fallback_keywords()

        except Exception as e:
            print(f"KeywordAgent: LLM analysis failed: {e}")
            return self._fallback_keywords()

    def _react_optimization_loop(self, initial_keywords: List[str]) -> List[str]:
        """React loop to test and optimize keywords"""

        current_keywords = initial_keywords.copy()
        best_keywords = []
        best_score = 0

        for iteration in range(self.max_iterations):
            print(f"KeywordAgent: Iteration {iteration + 1}, testing keywords: {current_keywords}")

            # Test current keywords
            results = []
            for keyword in current_keywords:
                result = self.test_keyword_tool(keyword)
                results.append(result)
                print(f"  - '{keyword}': {result['matches']} matches")

            # Evaluate results
            total_matches = sum(r['matches'] for r in results)

            # Score based on target range
            if self.target_range[0] <= total_matches <= self.target_range[1]:
                score = 100  # Perfect range
            elif total_matches < self.target_range[0]:
                score = total_matches / self.target_range[0] * 50  # Too few
            else:
                score = self.target_range[1] / total_matches * 50  # Too many

            print(f"  Total matches: {total_matches}, Score: {score:.1f}")

            # Update best result
            if score > best_score:
                best_score = score
                best_keywords = [kw for kw in current_keywords if self.test_keyword_tool(kw)['matches'] > 0]

            # If good enough, stop
            if score >= 80:
                print(f"KeywordAgent: Good result achieved (score: {score:.1f})")
                break

            # Refine keywords based on results
            current_keywords = self._refine_keywords(results, total_matches)

            if not current_keywords:
                break

        return best_keywords if best_keywords else self._fallback_keywords()

    def _refine_keywords(self, results: List[Dict], total_matches: int) -> List[str]:
        """Refine keywords based on test results"""

        # Keep keywords that have some matches
        working_keywords = [r['keyword'] for r in results if r['matches'] > 0]

        if total_matches < self.target_range[0]:
            # Too few matches - try to broaden
            if len(working_keywords) < 3:
                # Add more keywords
                broader_keywords = self._generate_broader_keywords(working_keywords)
                working_keywords.extend(broader_keywords)

        elif total_matches > self.target_range[1]:
            # Too many matches - try to narrow
            # Keep only the most specific keywords
            sorted_results = sorted([r for r in results if r['matches'] > 0],
                                    key=lambda x: x['matches'])
            working_keywords = [r['keyword'] for r in sorted_results[:2]]

        return working_keywords[:3]  # Keep it concise

    def _generate_broader_keywords(self, current_keywords: List[str]) -> List[str]:
        """Generate broader keywords when matches are too few"""
        broader = []

        for keyword in current_keywords:
            if 'sensemaking' in keyword:
                broader.extend(['sense-making', 'understanding', 'comprehension'])
            elif 'visualization' in keyword or 'visualisation' in keyword:
                broader.extend(['visual', 'vis', 'chart', 'graph'])
            elif 'interaction' in keyword:
                broader.extend(['interactive', 'user interface'])

        return list(set(broader))[:2]  # Return at most 2 additional keywords

    def _fallback_keywords(self) -> List[str]:
        """Fallback keywords when extraction fails"""
        domain_lower = self.domain.lower()

        if 'sensemaking' in domain_lower:
            return ['sensemaking']
        elif 'storytelling' in domain_lower:
            return ['storytelling', 'narrative']
        elif 'interaction' in domain_lower:
            return ['interaction', 'interactive']
        else:
            # Extract key terms from domain
            words = domain_lower.split()
            # Filter out common words
            stop_words = {'for', 'in', 'of', 'the', 'and', 'with', 'using', 'based'}
            key_words = [w for w in words if w not in stop_words and len(w) > 3]
            return key_words[:2] if key_words else ['visualization']


def data_process_node(state: State) -> Dict[str, Any]:
    """
    Smart data processing node with embedded keyword extraction agent.
    Filters dataset based on intelligently extracted keywords and time range.
    """
    updated_state = state.copy()

    if "messages" not in updated_state or not isinstance(updated_state["messages"], list):
        updated_state["messages"] = []

    try:
        # Get dataframe and task information
        df = updated_state.get("dataframe")
        if df is None:
            error_msg = "Node: Data Process Error - No dataframe found in state."
            print(error_msg)
            updated_state["messages"].append(AIMessage(content=error_msg))
            return updated_state

        task = updated_state.get("task", {})
        domain = task.get("domain", "")
        time_from = task.get("time_from", 1990)
        time_to = task.get("time_to", 2024)

        if not domain:
            error_msg = "Node: Data Process Error - No domain specified in task."
            print(error_msg)
            updated_state["messages"].append(AIMessage(content=error_msg))
            return updated_state

        print(f"Node: Data Process - Starting intelligent processing. Original shape: {df.shape}")
        print(f"Node: Data Process - Domain: '{domain}', Time range: {time_from}-{time_to}")

        # Step 1: Run keyword extraction agent
        keyword_agent = KeywordExtractionAgent(df, domain)
        extracted_keywords = keyword_agent.run()

        # Step 2: Filter by extracted keywords
        filtered_df = df.copy()

        if 'AuthorKeywords' in filtered_df.columns and extracted_keywords:
            keyword_mask = pd.Series([False] * len(filtered_df))

            for idx, author_keywords in filtered_df['AuthorKeywords'].items():
                if pd.notna(author_keywords) and isinstance(author_keywords, str):
                    author_keywords_lower = author_keywords.lower()
                    # Check if any extracted keyword appears
                    if any(keyword.lower() in author_keywords_lower for keyword in extracted_keywords):
                        keyword_mask.iloc[idx] = True

            filtered_df = filtered_df[keyword_mask]
            print(f"Node: Data Process - After keyword filtering with {extracted_keywords}: {filtered_df.shape}")
        else:
            print("Node: Data Process - Skipping keyword filtering (no AuthorKeywords or no keywords extracted)")

        # Step 3: Filter by time range
        if 'Year' in filtered_df.columns:
            filtered_df['Year'] = pd.to_numeric(filtered_df['Year'], errors='coerce')
            time_mask = (filtered_df['Year'] >= time_from) & (filtered_df['Year'] <= time_to)
            filtered_df = filtered_df[time_mask]
            print(f"Node: Data Process - After time filtering ({time_from}-{time_to}): {filtered_df.shape}")
        else:
            print("Node: Data Process - Skipping time filtering (no Year column)")

        # Step 4: Save results - simplified
        updated_state["processed_dataframe"] = filtered_df

        # Create summary
        if len(filtered_df) > 0:
            year_range = [int(filtered_df['Year'].min()),
                          int(filtered_df['Year'].max())] if 'Year' in filtered_df.columns else [time_from, time_to]

            summary = {
                "total_records": len(filtered_df),
                "year_range": year_range,
                "keywords_used": extracted_keywords,
                "domain": domain,
                "original_records": len(df)
            }

            updated_state["processed_summary"] = summary

            success_msg = f"Node: Data Process - Successfully processed data. Final shape: {filtered_df.shape}. Keywords used: {extracted_keywords}"
            print(success_msg)
            updated_state["messages"].append(AIMessage(content=success_msg))
        else:
            summary = {
                "total_records": 0,
                "year_range": [time_from, time_to],
                "keywords_used": extracted_keywords,
                "domain": domain,
                "original_records": len(df)
            }
            updated_state["processed_summary"] = summary

            warning_msg = f"Node: Data Process - No data found after filtering. Keywords: {extracted_keywords}"
            print(warning_msg)
            updated_state["messages"].append(AIMessage(content=warning_msg))

        return updated_state

    except Exception as e:
        error_msg = f"Node: Data Process Error - Processing failed. Error: {e}"
        print(error_msg)
        updated_state["messages"].append(AIMessage(content=error_msg))

        # Fallback: return original dataframe
        updated_state["processed_dataframe"] = updated_state.get("dataframe")
        updated_state["extracted_keywords"] = []

        return updated_state