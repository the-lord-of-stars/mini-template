import os
from typing import Dict, Any, List, Optional
from agents.star.state import State
from agents.star.report_html import generate_html_report, figure_to_base64, add_visualization_to_html
from agents.star.memory import shared_memory
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from agents.star.helpers import get_llm
from pathlib import Path
import re

class IterationSummary(BaseModel):
    """summary of one iteration"""
    title: str = Field(description="a very brief title of this iteration, no more than 10 words")
    # problem: str = Field(description="the problem this iteration is trying to solve")
    # analysis: str = Field(description="summary of the analysis method and approach")
    # findings: str = Field(description="key findings and insights from this iteration")
    narrative: str = Field(description="engaging, story-like paragraphs combining the problem, analysis, and findings for this iteration, no more than 200 words")

def format_insights_list(insights_text):
    """简单的insights格式化"""
    if not insights_text:
        return "No insights available"
    
    # 直接替换* 为HTML bullet point
    formatted = insights_text.replace('* ', '<br>• ')
    
    # 如果第一个字符是bullet point，移除开头的<br>
    if formatted.startswith('<br>• '):
        formatted = '• ' + formatted[6:]
    
    return formatted

class ReportSummary(BaseModel):
    """summary of the report"""
    intro: str = Field(description="short introduction to the analysis")
    quick_summary: str = Field(description="quick summary of all iterations")
    iterations: List[IterationSummary] = Field(description="detailed summary of each iteration")
    conclusion: str = Field(description="conclusion that ties all together and highlights the overall impact")

def build_analysis_prompt_structured(iteration_history: List[Dict]) -> str:
    """build the prompt for LLM, specifically for structured output"""
    
    prompt_parts = [
        "You are a professional data analyst specialized in data visualization research, skilled in writing engaging and coherent narrative reports.",
        "You have conducted a multi-iteration analysis on the topic of Sensemaking based on the IEEE VIS publication record dataset",
        "Your task: review the analysis process, and create a clear, coherent, story-driven professional report summary in a structured format",
        "for each iteration, you need to understand:",
        "1. the problem to solve",
        "2. the analysis method",  
        "3. the key findings and insights",
        "",
        "here is the iteration data:",
        ""
    ]
    
    for i, iteration in enumerate(iteration_history, 1):
        prompt_parts.append(f"=== Iteration {i} ===")
        
        # add question
        if "question" in iteration and iteration["question"]:
            question_text = iteration["question"].get("question", "no question specified")
            prompt_parts.append(f"question/goal: {question_text}")
            print("--------------------------------")
            print(f"question/goal: {question_text}")
        
        # add analysis code
        if "facts" in iteration and iteration["facts"]:
            facts = iteration["facts"]  
            # add analysis result
            if facts.get("stdout"):
                prompt_parts.append("analysis output:")
                # limit output length
                output = facts['stdout']
                if len(output) > 1000:
                    output = output[:1000] + "... (output is too long, truncated)"
                prompt_parts.append(f"```\n{output}\n```")
                print(f"facts: {output}")
        
        # add insights
        if "insights" in iteration and iteration["insights"]:
            prompt_parts.append("identified insights:")
            for j, insight in enumerate(iteration["insights"], 1):
                prompt_parts.append(f"{j}. {insight}")
                print(f"insight: {insight}")
                print("--------------------------------")
        
        prompt_parts.append("")
    
    # add output requirements
    prompt_parts.extend([
        "please generate the report summary based on the above information, including:",
        "",
        "1. intro: Set the scene: explain why the topic matters in a broader context based on your knowledge of the field. Introduce the problem or observation that triggered the analysis.Briefly state the investigation goal.",
        "2. quick_summary: Tell the overall “story arc” of the analysis in 3–5 sentences.Show how each iteration built upon the previous one (discovery → new question → deeper investigation → conclusions).Include any surprising or notable findings that shaped the direction.",
        "3. iterations: For each iteration, based on the research question and key findings, write ",
        "- 'title': a very brief title of this iteration, no more than 10 words",
        "- 'narrative': a detailed story-like paragraph that naturally integrates the problem, method, and findings. Use smooth transitions and a human, blog-like tone while staying professional.",
        "Add one sentence (if relevant) on how this iteration influenced the next step."
        "4. conclusion: Summarize the big-picture trends and their significance.If possible, Highlight the implications for the field. and Suggest possible next steps or open questions.",
        "",
        "please ensure Maintain a logical flow with smooth transitions. Avoid dry bullet points—write in full, well-structured sentences. Keep the language professional but accessible (avoid unnecessary jargon).Use occasional transition words and framing phrases (“Initially…”, “As the analysis progressed…”, “Interestingly…”)."
    ])
    
    return "\n".join(prompt_parts)

def generate_llm_summary_structured(iteration_history: List[Dict], llm) -> ReportSummary:
    """
    use LLM and structured output to analyze the iteration history and generate a structured report summary
    
    Args:
        iteration_history: history data of all iterations
        llm: LLM instance obtained through get_llm()
    
    Returns:
        ReportSummary: structured report summary
    """
    
    # 1. Extract sub-iteration history without visualization data for LLM analysis
    sub_iteration_history = []
    for iteration in iteration_history:
        sub_iteration = {
            "question": iteration.get("question", {}),
            "facts": iteration.get("facts", {}),
            "insights": iteration.get("insights", []),
        }
        sub_iteration_history.append(sub_iteration)
    
    # 2. build the input content for LLM
    analysis_content = build_analysis_prompt_structured(sub_iteration_history)
    
    # 3. use structured output to call LLM
    try:
        # use structured output of LLM
        print("-------------Analysis content-------------------")
        print(analysis_content)
        print("--------------------------------")
        summary = llm.with_structured_output(ReportSummary).invoke(analysis_content)
        print("-------------LLM output-------------------")
        print(summary)
        print("--------------------------------")
        
        return summary
        
    except Exception as e:
        print(f"Warning: Failed to generate structured LLM summary: {e}")
        
        # return default structured data
        iterations = []
        for i in range(len(iteration_history)):
            iteration = IterationSummary(
                title=f"iteration {i+1}",
                # problem=f"analysis of iteration {i+1}",
                # analysis="detailed analysis is unavailable",
                # findings="please check the detailed iteration results below",
                narrative="please check the detailed iteration results below"
            )

            iterations.append(iteration)
        
        return ReportSummary(
            intro="analysis summary is currently unavailable.",
            quick_summary="multiple data analysis iterations have been executed.",
            iterations=iterations,
            conclusion="please check the detailed iteration results below to understand the complete analysis."
        )

def extract_iteration_history(output_state: dict, shared_memory) -> List[Dict[str, Any]]:
    """
    从shared_memory中提取iteration_history
    
    Args:
        output_state: 输出状态字典
        shared_memory: 共享内存对象
    
    Returns:
        List[Dict[str, Any]]: 迭代历史列表
    """
    complete_history = []
    
    if shared_memory is not None:
        try:
            memory_states = shared_memory.get_history()
            # Group states by iteration and select the last state from each iteration
            iteration_groups = {}

            for state in memory_states:
                iteration_count = state.get("iteration_count", 0)
                if iteration_count > 0:  # Skip initial states without iteration
                    if iteration_count not in iteration_groups:
                        iteration_groups[iteration_count] = []
                    iteration_groups[iteration_count].append(state)

            # print(iteration_groups.keys())

            # Select the first complete state from each iteration
            for iteration_num in sorted(iteration_groups.keys()):
                states_in_iteration = iteration_groups[iteration_num]

                # Find the LAST state in this iteration that has complete information including visualizations
                selected_state = None
                for i, state in enumerate(reversed(states_in_iteration)):
                    if "question" in state and "facts" in state and "insights" in state:
                        # Prefer states with visualizations
                        if "visualizations" in state:
                            selected_state = state
                            break
                        elif selected_state is None:
                            selected_state = state
                
                # If no complete state found, use the last state
                if selected_state is None:
                    selected_state = states_in_iteration[-1]

                if selected_state:
                    visualization_data = None

                    # Method 1: Get from single field
                    if "visualization" in selected_state and selected_state["visualization"]:
                        visualization_data = selected_state["visualization"]
                        print(
                            f"✅ Found single visualization with keys: {list(visualization_data.keys()) if isinstance(visualization_data, dict) else type(visualization_data)}")

                    # Method 2: Get from plural field
                    elif "visualizations" in selected_state and selected_state["visualizations"]:
                        viz_container = selected_state["visualizations"]
                        if isinstance(viz_container, dict) and "visualizations" in viz_container:
                            viz_list = viz_container["visualizations"]
                            if isinstance(viz_list, list) and len(viz_list) > 0:
                                visualization_data = viz_list[-1]
                                print(f"✅ Found visualization from list")

                    if not visualization_data:
                        print("⚠️ No visualization found, creating empty structure")
                        visualization_data = {
                            "figure_object": None,
                            "code": "",
                            "altair_code": "",
                            "success": False
                        }
                    
                    iteration_data = {
                        "question": selected_state.get("question", {}),
                        "facts": selected_state.get("facts", {}),
                        "insights": selected_state.get("insights", []),
                        "visualization": visualization_data
                    }
                    complete_history.append(iteration_data)

        except Exception as e:
            print(f"Warning: Could not load history from shared memory: {e}")

    return complete_history

def synthesise(state: State) -> State:
    """
    Synthesise node: Generate HTML report from the current state and shared memory
    This node can be integrated into the workflow to generate reports at any point
    """
    
    try:
        # Create output directory if it doesn't exist
        output_dir = f"outputs/simple_iteration/{shared_memory.thread_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Generate final report
        print("-------------Synthesise node-------------------")
        iteration_history = extract_iteration_history(state, shared_memory)
        print("Iteration history collected")
        output_path = f"output.html"
        generate_html_report_final(iteration_history, output_path, state)
        print(f"HTML report generated: {output_path}")

        root_output_path = ""
        
        # Update state with report information
        new_state = state.copy()
        new_state["synthesise"] = {
            "report_generated": True,
            "report_path": output_path,
            "root_report_path": root_output_path,
            "success": True
        }
        
        return new_state
        
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        
        # Return state with error information
        new_state = state.copy()
        new_state["synthesise"] = {
            "report_generated": False,
            "error": str(e),
            "success": False
        }
        
        return new_state
def generate_html_report_final(iteration_history: List[Dict[str, Any]], output_path: str, output_state: Optional[Dict] = None) -> str:
    """
    Generate modern blog-style HTML report from iteration history
    """
    # Build the HTML document with modern blog styling
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>CiteMiner | Sensemaking Evolution</title>",
        "  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "  <link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap' rel='stylesheet'>",
        get_modern_css_styles(),
        "</head>",
        "<body>",
        
        # Header
        "  <header class='header'>",
        "    <div class='container'>",
        "      <div class='header-content'>",
        "        <a href='#' class='logo'>CiteMiner</a>",
        "        <nav>",
        "          <ul class='nav-links'>",
        "            <li><a href='#overview'>Overview</a></li>",
        "            <li><a href='#findings'>Findings</a></li>",
        "            <li><a href='#conclusion'>Conclusion</a></li>",
        "          </ul>",
        "        </nav>",
        "      </div>",
        "    </div>",
        "  </header>",
        
        "  <div class='container'>",
        "    <main class='main-content'>",
    ]

    if iteration_history:
        # Generate structured summary
        llm = get_llm(temperature=0.5)
        print("LLM initialized")
        summary = generate_llm_summary_structured(iteration_history, llm)
        print("Summary generated")
        # Add Overview section
        add_overview_section(html_lines, summary)
        print("Overview section added")
        # Add Findings section
        add_findings_section(html_lines, summary, iteration_history)
        print("Findings section added")
        # Add Conclusion section
        add_conclusion_section(html_lines, summary)
        print("Conclusion section added")
    
    # Close main content and add footer
    html_lines.extend([
        "       </section>",
        "    </main>",
        "  </div>",

        "  <!-- Interactive CTA Section -->",
        "  <section class='cta-section'>",
        "    <div class='container'>",
        "      <div class='cta-card'>",
        "        <h3>🔍 Explore More Research Insights</h3>",
        "        <p>Interested in analyzing other research topics or have follow-up questions about this analysis?</p>",
        "        <div class='cta-buttons'>",
        "          <button class='cta-button primary' onclick='showContactModal()'>",
        "            📊 Analyze New Dataset",
        "          </button>",
        "          <button class='cta-button secondary' onclick='showQuestionModal()'>",
        "            💬 Ask Follow-up Questions",
        "          </button>",
        "        </div>",
        "        <div class='cta-footer'>",
        "          <small>Want to explore your own research data? Contact us to learn more about CiteMiner's capabilities.</small>",
        "        </div>",
        "      </div>",
        "    </div>",
        "  </section>",
        
        # Footer
        "  <footer class='footer'>",
        "    <div class='container'>",
        "      <p>&copy; 2025 CiteMiner. Built for researchers, by researchers.</p>",
        "    </div>",
        "  </footer>",

        "  <!-- Modals -->",
        "  <div id='contactModal' class='modal'>",
        "    <div class='modal-content'>",
        "      <h4>🚀 Analyze New Dataset</h4>",
        "      <p>Tell us about the research data you'd like to explore:</p>",
        "      <input type='text' placeholder='Dataset topic (e.g., Machine Learning, Climate Science)' />",
        "      <textarea rows='4' placeholder='Describe your research questions and what insights you&apos;re looking for...'></textarea>",
        "      <div class='modal-buttons'>",
        "        <button class='modal-button cancel' onclick='closeModals()'>Cancel</button>",
        "        <button class='modal-button submit' onclick='submitDatasetRequest()'>Submit Request</button>",
        "      </div>",
        "    </div>",
        "  </div>",

        "  <div id='questionModal' class='modal'>",
        "    <div class='modal-content'>",
        "      <h4>💬 Follow-up Questions</h4>",
        "      <p>What would you like to know more about this analysis?</p>",
        "      <textarea rows='4' placeholder='Ask about specific findings, methodologies, or request deeper analysis on particular aspects...'></textarea>",
        "      <div class='modal-buttons'>",
        "        <button class='modal-button cancel' onclick='closeModals()'>Cancel</button>",
        "        <button class='modal-button submit' onclick='submitQuestion()'>Submit Question</button>",
        "      </div>",
        "    </div>",
        "  </div>",
        
        # JavaScript
        get_javascript_code(),
        
        "</body>",
        "</html>"
    ])

    # Write the file
    Path(output_path).write_text("\n".join(html_lines), encoding="utf-8")
    return output_path

def get_modern_css_styles() -> str:
    """Return the modern CSS styles as a string"""
    return """
  <style>
    :root {
      --primary-color: #6366f1;
      --primary-light: #a5b4fc;
      --secondary-color: #10b981;
      --accent-color: #f59e0b;
      --text-primary: #1f2937;
      --text-secondary: #6b7280;
      --text-muted: #9ca3af;
      --bg-primary: #ffffff;
      --bg-secondary: #f9fafb;
      --bg-accent: #f3f4f6;
      --border-color: #e5e7eb;
      --border-light: #f3f4f6;
      --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
      --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
      --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
      --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.7;
      color: var(--text-primary);
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 0 2rem;
    }

    /* Header */
    .header {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--border-light);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 1.5rem 0;
      margin-bottom: 2rem;
    }

    .header-content {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .logo {
      font-weight: 700;
      font-size: 1.5rem;
      color: var(--primary-color);
      text-decoration: none;
    }

    .nav-links {
      display: flex;
      gap: 2rem;
      list-style: none;
    }

    .nav-links a {
      color: var(--text-secondary);
      text-decoration: none;
      font-weight: 500;
      transition: color 0.2s;
    }

    .nav-links a:hover {
      color: var(--primary-color);
    }

    /* Main Content */
    .main-content {
      background: var(--bg-primary);
      border-radius: 24px;
      box-shadow: var(--shadow-xl);
      margin: 2rem auto 4rem;
      overflow: hidden;
    }

    /* Hero Section */
    .hero {
      background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
      color: white;
      padding: 4rem 3rem;
      position: relative;
      overflow: hidden;
    }

    .hero-content {
      position: relative;
      z-index: 2;
    }

    .hero h1 {
      font-family: 'Crimson Text', serif;
      font-size: 3.5rem;
      font-weight: 600;
      line-height: 1.2;
      margin-bottom: 1.5rem;
    }

    .hero-subtitle {
      font-size: 1.25rem;
      opacity: 0.9;
      margin-bottom: 2rem;
      max-width: 600px;
    }

    .hero-meta {
      display: flex;
      gap: 2rem;
      flex-wrap: wrap;
      opacity: 0.8;
    }

    .meta-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.9rem;
    }

    /* Article Sections */
    .article {
      padding: 3rem;
    }

    .article h2 {
      font-family: 'Crimson Text', serif;
      font-size: 2.5rem;
      color: var(--text-primary);
      margin: 3rem 0 1.5rem;
      position: relative;
    }

    .article h2::after {
      content: '';
      position: absolute;
      bottom: -0.5rem;
      left: 0;
      width: 60px;
      height: 3px;
      background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
      border-radius: 2px;
    }

    .article h3 {
      font-size: 1.5rem;
      color: var(--text-primary);
      margin: 2rem 0 1rem;
      font-weight: 600;
    }

    .article p {
      margin-bottom: 1.5rem;
      color: var(--text-secondary);
      font-size: 1.1rem;
    }

    /* Summary Cards */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 2rem;
      margin: 3rem 0;
    }

    .summary-card {
      background: var(--bg-secondary);
      border-radius: 16px;
      padding: 2rem;
      border: 1px solid var(--border-light);
      transition: transform 0.2s, box-shadow 0.2s;
    }

    .summary-card:hover {
      transform: translateY(-4px);
      box-shadow: var(--shadow-lg);
    }

    .summary-card h4 {
      color: var(--primary-color);
      font-size: 1.2rem;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Iteration Cards */
    .iteration-card {
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      border-radius: 20px;
      margin: 2rem 0;
      overflow: hidden;
      box-shadow: var(--shadow-md);
      transition: all 0.3s ease;
    }

    .iteration-card:hover {
      box-shadow: var(--shadow-lg);
      transform: translateY(-2px);
    }

    .iteration-header {
      background: linear-gradient(135deg, var(--secondary-color) 0%, #34d399 100%);
      color: white;
      padding: 2rem;
      position: relative;
    }

    .iteration-number {
      position: absolute;
      top: 1rem;
      right: 2rem;
      font-size: 3rem;
      font-weight: 700;
      opacity: 0.3;
    }

    .iteration-title {
      font-size: 1.5rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }

    .iteration-content {
      padding: 2rem;
    }

    .iteration-summary {
      font-size: 1.1rem;
      line-height: 1.8;
      color: var(--text-secondary);
      margin-bottom: 2rem;
    }

    /* Details Toggle */
    .details-toggle {
      background: var(--bg-accent);
      border-radius: 12px;
      margin: 1.5rem 0;
      overflow: hidden;
    }

    .details-toggle summary {
      padding: 1.5rem;
      cursor: pointer;
      font-weight: 600;
      color: var(--text-primary);
      list-style: none;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: background-color 0.2s;
    }

    .details-toggle summary:hover {
      background: var(--border-light);
    }

    .details-toggle summary::before {
      content: '▶';
      transition: transform 0.2s;
    }

    .details-toggle[open] summary::before {
      transform: rotate(90deg);
    }

    .details-content {
      padding: 0 1.5rem 1.5rem;
      border-top: 1px solid var(--border-color);
    }

    .detail-item {
      margin: 1rem 0;
      padding: 1rem;
      background: var(--bg-primary);
      border-radius: 8px;
    }

    .detail-label {
      font-weight: 600;
      color: var(--primary-color);
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Visualization Section */
    .viz-section {
      background: var(--bg-secondary);
      border-radius: 16px;
      padding: 2rem;
      margin: 2rem 0;
      border: 1px solid var(--border-light);
    }

    .viz-title {
      font-size: 1.3rem;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .chart-container {
      background: white;
      border-radius: 12px;
      padding: 1rem;
      box-shadow: var(--shadow-sm);
    }

    /* Code Toggle */
    .code-toggle {
      margin-top: 1rem;
      background: var(--text-primary);
      color: white;
      border-radius: 8px;
      overflow: hidden;
    }

    .code-toggle summary {
      padding: 1rem;
      cursor: pointer;
      font-family: 'Inter', monospace;
      font-size: 0.9rem;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .code-toggle pre {
      padding: 1rem;
      background: #1f2937;
      margin: 0;
      font-family: 'Monaco', 'Menlo', monospace;
      font-size: 0.85rem;
      overflow-x: auto;
    }

    /* Conclusion Section */
    .conclusion {
      background: linear-gradient(135deg, var(--accent-color) 0%, #f97316 100%);
      color: white;
      padding: 3rem;
      border-radius: 20px;
      margin: 3rem 0;
      position: relative;
      overflow: hidden;
    }

    .conclusion::before {
      content: '';
      position: absolute;
      top: -50%;
      right: -50%;
      width: 100%;
      height: 200%;
      background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
      pointer-events: none;
    }

    .conclusion h3 {
      font-family: 'Crimson Text', serif;
      font-size: 2rem;
      margin-bottom: 1.5rem;
      position: relative;
      z-index: 2;
    }

    .conclusion p {
      font-size: 1.1rem;
      line-height: 1.8;
      position: relative;
      z-index: 2;
      opacity: 0.95;
    }
    /* CTA Section */
.cta-section {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 4rem 0;
}

.cta-card {
  background: white;
  border-radius: 20px;
  padding: 3rem;
  text-align: center;
  box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  border: 1px solid #f3f4f6;
}

.cta-card h3 {
  color: #1f2937;
  font-size: 2rem;
  margin-bottom: 1rem;
  font-family: 'Crimson Text', serif;
}

.cta-card p {
  color: #6b7280;
  font-size: 1.2rem;
  margin-bottom: 2rem;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.cta-buttons {
  display: flex;
  gap: 1.5rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}

.cta-button {
  padding: 1rem 2rem;
  border: none;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.cta-button.primary {
  background: linear-gradient(135deg, #6366f1 0%, #10b981 100%);
  color: white;
}

.cta-button.primary:hover {
  transform: translateY(-2px);
}

.cta-button.secondary {
  background: white;
  color: #6366f1;
  border: 2px solid #6366f1;
}

.cta-button.secondary:hover {
  background: #6366f1;
  color: white;
}

.cta-footer {
  color: #9ca3af;
  font-size: 0.9rem;
}

/* Modal Styles */
.modal {
  display: none;
  position: fixed;
  z-index: 1000;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
}

.modal-content {
  background-color: white;
  margin: 10% auto;
  padding: 2rem;
  border-radius: 16px;
  width: 90%;
  max-width: 500px;
  box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1);
}

.modal textarea, .modal input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 1rem;
  font-family: inherit;
}

.modal-buttons {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

.modal-button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
}

.modal-button.cancel {
  background: #f3f4f6;
  color: #6b7280;
}

.modal-button.submit {
  background: #6366f1;
  color: white;
}

    /* Footer */
    .footer {
      text-align: center;
      padding: 3rem;
      color: var(--text-muted);
      background: var(--bg-secondary);
      border-top: 1px solid var(--border-light);
      margin-top: 4rem;
    }

    /* Responsive */
    @media (max-width: 768px) {
      .container {
        padding: 0 1rem;
      }
      
      .hero {
        padding: 3rem 2rem;
      }
      
      .hero h1 {
        font-size: 2.5rem;
      }
      
      .article {
        padding: 2rem;
      }
      
      .summary-grid {
        grid-template-columns: 1fr;
      }
      
      .nav-links {
        display: none;
      }
    }

    /* Animations */
    @keyframes fadeInUp {
      from {
        opacity: 0;
        transform: translateY(30px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .iteration-card {
      animation: fadeInUp 0.6s ease forwards;
    }

    .iteration-card:nth-child(2) { animation-delay: 0.1s; }
    .iteration-card:nth-child(3) { animation-delay: 0.2s; }
    .iteration-card:nth-child(4) { animation-delay: 0.3s; }
  </style>"""

def get_report_title(output_state) -> str:
    """Generate dynamic report title"""
    if output_state and "topic" in output_state:
        topic = output_state["topic"]
        return f"The Evolution of {topic.title()}"
    return "Research Insights Report"

def get_report_subtitle(output_state) -> str:
    """Generate dynamic report subtitle"""
    if output_state and "topic" in output_state:
        topic = output_state["topic"]
        return f"Exploring the dynamic landscape of {topic} research, uncovering collaboration networks, emerging topics, and methodological innovations that shaped the field."
    return "A comprehensive analysis of research trends and insights."

def add_overview_section(html_lines: List[str], summary: ReportSummary) -> None:
    """Add the Overview section"""
    html_lines.extend([
        "      <!-- Overview Section -->",
        "      <section class='article' id='overview'>",
        "        <h2>Overview</h2>",
        f"        <p>{summary.intro}</p>",
        "      </section>",
    ])

def add_findings_section(html_lines: List[str], summary: ReportSummary, iteration_history: List[Dict]) -> None:
    """Add the Findings section with iteration cards"""
    html_lines.extend([
        "      <!-- Findings Section -->",
        "      <section class='article' id='findings'>",
        "        <h2>Findings</h2>",
    ])
    
    for iteration_idx, iteration in enumerate(summary.iterations):
        history_data = iteration_history[iteration_idx]
        question = str(history_data["question"].get("question", "No question specified"))
        insights = format_insights_list(str(history_data["insights"]))
        html_lines.extend([
            "        <div class='iteration-card'>",
            "          <div class='iteration-header'>",
            f"            <div class='iteration-number'>{iteration_idx:02d}</div>",
            f"            <h3 class='iteration-title'>{get_iteration_title(iteration_idx, summary)}</h3>",
            "          </div>",
            "          <div class='iteration-content'>",
            f"            <p class='iteration-summary'>{iteration.narrative}</p>",
            "",
            "            <details class='details-toggle'>",
            "              <summary>📋 Detailed Analysis</summary>",
            "              <div class='details-content'>",
            "                <div class='detail-item'>",
            "                  <div class='detail-label'>🎯 Research Question</div>",
            # f"                  <p>{iteration.problem}</p>",
            f"                  <p>{question}</p>",
            "                </div>",
            "                <div class='detail-item'>",
            "                  <div class='detail-label'>🔍 Insights</div>",
            # f"                  <p>{iteration.analysis}</p>",
            # f"                  <p>{insights}</p>",
            f"                  <div>{insights}</div>",
            "                </div>",
            # "                <div class='detail-item'>",
            # "                  <div class='detail-label'>💡 Key Findings</div>",
            # f"                  <p>{iteration.findings}</p>",
            # "                </div>",
            "              </div>",
            "            </details>",
        ])

        print(f"adding iteration {iteration_idx} visualization ...")
        
        # 调试信息
        if iteration_idx < len(iteration_history):
            iteration_data = iteration_history[iteration_idx]
            print(f"🔍 Iteration {iteration_idx} keys: {list(iteration_data.keys())}")
            if "visualizations" in iteration_data:
                viz_data = iteration_data["visualizations"]
                print(f"🔍 Iteration {iteration_idx} visualizations type: {type(viz_data)}")
                print(f"🔍 Iteration {iteration_idx} visualizations length: {len(viz_data) if isinstance(viz_data, list) else 'Not a list'}")
                if isinstance(viz_data, list) and len(viz_data) > 0:
                    first_viz = viz_data[0]
                    print(f"🔍 Iteration {iteration_idx} first viz keys: {list(first_viz.keys()) if isinstance(first_viz, dict) else 'Not a dict'}")
                    if isinstance(first_viz, dict) and "figure_object" in first_viz:
                        print(f"🔍 Iteration {iteration_idx} figure_object exists: {bool(first_viz['figure_object'])}")
        
        # 检查是否有可视化数据（支持两种格式）
        has_visualization = False
        if iteration_idx < len(iteration_history):
            iteration_data = iteration_history[iteration_idx]
            if "visualizations" in iteration_data:
                has_visualization = True
            elif "visualization" in iteration_data:
                has_visualization = True
        
        if has_visualization:
            add_modern_visualization_to_html(html_lines, iteration_history[iteration_idx], iteration_idx)
            print("added visualization")
        
        html_lines.extend([
            "          </div>",
            "        </div>",
        ])
    
    html_lines.append("      </section>")

def add_conclusion_section(html_lines: List[str], summary: ReportSummary) -> None:
    """Add the Conclusion section"""
    html_lines.extend([
        "        <!-- Conclusion Section -->",
        "        <section class='article' id='conclusion'>",
        "          <div class='conclusion'>",
        "            <h3>🎯 Conclusion</h3>",
        f"            <p>{summary.conclusion}</p>",
        "          </div>",
        "        </section>",
    ])

def get_iteration_title(iteration_num: int, summary: ReportSummary) -> str:
    """Generate iteration titles"""
    return summary.iterations[iteration_num-1].title

def add_modern_visualization_to_html(html_lines: List[str], iteration: Dict, iteration_num: int) -> None:
    """Add visualization with modern styling"""
    # 支持两种格式：visualizations（复数）和visualization（单数）
    viz = None
    
    if "visualizations" in iteration:
        visualizations = iteration["visualizations"]
        if isinstance(visualizations, list) and len(visualizations) > 0:
            viz = visualizations[0]
        elif isinstance(visualizations, dict) and "visualizations" in visualizations:
            viz_list = visualizations["visualizations"]
            if isinstance(viz_list, list) and len(viz_list) > 0:
                viz = viz_list[0]
    
    elif "visualization" in iteration:
        viz = iteration["visualization"]
    
    if viz is None:
        return
    
    html_lines.extend([
        "            <div class='viz-section'>",
        # "              <h4 class='viz-title'>📊 Visualization</h4>",
        "              <div class='chart-container'>",
        # f"                <div class='chart-title'>Chart for Iteration {iteration_num}</div>",
    ])
    
    if viz.get("figure_object") and viz["figure_object"]:
        try:
            figure_html = viz["figure_object"]
            
            # 调试：检查 figure_html 内容
            print(f"🔍 Figure HTML type: {type(figure_html)}")
            print(f"🔍 Figure HTML length: {len(str(figure_html))}")
            print(f"🔍 Figure HTML preview: {str(figure_html)[:200]}...")
            
            # 检查是否包含 'vis' 引用
            if 'vis' in str(figure_html):
                print("⚠️ Warning: figure_html contains 'vis' references")
            
            html_lines.append(figure_html)
            print(f"✅ Added embedded HTML chart for iteration {iteration_num}")
            
        except Exception as e:
            print(f"❌ Error type: {type(e)}")
            print(f"❌ Error embedding HTML chart for iteration {iteration_num}: {e}")
            html_lines.append(f"                <div class='error'>Error embedding chart: {str(e)}</div>")
    else:
        print(f"❌ No visualization data found for iteration {iteration_num}")
        
    html_lines.extend([
        # "              </div>",
        "            </div>",
    ])

def get_javascript_code() -> str:
    """Return JavaScript code for smooth scrolling and modal functionality"""
    return """
  <script>
    // Simple modal functions - ensure they work
    function showContactModal() {
      console.log('showContactModal called');
      const modal = document.getElementById('contactModal');
      if (modal) {
        modal.style.display = 'block';
        console.log('Contact modal displayed');
      } else {
        console.error('Contact modal not found');
        alert('Modal not found');
      }
    }

    function showQuestionModal() {
      console.log('showQuestionModal called');
      const modal = document.getElementById('questionModal');
      if (modal) {
        modal.style.display = 'block';
        console.log('Question modal displayed');
      } else {
        console.error('Question modal not found');
        alert('Modal not found');
      }
    }

    function closeModals() {
      const contactModal = document.getElementById('contactModal');
      const questionModal = document.getElementById('questionModal');
      
      if (contactModal) contactModal.style.display = 'none';
      if (questionModal) questionModal.style.display = 'none';
      
      console.log('Modals closed');
    }

    function submitDatasetRequest() {
      alert('Thank you for your interest! We\\'ll be in touch soon.');
      closeModals();
    }

    function submitQuestion() {
      alert('Thank you for your question! We\\'ll provide additional insights.');
      closeModals();
    }

    // Close modal when clicking outside
    window.onclick = function(event) {
      const contactModal = document.getElementById('contactModal');
      const questionModal = document.getElementById('questionModal');
      
      if (event.target === contactModal) {
        contactModal.style.display = 'none';
      }
      if (event.target === questionModal) {
        questionModal.style.display = 'none';
      }
    }

    // Add smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        }
      });
    });

    // Verify functions are loaded
    console.log('Modal functions loaded successfully');
    console.log('showContactModal function:', typeof showContactModal);
    console.log('showQuestionModal function:', typeof showQuestionModal);
  </script>"""