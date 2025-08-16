import os
from typing import Dict, Any, List, Optional
from state import State
from report_html import generate_html_report, figure_to_base64, add_visualization_to_html
from memory import shared_memory
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from helpers import get_llm
from pathlib import Path

class IterationSummary(BaseModel):
    """summary of one iteration"""
    problem: str = Field(description="the problem this iteration is trying to solve")
    analysis: str = Field(description="summary of the analysis method and approach")
    findings: str = Field(description="key findings and insights from this iteration")
    narrative: str = Field(description="a short, engaging, story-like paragraph combining the problem, analysis, and findings for this iteration")


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
        "1. intro: Set the scene: explain why the topic matters in a broader context.Introduce the problem or observation that triggered the analysis.Briefly state the investigation goal.",
        "2. quick_summary: Tell the overall “story arc” of the analysis in 3–5 sentences.Show how each iteration built upon the previous one (discovery → new question → deeper investigation → conclusions).Include any surprising or notable findings that shaped the direction.",
        "3. iterations: For each iteration:",
        "   - problem: Describe the research question in the context of the story.",
        "   - analysis: Explain the method briefly but connect it to the motivation.",
        "   - findings: Present the key insights, noting whether they confirmed or challenged expectations.",
        "For each iteration, also write 'narrative': a detailed story-like paragraph that naturally integrates the problem, method, and findings. Use smooth transitions and a human, blog-like tone while staying professional."
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
                problem=f"analysis of iteration {i+1}",
                analysis="detailed analysis is unavailable",
                findings="please check the detailed iteration results below",
                narrative="please check the detailed iteration results below"
            )

            iterations.append(iteration)
        
        return ReportSummary(
            intro="analysis summary is currently unavailable.",
            quick_summary="multiple data analysis iterations have been executed.",
            iterations=iterations,
            conclusion="please check the detailed iteration results below to understand the complete analysis."
        )

def add_structured_summary_to_html(html_lines: List[str], summary: ReportSummary, iteration_history: List[Dict]) -> None:
    """add structured LLM summary to HTML"""
    
    # add summary section
    html_lines.extend([
        "<div class='llm-summary-section' style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 25px; border-radius: 12px; margin: 25px 0; border-left: 5px solid #28a745; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>",
        "  <h2 style='color: #28a745; margin-top: 0; font-size: 24px; font-weight: 600; display: flex; align-items: center;'>",
        "    <span style='margin-right: 10px;'>📋</span>execution summary",
        "  </h2>",
        ""
    ])
    
    # intro section
    if summary.intro:
        html_lines.extend([
            "  <div class='summary-intro' style='margin-bottom: 25px; padding: 20px; background-color: white; border-radius: 8px; border-left: 3px solid #17a2b8;'>",
            "    <h3 style='margin-top: 0; color: #17a2b8; font-size: 18px;'>🎯 analysis overview</h3>",
            f"    <p style='line-height: 1.6; margin-bottom: 0; color: #495057;'>{summary.intro}</p>",
            "  </div>"
        ])
    
    # quick summary section
    if summary.quick_summary:
        html_lines.extend([
            "  <div class='quick-summary' style='margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%); border-radius: 8px; border: 1px solid #c3e6cb;'>",
            "    <h3 style='margin-top: 0; color: #155724; font-size: 18px;'>⚡ key points</h3>",
            f"    <p style='line-height: 1.6; margin-bottom: 0; color: #155724; font-weight: 500;'>{summary.quick_summary}</p>",
            "  </div>"
        ])
    
    # iteration summary section
    if summary.iterations:
        html_lines.extend([
            "  <div class='iterations-summary' style='margin-bottom: 25px;'>",
            "    <h3 style='color: #495057; font-size: 20px; margin-bottom: 20px; border-bottom: 2px solid #dee2e6; padding-bottom: 10px;'>",
            "      Main analysis",
            "    </h3>"
        ])
        
        for i, iteration in enumerate(summary.iterations, 1):

            html_lines.extend([
                f"    <div class='iteration-summary' style='margin-bottom: 20px; padding: 20px; background-color: white; border-radius: 8px; border: 1px solid #e9ecef; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>",
                f"      <h4 style='color: #28a745; margin-top: 0; margin-bottom: 15px; font-size: 16px; border-bottom: 1px solid #e9ecef; padding-bottom: 8px;'>",
                f"        Iteration {i}",
                "      </h4>",
            ])

            if hasattr(iteration, "narrative") and iteration.narrative:
                html_lines.extend([
                    f"      <p style='color: #495057; line-height: 1.6; margin-bottom: 15px;'>{iteration.narrative}</p>"
                ])

            html_lines.extend([
                f"      <details style='margin-bottom: 15px;'>",
                f"        <summary style='cursor: pointer; color: #6c757d; font-weight: bold;'>📄 Show details</summary>",
                f"        <div style='margin-top: 10px;'>",
                f"          <div style='margin-bottom: 8px;'><strong style='color: #6c757d;'>🎯 Problem: </strong><span style='color: #495057;'>{iteration.problem}</span></div>",
                f"          <div style='margin-bottom: 8px;'><strong style='color: #6c757d;'>🔍 Analysis Method: </strong><span style='color: #495057;'>{iteration.analysis}</span></div>",
                f"          <div style='margin-bottom: 8px;'><strong style='color: #6c757d;'>💡 Key Findings: </strong><span style='color: #495057;'>{iteration.findings}</span></div>",
                f"        </div>",
                f"      </details>"
            ])

            print(f"iteration {i} visualization")
            # Add visualization if available
            if "visualization" in iteration_history[i-1]:
                print("Existed visualization")
                add_visualization_to_html(html_lines, iteration_history[i-1], i)
                print("Added visualization")
            else:
                print("No visualization")
        
        html_lines.append("  </div>")
    
    # conclusion section
    if summary.conclusion:
        html_lines.extend([
            "  <div class='summary-conclusion' style='margin-bottom: 0; padding: 20px; background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); border-radius: 8px; border: 1px solid #abdde5;'>",
            "    <h3 style='margin-top: 0; color: #0c5460; font-size: 18px;'>🎯 conclusion</h3>",
            f"    <p style='line-height: 1.6; margin-bottom: 0; color: #0c5460; font-weight: 500;'>{summary.conclusion}</p>",
            "  </div>"
        ])
    
    html_lines.append("</div>")
    html_lines.append("<hr style='margin: 30px 0; border: none; border-top: 2px solid #e9ecef;'/>")

def integrate_structured_summary_into_report(html_lines: List[str], iteration_history: List[Dict], llm) -> None:
    """integrate structured LLM summary into HTML report"""
    
    if iteration_history and llm:
        try:
            # generate structured LLM summary
            summary = generate_llm_summary_structured(iteration_history, llm)
            print("-------------Summary double-check-------------------")
            print(summary)
            print("--------------------------------")
            
            # add summary to HTML
            add_structured_summary_to_html(html_lines, summary, iteration_history)
            
        except Exception as e:
            print(f"Warning: Failed to generate structured LLM summary: {e}")
            # add error hint to HTML
            html_lines.extend([
                "<div class='llm-summary-error' style='background-color: #f8d7da; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 5px solid #dc3545;'>",
                "  <h3 style='margin-top: 0; color: #721c24;'>⚠️ note</h3>",
                "  <p style='margin-bottom: 0; color: #721c24;'><strong>structured LLM summary is currently unavailable.</strong> please check the detailed analysis content below.</p>",
                "</div>",
                "<hr style='margin: 30px 0; border: none; border-top: 2px solid #e9ecef;'/>"
            ])

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
        iteration_history = extract_iteration_history(state, shared_memory)
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
        summary = generate_llm_summary_structured(iteration_history, llm)
        
        # Add Overview section
        add_overview_section(html_lines, summary)
        
        # Add Findings section
        add_findings_section(html_lines, summary, iteration_history)
        
        # Add Conclusion section
        add_conclusion_section(html_lines, summary)
    
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

def generate_html_report_debug(output_state: dict, output_path: str, shared_memory) -> str:
    """
    Builds a simple HTML report that displays insights from the state data and shared memory.
    This is a direct copy of generate_html_report from report_html.py
    """

    # 2. Build the HTML document
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Insights Report</title>",
        "  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "  <style>",
        "    body { font-family: sans-serif; margin: 2em; line-height: 1.6; }",
        "    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "    h2 { color: #34495e; margin-top: 2em; }",
        "    .insight { margin-bottom: 1.5em; padding: 1em; background-color: #f8f9fa; border-left: 4px solid #3498db; border-radius: 3px; }",
        "    .insight-number { font-weight: bold; color: #3498db; margin-bottom: 0.5em; }",
        "    .topic { font-size: 1.2em; color: #7f8c8d; margin-bottom: 1em; }",
        "    .question { font-style: italic; color: #95a5a6; margin-bottom: 1em; }",
        "    .iteration { margin-bottom: 2em; padding: 1.5em; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; }",
        "    .iteration h3 { color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "    .facts { margin: 1em 0; }",
        "    .facts pre { background-color: #f1f3f4; padding: 1em; border-radius: 3px; overflow-x: auto; font-size: 0.8em; }",
        "    .facts-output { margin: 1em 0; }",
        "    .facts-output pre { background-color: #e8f5e8; padding: 1em; border-radius: 3px; overflow-x: auto; font-size: 0.8em; }",
        "    .insights { margin: 1em 0; }",
        "    .section-title { font-weight: bold; color: #34495e; margin-bottom: 0.5em; }",
        "    .collapsible { cursor: pointer; }",
        "    .collapsible:hover { background-color: #e9ecef; }",
        "    .collapsible-content { display: none; padding: 1em; background-color: #f8f9fa; border-radius: 3px; margin-top: 0.5em; }",
        "    hr { margin: 2em 0; border: none; border-top: 1px solid #ecf0f1; }",
        "    .chart-container { margin: 20px 0; }",
        "    .chart-title { font-size: 18px; font-weight: bold; color: #555; margin-bottom: 10px; }",
        "    .visualization-section { background-color: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; }",
        "  </style>",
        "  <script>",
        "    function toggleSection(id) {",
        "      var content = document.getElementById(id);",
        "      if (content.style.display === 'none' || content.style.display === '') {",
        "        content.style.display = 'block';",
        "      } else {",
        "        content.style.display = 'none';",
        "      }",
        "    }",
        "  </script>",
        "</head>",
        "<body>",
    ]

    # Add topic information if available
    if "topic" in output_state:
        html_lines.append(f"<h1>Data Insights Report</h1>")
        html_lines.append(f"<div class='topic'><strong>Topic:</strong> {output_state['topic']}</div>")

    html_lines.append("<hr/>")

    # Get complete history from shared memory if available
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

    iteration_history = complete_history

    # Display iteration history if available
    if iteration_history:
        total_iterations = len(iteration_history)
        total_insights = sum(len(iter.get("insights", [])) for iter in iteration_history)

        # integrate structured summary into report
        llm = get_llm(temperature=0.5) 
        integrate_structured_summary_into_report(html_lines, iteration_history, llm)

        # display iteration history
        html_lines.append("<h2>Analysis Iterations</h2>")
        html_lines.append(
            f"<div style='margin-bottom: 1em; padding: 1em; background-color: #e8f4fd; border-radius: 5px;'>")
        html_lines.append(
            f"  <strong>Summary:</strong> {total_iterations} iterations completed, {total_insights} total insights generated")
        html_lines.append("</div>")

        for i, iteration in enumerate(iteration_history, 1):
            html_lines.append(f"<div class='iteration'>")
            html_lines.append(f"  <h3>Iteration {i}</h3>")

            # Display question
            if "question" in iteration and iteration["question"]:
                question_text = iteration["question"].get("question", "No question available")
                html_lines.append(f"  <div class='question'><strong>Question:</strong> {question_text}</div>")

            # Display facts (code and output) - make them collapsible
            if "facts" in iteration and iteration["facts"]:
                facts = iteration["facts"]

                # Analysis Code (collapsible)
                if facts.get("code"):
                    html_lines.append(f"  <div class='facts'>")
                    html_lines.append(
                        f"    <div class='section-title collapsible' onclick='toggleSection(\"code-{i}\")'>📊 Analysis Code (click to expand)</div>")
                    html_lines.append(f"    <div id='code-{i}' class='collapsible-content'>")
                    html_lines.append(f"      <pre><code>{facts['code']}</code></pre>")
                    html_lines.append(f"    </div>")
                    html_lines.append(f"  </div>")

                # Analysis Results (collapsible)
                if facts.get("stdout"):
                    html_lines.append(f"  <div class='facts-output'>")
                    html_lines.append(
                        f"    <div class='section-title collapsible' onclick='toggleSection(\"output-{i}\")'>📈 Analysis Results (click to expand)</div>")
                    html_lines.append(f"    <div id='output-{i}' class='collapsible-content'>")
                    html_lines.append(f"      <pre>{facts['stdout']}</pre>")
                    html_lines.append(f"    </div>")
                    html_lines.append(f"  </div>")
                else:
                    html_lines.append(f"  <div class='facts-output'>")
                    html_lines.append(
                        f"    <div class='section-title collapsible' onclick='toggleSection(\"output-{i}\")'>📈 Analysis Results (click to expand)</div>")
                    html_lines.append(f"    <div id='output-{i}' class='collapsible-content'>")
                    html_lines.append(f"      <pre> None </pre>")
                    html_lines.append(f"      <pre>STDERR {facts['stderr']}</pre>")
                    html_lines.append(f"      <pre>EXIT_CODE {facts['exit_code']}</pre>")
                    html_lines.append(f"    </div>")
                    html_lines.append(f"  </div>")

            # Display vis (code and output) - make them collapsible
            add_visualization_to_html(html_lines, iteration, i)

            # Display insights
            if "insights" in iteration and iteration["insights"]:
                html_lines.append(f"  <div class='insights'>")
                html_lines.append(f"    <div class='section-title'>💡 Key Insights:</div>")
                for j, insight in enumerate(iteration["insights"], 1):
                    html_lines.append(f"    <div class='insight'>")
                    html_lines.append(f"      <div class='insight-number'>Insight {j}</div>")
                    html_lines.append(f"      <div>{insight}</div>")
                    html_lines.append(f"    </div>")
                html_lines.append(f"  </div>")

            html_lines.append("</div>")
            html_lines.append("<hr/>")

    # Display current insights if available (for backward compatibility)
    elif "insights" in output_state and output_state["insights"]:
        html_lines.append("<h2>Key Insights</h2>")

        insights = output_state["insights"]
        for i, insight in enumerate(insights, 1):
            html_lines.append(f"<div class='insight'>")
            html_lines.append(f"  <div class='insight-number'>Insight {i}</div>")
            html_lines.append(f"  <div>{insight}</div>")
            html_lines.append("</div>")
    else:
        html_lines.append("<h2>Key Insights</h2>")
        html_lines.append("<p><em>No insights available in the current state.</em></p>")

    html_lines.extend([
        "</body>",
        "</html>"
    ])

    # 4. Write out html file
    Path(output_path).write_text("\n".join(html_lines), encoding="utf-8")
    
    return output_path
  


def synthesise_final(state: State) -> State:
    """
    Final synthesise node: Generate the final HTML report
    This is typically used at the end of the workflow
    """
    return synthesise(state)


def synthesise_intermediate(state: State) -> State:
    """
    Intermediate synthesise node: Generate HTML report for intermediate results
    This can be used during the workflow to capture intermediate states
    """
    return synthesise(state)

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
        "        ",
        # "        <div class='summary-grid'>",
        # "          <div class='summary-card'>",
        # "            <h4>🎯 Research Evolution</h4>",
        # "            <p>Understanding how research evolved through collaborative networks and emerging methodologies.</p>",
        # "          </div>",
        # "          <div class='summary-card'>",
        # "            <h4>⚡ Key Discovery</h4>",
        # "            <p>Strong partnerships between researchers drove significant diversification in research topics and methodological innovations.</p>",
        # "          </div>",
        # "          <div class='summary-card'>",
        # "            <h4>📈 Impact</h4>",
        # "            <p>The interconnected research community fostered a renaissance of innovation and methodological progress.</p>",
        # "          </div>",
        # "        </div>",
        "      </section>",
    ])

def add_findings_section(html_lines: List[str], summary: ReportSummary, iteration_history: List[Dict]) -> None:
    """Add the Findings section with iteration cards"""
    html_lines.extend([
        "      <!-- Findings Section -->",
        "      <section class='article' id='findings'>",
        "        <h2>Findings</h2>",
    ])
    
    for iteration_idx, iteration in enumerate(summary.iterations, 1):
        html_lines.extend([
            "        <div class='iteration-card'>",
            "          <div class='iteration-header'>",
            f"            <div class='iteration-number'>{iteration_idx:02d}</div>",
            f"            <h3 class='iteration-title'>{get_iteration_title(iteration_idx)}</h3>",
            "          </div>",
            "          <div class='iteration-content'>",
            f"            <p class='iteration-summary'>{iteration.narrative}</p>",
            "",
            "            <details class='details-toggle'>",
            "              <summary>📋 Detailed Analysis</summary>",
            "              <div class='details-content'>",
            "                <div class='detail-item'>",
            "                  <div class='detail-label'>🎯 Research Question</div>",
            f"                  <p>{iteration.problem}</p>",
            "                </div>",
            "                <div class='detail-item'>",
            "                  <div class='detail-label'>🔍 Analysis Method</div>",
            f"                  <p>{iteration.analysis}</p>",
            "                </div>",
            "                <div class='detail-item'>",
            "                  <div class='detail-label'>💡 Key Findings</div>",
            f"                  <p>{iteration.findings}</p>",
            "                </div>",
            "              </div>",
            "            </details>",
        ])

        print(f"adding iteration {iteration_idx} visualization ...")
        
        if iteration_idx-1 < len(iteration_history) and "visualization" in iteration_history[iteration_idx-1]:
            add_modern_visualization_to_html(html_lines, iteration_history[iteration_idx-1], iteration_idx)
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

def get_iteration_title(iteration_num: int) -> str:
    """Generate iteration titles"""
    titles = {
        1: "Author Contributions & Collaborative Networks",
        2: "Collaboration Networks & Topic Diversity", 
        3: "Methodological Innovations & Tools"
    }
    return titles.get(iteration_num, f"Analysis Iteration {iteration_num}")

def add_modern_visualization_to_html(html_lines: List[str], iteration: Dict, iteration_num: int) -> None:
    """Add visualization with modern styling"""
    if "visualization" not in iteration:
        return
        
    viz = iteration["visualization"]
    if not viz:  # 移除 success 检查
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