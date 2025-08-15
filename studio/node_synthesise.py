import os
from typing import Dict, Any
from state import State
from report_html import generate_html_report, figure_to_base64, add_visualization_to_html
from memory import shared_memory
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from helpers import get_llm


def synthesise(state: State) -> State:
    """
    Synthesise node: Generate HTML report from the current state and shared memory
    This node can be integrated into the workflow to generate reports at any point
    """
    
    try:
        # Create output directory if it doesn't exist
        output_dir = f"outputs/simple_iteration/{shared_memory.thread_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report in the thread-specific directory
        # output_path = f"{output_dir}/output.html"
        output_path = f"output.html"
        generate_html_reportv2(state, output_path, shared_memory)
        
        # Also generate a report in the root directory for easy access
        root_output_path = ""
        # generate_html_reportv2(state, root_output_path, shared_memory)
        
        print(f"HTML report generated: {output_path}")
        print(f"HTML report also saved to: {root_output_path}")
        
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


def generate_html_reportv2(output_state: dict, output_path: str, shared_memory) -> str:
    """
    Builds a simple HTML report that displays insights from the state data and shared memory.
    This is a direct copy of generate_html_report from report_html.py
    """
    from pathlib import Path

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

    # Use iteration_history from state if available, otherwise use complete_history
    # iteration_history = state.get("iteration_history", [])
    # if not iteration_history and complete_history:
    iteration_history = complete_history

    # Display iteration history if available
    if iteration_history:
        total_iterations = len(iteration_history)
        total_insights = sum(len(iter.get("insights", [])) for iter in iteration_history)

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

