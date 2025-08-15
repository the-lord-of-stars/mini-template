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
        root_output_path = "{output_dir}/output.html"
        generate_html_reportv2(state, root_output_path, shared_memory)
        
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


def synthesise_v2(state: State) -> State:
    """
    Synthesise node v2: Generate story-telling HTML report using LLM
    This version uses generate_html_reportv2 for more engaging narrative reports
    """
    
    try:
        # Generate report using the new story-telling approach
        output_path = "output_v2.html"
        generate_html_reportv2(state, output_path, shared_memory)
        
        # Also generate the original report for comparison
        original_output_path = "output.html"
        generate_html_report(state, original_output_path, shared_memory)
        
        print(f"Story-telling HTML report generated: {output_path}")
        print(f"Original HTML report also saved to: {original_output_path}")
        
        # Update state with report information
        new_state = state.copy()
        new_state["synthesise"] = {
            "report_generated": True,
            "report_path": output_path,
            "original_report_path": original_output_path,
            "success": True,
            "version": "v2"
        }
        
        return new_state
        
    except Exception as e:
        print(f"Error generating story-telling HTML report: {e}")
        
        # Return state with error information
        new_state = state.copy()
        new_state["synthesise"] = {
            "report_generated": False,
            "error": str(e),
            "success": False,
            "version": "v2"
        }
        
        return new_state


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


# Test function
def test_synthesise():
    """Test the synthesise node"""
    from agent import Agent
    
    # Create an agent and run a simple workflow
    agent = Agent()
    agent.initialize()
    
    # Run the workflow
    result = agent.process()
    
    print("Test completed!")
    print(f"Result keys: {list(result.keys())}")
    
    if "synthesise" in result:
        print(f"Synthesise result: {result['synthesise']}")


def test_generate_html_reportv2():
    """Test the new story-telling report generator with sample data"""
    
    # Create sample state with iteration history
    sample_state = {
        "topic": "Evolution of Research on Sensemaking",
        "iteration_history": [
            {
                "question": {
                    "question": "How has the frequency of papers on sensemaking evolved over the years in the dataset, and what trends can be observed in terms of publication volume and citation counts?"
                },
                "facts": {
                    "stdout": "### Begin of facts\nTotal sensemaking papers: 16\nYears covered: 1999 to 2019\nYear with most sensemaking papers: 2012\nYear with highest average citation count: 2014\n### End of facts"
                },
                "insights": [
                    "The dataset covers a span of 20 years, from 1999 to 2019, during which 16 papers on sensemaking were published. This indicates a consistent interest in the topic over two decades.",
                    "The year 2012 stands out as having the highest number of sensemaking papers published, suggesting a peak in research activity or interest in this area during that time.",
                    "In terms of citation impact, 2014 is notable for having the highest average citation count for sensemaking papers, indicating that the work published around this time was particularly influential or well-received by the academic community."
                ]
            },
            {
                "question": {
                    "question": "What are the most common author affiliations in the sensemaking research papers, and how do these affiliations correlate with the citation impact of the papers?"
                },
                "facts": {
                    "stdout": "### Begin of facts\nMost common author affiliations in sensemaking research:\nAffiliation\nMiddlesex University                     6\nVirginia Tech                            5\nPacific Northwest National Laboratory    4\nSmith College                            4\nSimon Fraser University, Canada          3\nName: count, dtype: int64\n\nAverage citation impact of these affiliations:\nAffiliation\nMiddlesax University                     29.000000\nVirginia Tech                            45.400000\nPacific Northwest National Laboratory    25.000000\nSmith College                            16.000000\nSimon Fraser University, Canada           9.666667\nName: CitationCount, dtype: float64\n### End of facts"
                },
                "insights": [
                    "The most common author affiliations in sensemaking research papers are Middlesex University, Virginia Tech, Pacific Northwest National Laboratory, Smith College, and Simon Fraser University, Canada. This suggests that these institutions are key contributors to the field of sensemaking research.",
                    "Virginia Tech, despite having fewer papers than Middlesex University, has the highest average citation impact among the top affiliations, indicating that its contributions are particularly influential in the field.",
                    "Middlesex University, while having the highest number of papers, has a lower average citation impact compared to Virginia Tech, suggesting that while it is a prolific contributor, its papers may not be as widely cited or influential."
                ]
            },
            {
                "question": {
                    "question": "What is the relationship between the number of downloads on Xplore and the citation impact of sensemaking research papers, and do papers with higher downloads tend to have higher citation counts?"
                },
                "facts": {
                    "stdout": "### Begin of facts\nCorrelation between downloads and citation count: 0.74\n### End of facts"
                },
                "insights": [
                    "The dataset reveals a strong positive correlation (0.74) between the number of downloads on Xplore and the citation count of sensemaking research papers. This suggests that papers which are downloaded more frequently tend to also have higher citation counts, indicating a potential relationship between the accessibility or popularity of a paper and its academic impact.",
                    "The correlation implies that researchers and practitioners may be more likely to cite papers that are easily accessible or widely read, as indicated by the download numbers. This could be due to the increased visibility and dissemination of these papers within the academic community."
                ]
            }
        ],
        "question": {
            "question": "How does the presence of author keywords related to sensemaking influence the download and citation counts of research papers?"
        },
        "facts": {
            "stdout": "### Begin of facts\nCorrelation between downloads and citation count: 0.74\n### End of facts"
        },
        "insights": [
            "The dataset reveals a strong positive correlation (0.74) between the number of downloads on Xplore and the citation count of sensemaking research papers.",
            "This suggests that papers which are downloaded more frequently tend to also have higher citation counts, indicating a potential relationship between the accessibility or popularity of a paper and its academic impact."
        ],
        "visualizations": {
            "visualizations": [
                {
                    "chart_type": "Time trend line chart",
                    "insight": "Generated Time trend line chart showing the evolution of sensemaking research over time"
                },
                {
                    "chart_type": "Grouped Bar Chart", 
                    "insight": "Generated Grouped Bar Chart showing the contribution of different author affiliations"
                },
                {
                    "chart_type": "Scatter Plot",
                    "insight": "Generated Scatter Plot showing the correlation between downloads and citations"
                }
            ]
        }
    }
    
    # Test the new report generator
    try:
        output_path = "test_report_v2.html"
        result_path = generate_html_reportv2(sample_state, output_path, None)
        
        print(f"✅ Test successful! Story-telling report generated: {result_path}")
        print(f"📄 You can open {result_path} in your browser to view the report")
        
        # Also test with real data from the latest run
        print("\n🔍 Testing with real data from latest run...")
        latest_thread = None
        if os.path.exists("outputs/simple_iteration"):
            threads = [d for d in os.listdir("outputs/simple_iteration") if d.startswith("thread_")]
            if threads:
                latest_thread = max(threads, key=lambda x: os.path.getctime(os.path.join("outputs/simple_iteration", x)))
                print(f"📁 Using latest thread: {latest_thread}")
                
                # Load the memory backup
                memory_path = f"outputs/simple_iteration/{latest_thread}/memory_backup.json"
                if os.path.exists(memory_path):
                    import json
                    with open(memory_path, 'r') as f:
                        memory_data = json.load(f)
                    
                    # Get the last state
                    if memory_data.get("states"):
                        last_state = memory_data["states"][-1]
                        real_output_path = "test_real_data_v2.html"
                        real_result_path = generate_html_reportv2(last_state, real_output_path, None)
                        print(f"✅ Real data test successful! Report generated: {real_result_path}")
                        print(f"📄 You can open {real_result_path} in your browser to view the real data report")
                    else:
                        print("❌ No states found in memory backup")
                else:
                    print("❌ Memory backup file not found")
            else:
                print("❌ No thread directories found")
        else:
            print("❌ No outputs directory found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_generate_html_reportv2()
