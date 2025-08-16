import os

def report_generation_node(state: dict) -> dict:
    """
    Generate HTML report from analysis results
    """
    print("Generating HTML report...")

    # Get artifacts from state
    artifacts = state.get("artifacts", [])

    if not artifacts:
        print("No artifacts found, skipping report generation")
        return state

    # Generate HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Data Analysis Report</title>
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
            }
            .fact-box {
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
            }
            .insight-box {
                background-color: #f0f8e8;
                border-left: 4px solid #28a745;
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
                color: #28a745;
                margin-bottom: 8px;
            }
            h1 { 
                text-align: center; 
                color: #333;
            }
            .summary-section {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <h1>Data Analysis Report</h1>

        <div class="summary-section">
            <h2>Analysis Summary</h2>
            <p><strong>Total Tasks:</strong> {len(artifacts)}</p>
            <p><strong>Total Figures:</strong> {sum(len(artifact.get('figures', [])) for artifact in artifacts)}</p>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """

    # Add task sections
    for i, artifact in enumerate(artifacts):
        task_num = i + 1
        objective = artifact.get("objective", f"Task {task_num}")
        ops = artifact.get("ops", [])
        figures = artifact.get("figures", [])

        html_content += f"""
        <div class="task-section">
            <div class="task-title">Task {task_num}: {objective}</div>
            <div class="task-info">
                <strong>Operations:</strong> {', '.join(ops)}<br>
                <strong>Number of figures generated:</strong> {len(figures)}
            </div>
        """

        # Add charts for this task
        for j, fig_data in enumerate(figures):
            figure = fig_data["figure"]
            op = fig_data["op"]
            fact = fig_data.get("fact", "No facts available")
            insight = fig_data.get("insight", "No insights available")

            div_id = f"chart_{task_num}_{j + 1}"

            html_content += f"""
            <div class="chart-container">
                <div class="chart-title">Figure: {op}</div>
                <div id="{div_id}"></div>
                <div class="fact-box">
                    <div class="fact-label">📊 Facts:</div>
                    {fact}
                </div>
                <div class="insight-box">
                    <div class="insight-label">💡 Insights:</div>
                    {insight}
                </div>
            </div>
            """

        html_content += "</div>"  # End task-section

    # Add JavaScript for plotting
    html_content += """
    <script>
    """

    # Generate Plotly JavaScript for each chart
    for i, artifact in enumerate(artifacts):
        figures = artifact.get("figures", [])
        task_num = i + 1

        for j, fig_data in enumerate(figures):
            figure = fig_data["figure"]
            div_id = f"chart_{task_num}_{j + 1}"

            # Convert figure to JSON
            fig_json = figure.to_json()

            html_content += f"""
            Plotly.newPlot('{div_id}', {fig_json});
            """

    html_content += """
    </script>
    </body>
    </html>
    """

    # Save HTML file
    output_filename = state.get("report_filename", "output.html")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Update state with report info
    state["report_generated"] = True
    state["report_filename"] = output_filename
    state["report_path"] = os.path.abspath(output_filename)

    print(f"HTML report saved: {output_filename}")
    print(f"Report contains {len(artifacts)} analysis tasks")
    print(f"Total figures: {sum(len(artifact.get('figures', [])) for artifact in artifacts)}")

    return state
