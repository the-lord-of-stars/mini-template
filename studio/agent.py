# from module_report import decode_output_fixed_v2
from graph import create_workflow
from state import InputState, OutputState
import json
import re

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()
        png_data = self.workflow.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_data)

    def initialize_state(self, file_path: str, file_url: str, user_query: str) -> InputState:
        """
        Prepares the initial input state for the workflow.
        """
        if not file_path:
            raise ValueError("File path must be provided to initialize_state_from_csv.")

        initial_state: InputState = {
            "file_path": file_path,
            "dataset_url": file_url,
            "dataset_info": "",
            "user_query": user_query,
            "messages": []
        }
        return initial_state

    # def decode_output(self, output_state):
    #     # html_content = generate_report_html(output_state)
    #     if 'analysis_result' in output_state:
    #         spec = extract_vega_lite_spec_from_analysis_result(output_state['analysis_result'])
    #         if spec:
    #             output_state['analysis_result']['vega_lite_spec'] = spec
    #     decode_output_fixed_v2(self, output_state)

    def process(self):
        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")

        user_query = "The provided dataset is IEEE VIS publication record from 1990 till now (2024). Based on this dataset, analyse the development and collaboration in this domain in recent 30 years."
        file_path = "./dataset.csv"
        file_url = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
        print(f"Agent: Starting processing for query: '{user_query}' with file: '{file_path}'")

        # initialize the state & read the dataset
        input_state = self.initialize_state(file_path, file_url, user_query)

        # invoke the workflow
        output_state: OutputState = self.workflow.invoke(input_state)

        # print("------Output State-----")
        # print(output_state)

        # decode the output
        self.decode_output(output_state)

        return output_state

    def decode_output(self, output_state):
        """
        Generate comprehensive HTML report from analysis results
        Handles multiple analysis types: publication trends + author collaboration
        TODO: need to update to an ideal version. The current version is llm-generated for a quick test.
        """
        try:
            print("Generating comprehensive HTML report...")

            # Extract analysis results
            analysis_result = output_state.get('analysis_result', {})
            if not analysis_result or 'analyses' not in analysis_result:
                print("No analysis results found, generating empty report")
                self._generate_empty_report()
                return

            analyses = analysis_result['analyses']
            execution_summary = analysis_result.get('execution_summary', {})
            domain = analysis_result.get('domain', 'Research Analysis')

            print(f"Found {len(analyses)} completed analyses: {list(analyses.keys())}")

            # Generate HTML content
            html_content = self._generate_comprehensive_html_report(
                analyses=analyses,
                execution_summary=execution_summary,
                domain=domain,
                output_state=output_state
            )

            # Save HTML file
            with open('output.html', 'w', encoding='utf-8') as f:
                f.write(html_content)

            print("✅ Comprehensive HTML report saved as 'output.html'")

        except Exception as e:
            print(f"❌ Error generating HTML report: {str(e)}")
            self._generate_error_report(str(e))

    def _generate_comprehensive_html_report(self, analyses, execution_summary, domain, output_state):
        """Generate comprehensive HTML report with all analysis types"""

        # HTML template start
        html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analysis Report - {domain}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 1.1em;
            }}
            .content {{
                padding: 30px;
            }}
            .analysis-section {{
                margin-bottom: 40px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                overflow: hidden;
            }}
            .section-header {{
                background: #f8f9fa;
                padding: 20px;
                border-bottom: 1px solid #e0e0e0;
            }}
            .section-header h2 {{
                margin: 0;
                color: #333;
                font-size: 1.8em;
            }}
            .section-content {{
                padding: 25px;
            }}
            .visualization-container {{
                margin: 20px 0;
                padding: 20px;
                background: #fafafa;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }}
            .base64-image {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .interactive-chart {{
                width: 100%;
                min-height: 400px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .narrative {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
                margin: 20px 0;
            }}
            .insights {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .insight-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .insight-card h4 {{
                margin: 0 0 10px 0;
                color: #667eea;
            }}
            .insight-card p {{
                margin: 0;
                color: #666;
            }}
            .code-section {{
                background: #f8f8f8;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .code-header {{
                background: #e8e8e8;
                padding: 10px 15px;
                border-bottom: 1px solid #d0d0d0;
                font-weight: bold;
                color: #333;
            }}
            .code-content {{
                padding: 15px;
                background: #f8f8f8;
                overflow-x: auto;
            }}
            .code-content pre {{
                margin: 0;
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }}
            .summary {{
                background: #e3f2fd;
                padding: 20px;
                border-radius: 8px;
                margin: 30px 0;
            }}
            .summary h3 {{
                margin: 0 0 15px 0;
                color: #1976d2;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            .summary-item {{
                background: white;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }}
            .summary-item .number {{
                font-size: 2em;
                font-weight: bold;
                color: #1976d2;
            }}
            .summary-item .label {{
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Analysis Report</h1>
                <p>{domain} • Generated on {self._get_current_timestamp()}</p>
            </div>

            <div class="content">
    """

        # Executive Summary
        html_content += self._generate_executive_summary(analyses, execution_summary)

        # Publication Trends Analysis Section
        if 'publication_trends' in analyses:
            html_content += self._generate_trends_section(analyses['publication_trends'])

        # Author Collaboration Analysis Section
        if 'author_collaboration' in analyses:
            html_content += self._generate_collaboration_section(analyses['author_collaboration'])

        # Technical Details Section
        html_content += self._generate_technical_section(execution_summary)

        # HTML template end
        html_content += """
            </div>
        </div>
    </body>
    </html>
    """

        return html_content

    def _generate_executive_summary(self, analyses, execution_summary):
        """Generate executive summary section"""

        total_analyses = len(analyses)
        domain = execution_summary.get('domain', 'Research')

        # Calculate summary metrics
        summary_metrics = []

        if 'publication_trends' in analyses:
            trends_data = analyses['publication_trends']['data_insights']
            summary_metrics.append({
                'number': trends_data.get('total_publications', 0),
                'label': 'Total Publications'
            })
            year_range = trends_data.get('year_range', [])
            if year_range:
                summary_metrics.append({
                    'number': f"{year_range[0]}-{year_range[1]}",
                    'label': 'Time Period'
                })

        if 'author_collaboration' in analyses:
            collab_data = analyses['author_collaboration']['data_insights']
            summary_metrics.append({
                'number': collab_data.get('total_authors', 0),
                'label': 'Total Authors'
            })
            summary_metrics.append({
                'number': f"{collab_data.get('collaboration_rate', 0):.1%}",
                'label': 'Collaboration Rate'
            })

        summary_metrics.append({
            'number': total_analyses,
            'label': 'Analyses Completed'
        })

        summary_html = f"""
            <div class="summary">
                <h3>Executive Summary</h3>
                <p>This report presents a comprehensive analysis of {domain} research, examining publication patterns, collaboration networks, and key trends.</p>
                <div class="summary-grid">
    """

        for metric in summary_metrics:
            summary_html += f"""
                    <div class="summary-item">
                        <div class="number">{metric['number']}</div>
                        <div class="label">{metric['label']}</div>
                    </div>
    """

        summary_html += """
                </div>
            </div>
    """

        return summary_html

    def _generate_trends_section(self, trends_analysis):
        """Generate publication trends analysis section"""

        visualization = trends_analysis.get('visualization', {})
        narrative = trends_analysis.get('narrative', '')
        data_insights = trends_analysis.get('data_insights', {})

        section_html = """
            <div class="analysis-section">
                <div class="section-header">
                    <h2>📈 Publication Trends Analysis</h2>
                </div>
                <div class="section-content">
    """

        # Add visualization
        if 'image_base64' in visualization:
            section_html += f"""
                    <div class="visualization-container">
                        <h4>Publication Trends Over Time</h4>
                        <img src="data:image/png;base64,{visualization['image_base64']}" 
                             alt="Publication Trends Chart" class="base64-image">
                    </div>
    """

        # Add narrative
        if narrative:
            section_html += f"""
                    <div class="narrative">
                        <h4>Analysis Insights</h4>
                        <p>{narrative}</p>
                    </div>
    """

        # Add key insights
        if data_insights:
            section_html += """
                    <div class="insights">
    """
            insights = [
                ("Total Publications", data_insights.get('total_publications', 'N/A')),
                ("Year Range",
                 f"{data_insights.get('year_range', ['N/A', 'N/A'])[0]} - {data_insights.get('year_range', ['N/A', 'N/A'])[1]}"),
                ("Peak Year", data_insights.get('peak_year', 'N/A')),
                ("Conferences", data_insights.get('conferences_count', 'N/A'))
            ]

            for title, value in insights:
                section_html += f"""
                        <div class="insight-card">
                            <h4>{title}</h4>
                            <p>{value}</p>
                        </div>
    """

            section_html += """
                    </div>
    """

        # Add reproducible code
        if 'chart_code' in visualization:
            section_html += f"""
                    <div class="code-section">
                        <div class="code-header">📋 Reproducible Code</div>
                        <div class="code-content">
                            <pre>{visualization['chart_code']}</pre>
                        </div>
                    </div>
    """

        section_html += """
                </div>
            </div>
    """

        return section_html

    def _generate_collaboration_section(self, collaboration_analysis):
        """Generate author collaboration analysis section"""

        visualization = collaboration_analysis.get('visualization', {})
        narrative = collaboration_analysis.get('narrative', '')
        data_insights = collaboration_analysis.get('data_insights', {})
        network_stats = collaboration_analysis.get('network_stats', {})

        section_html = """
            <div class="analysis-section">
                <div class="section-header">
                    <h2>🤝 Author Collaboration Analysis</h2>
                </div>
                <div class="section-content">
    """

        # Add interactive visualization
        if 'interactive_html' in visualization:
            section_html += f"""
                    <div class="visualization-container">
                        <h4>Interactive Collaboration Network</h4>
                        <div class="interactive-chart">
                            {visualization['interactive_html']}
                        </div>
                    </div>
    """

        # Add narrative
        if narrative:
            section_html += f"""
                    <div class="narrative">
                        <h4>Collaboration Insights</h4>
                        <p>{narrative}</p>
                    </div>
    """

        # Add key insights
        insights_data = []
        if data_insights:
            insights_data.extend([
                ("Total Authors", data_insights.get('total_authors', 'N/A')),
                ("Collaboration Rate", f"{data_insights.get('collaboration_rate', 0):.1%}"),
                ("Avg Papers/Author", data_insights.get('avg_papers_per_author', 'N/A')),
                ("Strong Collaborations", data_insights.get('strong_collaborations', 'N/A'))
            ])

        if network_stats:
            insights_data.extend([
                ("Network Nodes", network_stats.get('nodes', 'N/A')),
                ("Network Edges", network_stats.get('edges', 'N/A')),
                ("Network Density", f"{network_stats.get('density', 0):.3f}"),
                ("Connected Components", network_stats.get('connected_components', 'N/A'))
            ])

        if insights_data:
            section_html += """
                    <div class="insights">
    """
            for title, value in insights_data:
                section_html += f"""
                        <div class="insight-card">
                            <h4>{title}</h4>
                            <p>{value}</p>
                        </div>
    """
            section_html += """
                    </div>
    """

        # Add reproducible code
        if 'chart_code' in visualization:
            section_html += f"""
                    <div class="code-section">
                        <div class="code-header">📋 Reproducible Code</div>
                        <div class="code-content">
                            <pre>{visualization['chart_code']}</pre>
                        </div>
                    </div>
    """

        section_html += """
                </div>
            </div>
    """

        return section_html

    def _generate_technical_section(self, execution_summary):
        """Generate technical details section"""

        return f"""
            <div class="analysis-section">
                <div class="section-header">
                    <h2>⚙️ Technical Details</h2>
                </div>
                <div class="section-content">
                    <div class="insights">
                        <div class="insight-card">
                            <h4>Analysis Mode</h4>
                            <p>{execution_summary.get('agent_mode', 'Standard Analysis')}</p>
                        </div>
                        <div class="insight-card">
                            <h4>Execution Time</h4>
                            <p>{execution_summary.get('rounds_used', 'N/A')} rounds</p>
                        </div>
                        <div class="insight-card">
                            <h4>Domain</h4>
                            <p>{execution_summary.get('domain', 'General Research')}</p>
                        </div>
                        <div class="insight-card">
                            <h4>Status</h4>
                            <p>✅ Analysis Completed</p>
                        </div>
                    </div>
                </div>
            </div>
    """

    def _generate_empty_report(self):
        """Generate empty report when no analysis results are available"""
        html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analysis Report - No Results</title>
    </head>
    <body>
        <div style="text-align: center; padding: 50px;">
            <h1>No Analysis Results Available</h1>
            <p>The analysis workflow did not produce any results to display.</p>
        </div>
    </body>
    </html>
    """
        with open('output.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_error_report(self, error_message):
        """Generate error report when HTML generation fails"""
        html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analysis Report - Error</title>
    </head>
    <body>
        <div style="text-align: center; padding: 50px;">
            <h1>Report Generation Error</h1>
            <p>An error occurred while generating the analysis report:</p>
            <p style="color: red; font-family: monospace;">{error_message}</p>
        </div>
    </body>
    </html>
    """
        with open('output.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _get_current_timestamp(self):
        """Get current timestamp for report header"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# def extract_vega_lite_spec_from_analysis_result(analysis_result):
#     messages = analysis_result.get('messages', [])
#     for msg in messages:
#         content = getattr(msg, 'content', '')
#         match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
#         if match:
#             json_text = match.group(1)
#             try:
#                 spec = json.loads(json_text)
#                 return spec
#             except json.JSONDecodeError:
#                 print("Failed to decode Vega-Lite JSON spec")
#     return None