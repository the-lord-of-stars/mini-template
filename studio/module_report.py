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
    output_filename = state.get("report_filename", "analysis_report.html")
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


# 如果你需要更灵活的报告生成，可以添加配置选项
def report_generation_node_configurable(state: dict) -> dict:
    """
    Configurable report generation node
    """
    # 可配置的报告选项
    report_config = state.get("report_config", {})

    include_facts = report_config.get("include_facts", True)
    include_insights = report_config.get("include_insights", True)
    report_title = report_config.get("title", "Data Analysis Report")
    output_filename = report_config.get("filename", "analysis_report.html")

    # 其余逻辑类似...
    return report_generation_node(state)




# import json
# import re
# from helpers import get_llm
# from langchain_core.messages import HumanMessage, SystemMessage
#
#
# class ImprovedAgentReport:
#     def __init__(self):
#         self.llm = get_llm(temperature=0.3)
#
#     def generate_narrative(self, output_state):
#         prompt = self._build_prompt(output_state)
#
#         response = self.llm.invoke([
#             SystemMessage(
#                 content="You are a professional data analysis report writer. Generate CLEAN HTML content. Do NOT use markdown code blocks or backticks."),
#             HumanMessage(content=prompt)
#         ])
#
#         narrative = response.content if hasattr(response, 'content') else str(response)
#
#         # 清理HTML：移除任何markdown标记
#         narrative = re.sub(r'```html\s*', '', narrative)
#         narrative = re.sub(r'```\s*', '', narrative)
#         narrative = re.sub(r'`+', '', narrative)  # 移除所有backticks
#
#         return narrative
#
#     def _build_prompt(self, output_state):
#         tasks = output_state.get('analysis_tasks', [])
#         analysis_result = output_state.get('analysis_result', {})
#
#         tasks_text = "\n".join(f"- {task.get('objective', 'No description available')}" for task in tasks)
#         summary = analysis_result.get('analysis_summary', 'No summary provided.')
#         insights = analysis_result.get('insights', [])
#         insights_text = "\n".join(f"• {insight}" for insight in insights) if insights else "No key insights provided."
#
#         prompt = f"""
# Generate a professional data analysis report using CLEAN HTML tags only.
#
# **Analysis Tasks Completed:**
# {tasks_text}
#
# **Analysis Summary:**
# {summary}
#
# **Key Insights:**
# {insights_text}
#
# **STRICT Requirements:**
# 1. Use ONLY clean HTML tags: <h2>, <h3>, <p>, <ul>, <li>, <strong>
# 2. NO markdown syntax, NO backticks, NO code blocks
# 3. Start with <h2>Executive Summary</h2>
# 4. Include sections: Executive Summary, Analysis Overview, Key Findings, Conclusions
# 5. Write in professional, clear English
# 6. Focus on IEEE VIS publication analysis insights
#
# Generate clean HTML content suitable for direct embedding in a webpage.
# """
#         return prompt
#
#
# def improve_vega_lite_visualization(vega_spec):
#     """改进Vega-Lite可视化设计"""
#
#     if not vega_spec or '$schema' not in vega_spec:
#         return vega_spec
#
#     # 检查是否是hconcat格式
#     if 'hconcat' in vega_spec:
#         # 改进多图表布局
#         improved_spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "title": {
#                 "text": "IEEE VIS Publication Trends Analysis",
#                 "fontSize": 16,
#                 "anchor": "start"
#             },
#             "data": vega_spec["data"],  # 保持原有数据源
#             "transform": vega_spec.get("transform", []),
#             "vconcat": [  # 改为垂直布局，更易阅读
#                 {
#                     "title": "Publication Count Over Time",
#                     "mark": {"type": "line", "point": True, "strokeWidth": 3},
#                     "encoding": {
#                         "x": {
#                             "field": "Year",
#                             "type": "temporal",
#                             "title": "Year",
#                             "axis": {"labelAngle": 0}
#                         },
#                         "y": {
#                             "field": "Number_of_Publications",
#                             "type": "quantitative",
#                             "title": "Number of Publications"
#                         },
#                         "color": {"value": "#1f77b4"},
#                         "tooltip": [
#                             {"field": "Year", "type": "temporal", "title": "Year"},
#                             {"field": "Number_of_Publications", "type": "quantitative", "title": "Publications"}
#                         ]
#                     },
#                     "width": 600,
#                     "height": 250
#                 },
#                 {
#                     "title": "Average Citation Trends",
#                     "layer": [
#                         {
#                             "mark": {"type": "line", "point": True, "strokeWidth": 2},
#                             "encoding": {
#                                 "x": {
#                                     "field": "Year",
#                                     "type": "temporal",
#                                     "title": "Year"
#                                 },
#                                 "y": {
#                                     "field": "Avg_AminerCitationCount",
#                                     "type": "quantitative",
#                                     "title": "Average Citation Count"
#                                 },
#                                 "color": {"value": "#ff7f0e"},
#                                 "tooltip": [
#                                     {"field": "Year", "type": "temporal"},
#                                     {"field": "Avg_AminerCitationCount", "type": "quantitative",
#                                      "title": "Avg Aminer Citations"}
#                                 ]
#                             }
#                         },
#                         {
#                             "mark": {"type": "line", "point": True, "strokeWidth": 2, "strokeDash": [5, 5]},
#                             "encoding": {
#                                 "x": {
#                                     "field": "Year",
#                                     "type": "temporal"
#                                 },
#                                 "y": {
#                                     "field": "Avg_CitationCount_CrossRef",
#                                     "type": "quantitative"
#                                 },
#                                 "color": {"value": "#2ca02c"},
#                                 "tooltip": [
#                                     {"field": "Year", "type": "temporal"},
#                                     {"field": "Avg_CitationCount_CrossRef", "type": "quantitative",
#                                      "title": "Avg CrossRef Citations"}
#                                 ]
#                             }
#                         }
#                     ],
#                     "resolve": {"scale": {"y": "independent"}},
#                     "width": 600,
#                     "height": 250
#                 },
#                 {
#                     "title": "Average Downloads Trend",
#                     "mark": {"type": "area", "line": True, "point": True},
#                     "encoding": {
#                         "x": {
#                             "field": "Year",
#                             "type": "temporal",
#                             "title": "Year"
#                         },
#                         "y": {
#                             "field": "Avg_Downloads_Xplore",
#                             "type": "quantitative",
#                             "title": "Average Downloads"
#                         },
#                         "color": {"value": "#9467bd"},
#                         "opacity": {"value": 0.7},
#                         "tooltip": [
#                             {"field": "Year", "type": "temporal"},
#                             {"field": "Avg_Downloads_Xplore", "type": "quantitative", "title": "Avg Downloads"}
#                         ]
#                     },
#                     "width": 600,
#                     "height": 200
#                 }
#             ],
#             "spacing": 20
#         }
#         return improved_spec
#
#     return vega_spec
#
#
# def extract_vega_lite_from_response_improved(analysis_result):
#     """改进的Vega-Lite提取，保持远程URL"""
#
#     # 首先检查已解析的vega_lite_spec
#     vega_spec = analysis_result.get('vega_lite_spec', {})
#     if vega_spec and '$schema' in vega_spec:
#         return improve_vega_lite_visualization(vega_spec)
#
#     # 从full_response中提取
#     full_response = analysis_result.get('full_response', '')
#     if not full_response:
#         return {}
#
#     try:
#         # 查找JSON代码块
#         json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', full_response, re.DOTALL)
#
#         for json_block in json_blocks:
#             try:
#                 parsed_json = json.loads(json_block)
#                 if "$schema" in parsed_json and "vega-lite" in parsed_json.get("$schema", ""):
#                     return improve_vega_lite_visualization(parsed_json)
#             except json.JSONDecodeError:
#                 continue
#
#         # 如果没找到，尝试其他方法...
#         # (保持原有的搜索逻辑)
#
#     except Exception as e:
#         print(f"Error extracting Vega-Lite: {e}")
#
#     return {}
#
#
# def generate_improved_report_html(output_state) -> str:
#     """生成改进的HTML报告"""
#
#     # 生成清洁的叙述内容
#     report_agent = ImprovedAgentReport()
#     narrative = report_agent.generate_narrative(output_state)
#
#     # 提取和改进Vega-Lite规范
#     analysis_result = output_state.get('analysis_result', {})
#     vega_spec = extract_vega_lite_from_response_improved(analysis_result)
#
#     # 检查可视化
#     has_visualization = bool(vega_spec and '$schema' in vega_spec)
#
#     print(f"Visualization available: {has_visualization}")
#     if has_visualization:
#         print(f"Improved Vega-Lite spec keys: {list(vega_spec.keys())}")
#
#     # 生成改进的HTML
#     if has_visualization:
#         visualization_section = f"""
# <div style="margin: 30px 0;">
#     <h2>Data Visualization</h2>
#     <p><em>Interactive charts showing IEEE VIS publication trends, citation patterns, and download statistics over time.</em></p>
#     <div id="vis" style="margin: 20px 0; text-align: center; border: 1px solid #ddd; border-radius: 8px; padding: 20px;"></div>
# </div>
#
# <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
# <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
# <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
#
# <script>
# const spec = {json.dumps(vega_spec, indent=2)};
#
# vegaEmbed('#vis', spec, {{
#     "actions": true,
#     "tooltip": true,
#     "renderer": "svg"
# }}).then(result => {{
#     console.log('✅ Enhanced Vega-Lite visualization loaded successfully');
# }}).catch(error => {{
#     console.error('❌ Vega-Lite error:', error);
#     document.getElementById('vis').innerHTML = '<div style="padding: 20px; color: #dc3545; border: 1px solid #dc3545; border-radius: 4px;"><strong>Visualization Error:</strong> Could not load the chart. Please check the console for details.</div>';
# }});
# </script>
# """
#     else:
#         visualization_section = """
# <div style="margin: 30px 0;">
#     <h2>Visualization</h2>
#     <div style="padding: 20px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404;">
#         <strong>Note:</strong> Visualization could not be generated from the analysis results. Please check the data and analysis configuration.
#     </div>
# </div>
# """
#
#     # 组合完整HTML
#     html_content = f"""
# <div style="max-width: 1200px; margin: 0 auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333;">
#
#     <header style="text-align: center; margin-bottom: 40px; padding: 20px 0; border-bottom: 2px solid #007bff;">
#         <h1 style="color: #007bff; margin: 0; font-size: 2.5em;">IEEE VIS Publication Analysis Report</h1>
#         <p style="color: #6c757d; margin: 10px 0 0 0; font-size: 1.1em;">Comprehensive Analysis of Publication Trends and Impact Metrics</p>
#     </header>
#
#     <main>
#         {narrative}
#
#         {visualization_section}
#     </main>
#
#     <footer style="margin-top: 50px; padding: 20px 0; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d; font-size: 0.9em;">
#         <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
#             <div>
#                 <strong>Analysis Details:</strong>
#                 Task ID: {analysis_result.get('task_id', 'Unknown')} |
#                 Data Points: {analysis_result.get('data_points_used', 'Unknown')}
#             </div>
#             <div>
#                 <em>Generated using autonomous analysis workflow</em>
#             </div>
#         </div>
#     </footer>
#
# </div>
# """
#
#     return html_content
#
#
# # 更新的decode_output函数
# # 修复HTML生成问题
#
# def decode_output_fixed_v2(self, output_state):
#     """修复版本：直接写入HTML文件，避免generate_html_report的包装问题"""
#     try:
#         html_content = generate_improved_report_html(output_state)
#
#         # 生成完整的HTML文档
#         full_html = f"""<!DOCTYPE html>
# <html lang='en'>
# <head>
#   <meta charset='utf-8'>
#   <title>IEEE VIS Publication Analysis Report</title>
#   <script src='https://cdn.jsdelivr.net/npm/vega@5'></script>
#   <script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>
#   <script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>
#   <style>
#     body {{
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         margin: 0;
#         padding: 20px;
#         background-color: #f8f9fa;
#     }}
#     .container {{
#         max-width: 1200px;
#         margin: 0 auto;
#         background-color: white;
#         border-radius: 8px;
#         box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#         overflow: hidden;
#     }}
#   </style>
# </head>
# <body>
#   <div class="container">
#     {html_content}
#   </div>
# </body>
# </html>"""
#
#         # 直接写入文件，不使用generate_html_report
#         with open("output.html", "w", encoding="utf-8") as f:
#             f.write(full_html)
#
#         print("✅ Fixed HTML report generated successfully!")
#         print("📊 Check output.html for the working visualization")
#
#     except Exception as e:
#         print(f"❌ Error generating fixed report: {e}")
#
#         # 简单fallback
#         fallback_html = f"""<!DOCTYPE html>
# <html lang='en'>
# <head>
#   <meta charset='utf-8'>
#   <title>Error Report</title>
# </head>
# <body>
#   <div style="max-width: 800px; margin: 50px auto; padding: 20px; font-family: sans-serif;">
#     <h1 style="color: #dc3545;">Report Generation Error</h1>
#     <p><strong>Error:</strong> {str(e)}</p>
#     <h2>Debug Information:</h2>
#     <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto;">
# {json.dumps(output_state.get('analysis_result', {}), indent=2)}
#     </pre>
#   </div>
# </body>
# </html>"""
#
#         with open("output.html", "w", encoding="utf-8") as f:
#             f.write(fallback_html)
#
#
# # 或者，如果你想继续使用generate_html_report，需要修复它的调用方式
# def decode_output_fixed_v3(self, output_state):
#     """使用generate_html_report但避免内容被包装的版本"""
#     try:
#         # 生成纯HTML内容（不包含DOCTYPE等）
#         content_html = generate_improved_report_html(output_state)
#
#         # 检查generate_html_report函数的实现
#         # 如果它会自动包装内容，我们需要传递特殊格式
#
#         # 尝试传递原始HTML
#         from report_html import generate_html_report
#
#         # 方法1：尝试传递原始HTML（不包装在message中）
#         generate_html_report(content_html, "output.html")
#
#         print("✅ HTML report generated using fixed method!")
#
#     except Exception as e:
#         print(f"Method 1 failed: {e}")
#
#         try:
#             # 方法2：如果上面失败，尝试不同的调用方式
#             from report_html import generate_html_report
#
#             # 可能需要特殊的格式标记来避免包装
#             special_content = {
#                 "message": content_html,
#                 "raw_html": True  # 如果支持这个标志
#             }
#
#             generate_html_report(special_content, "output.html")
#             print("✅ HTML report generated using method 2!")
#
#         except Exception as e2:
#             print(f"Method 2 also failed: {e2}")
#
#             # 回退到直接文件写入
#             decode_output_fixed_v2(self, output_state)
#
#
# # 推荐：检查你的generate_html_report函数
# def inspect_generate_html_report():
#     """检查generate_html_report函数的实现"""
#     try:
#         from report_html import generate_html_report
#         import inspect
#
#         # 打印函数源码
#         source = inspect.getsource(generate_html_report)
#         print("=== generate_html_report source code ===")
#         print(source)
#         print("=========================================")
#
#     except Exception as e:
#         print(f"Could not inspect generate_html_report: {e}")
#
#
# # 简化的测试方案
# def test_html_generation(output_state):
#     """测试HTML生成"""
#
#     print("Testing HTML generation...")
#
#     # 生成内容
#     content = generate_improved_report_html(output_state)
#
#     print(f"Generated content length: {len(content)}")
#     print("First 200 characters:")
#     print(content[:200])
#     print("...")
#     print("Last 200 characters:")
#     print(content[-200:])
#
#     # 检查是否包含script标签
#     if '<script>' in content:
#         print("✅ Contains script tags")
#     else:
#         print("❌ No script tags found")
#
#     # 检查是否包含vega相关内容
#     if 'vegaEmbed' in content:
#         print("✅ Contains vegaEmbed call")
#     else:
#         print("❌ No vegaEmbed call found")