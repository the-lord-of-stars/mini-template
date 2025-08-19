import re
import json
from pathlib import Path
from typing import Dict, Any, List

def markdown_to_html(md: str) -> str:
    """Convert markdown to HTML with basic formatting."""
    if not md:
        return ""
    
    # horizontal rules
    html = re.sub(r'^---\s*$', r'<hr class="my-8 border-gray-300"/>', md, flags=re.MULTILINE)
    # headings
    html = re.sub(r'^###\s*(.+)$', r'<h3 class="text-xl font-semibold text-gray-800 mt-6 mb-3">\1</h3>', html, flags=re.MULTILINE)
    # bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong class="font-semibold">\1</strong>', html)
    # paragraphs
    parts = [p.strip() for p in html.split('\n\n') if p.strip()]
    return '\n'.join(f'<p class="text-gray-700 leading-relaxed mb-4">{p}</p>' for p in parts)

def extract_facts_from_text(text: str) -> List[str]:
    """Extract facts from text that contains ### Begin of facts and ### End of facts markers."""
    if not text:
        return []
    
    facts = []
    pattern = r'### Begin of facts\n(.*?)\n### End of facts'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        fact_lines = [line.strip() for line in match.split('\n') if line.strip()]
        facts.extend(fact_lines)
    
    return facts

def generate_html_report(output_state: dict, output_path: str):
    """
    Builds a beautiful HTML report from the vis_report state data.
    
    Args:
        output_state: State dictionary containing report_outline and other data
        output_path: Path to save the HTML report
    """
    
    # Extract report outline from state
    report_outline = output_state.get("report_outline", [])
    config = output_state.get("config", {})
    
    # Build the HTML document
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>Research Report: Automated Visualization</title>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>",
        "  <script src='https://cdn.tailwindcss.com'></script>",
        "  <script>",
        "    tailwind.config = {",
        "      theme: {",
        "        extend: {",
        "          colors: {",
        "            primary: {",
        "              50: '#fff7ed',",
        "              100: '#ffedd5',",
        "              500: '#f97316',",
        "              600: '#ea580c',",
        "              700: '#c2410c',",
        "            }",
        "          }",
        "        }",
        "      }",
        "    }",
        "  </script>",
        "  <style>",
        "    .card-hover {",
        "      transition: all 0.3s ease;",
        "    }",
        "    .card-hover:hover {",
        "      transform: translateY(-2px);",
        "      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);",
        "    }",
        "    .section-number {",
        "      background: linear-gradient(135deg, #f97316, #c2410c);",
        "      -webkit-background-clip: text;",
        "      -webkit-text-fill-color: transparent;",
        "      background-clip: text;",
        "    }",
        "  </style>",
        "</head>",
        "<body class='bg-gray-50'>",
        
        # Header
        "  <header class='bg-orange-500 text-white py-12'>",
        "    <div class='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>",
        "      <div class='text-center'>",
        f"        <h1 class='text-4xl font-bold mb-4'>{output_state['config']['topic']}</h1>",
        "      </div>",
        "    </div>",
        "  </header>",
        
        # Main content
        "  <main class='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8'>",
    ]
    
    # # Add topic information if available
    # if config.get("topic"):
    #     html_lines.extend([
    #         "    <div class='bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8'>",
    #         "      <h2 class='text-2xl font-semibold text-gray-800 mb-4'>Research Topic</h2>",
    #         f"      <p class='text-gray-700 text-lg'>{config['topic']}</p>",
    #         "    </div>"
    #     ])
    
    # Process each section
    for section in report_outline:
        section_number = section.get("section_number", 0)
        section_name = section.get("section_name", "Untitled Section")
        section_description = section.get("section_description", "")
        content = section.get("content", [])
        
        html_lines.extend([
            f"    <section class='mb-12'>",
            f"      <div class='bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden'>",
            f"        <div class='bg-gradient-to-r from-orange-50 to-orange-100 px-6 py-4 border-b border-gray-200'>",
            f"          <div class='flex items-center space-x-3'>",
            f"            <span class='section-number text-2xl font-bold'>{section_number}</span>",
            f"            <h2 class='text-2xl font-semibold text-gray-800'>{section_name}</h2>",
            f"          </div>",
            # f"          <p class='text-gray-600 mt-2'>{section_description}</p>",
            f"        </div>",
            f"        <div class='p-6'>"
        ])
        
        # Process content items
        for content_item in content:
            content_type = content_item.get("type", "")
            text = content_item.get("text", "")
            
            if content_type == "introduction":
                html_lines.extend([
                    "          <div class='mb-6'>",
                    # "            <h3 class='text-lg font-semibold text-gray-800 mb-3'>Introduction</h3>",
                    f"            <div class='text-gray-700 leading-relaxed'>{markdown_to_html(text)}</div>",
                    "          </div>"
                ])
            
            elif content_type == "visualisation":
                visualisation = content_item.get("visualisation", {})
                facts = content_item.get("facts", "")
                vis_text = content_item.get("text", "")
                
                html_lines.extend([
                    "          <div class='mb-8'>",
                    "            <div class='rounded-lg p-4 mb-4'>",
                    # "              <h3 class='text-lg font-semibold text-gray-800 mb-3'>📊 Visualization</h3>"
                ])
                
                # Add visualization if available
                if visualisation and visualisation.get("specification"):
                    try:
                        spec = json.loads(visualisation["specification"])
                        div_id = f"vis-{section_number}-{content_item.get('id')}"
                        html_lines.extend([
                            f"              <div id='{div_id}' class='mb-4'></div>",
                            "              <script>",
                            f"                vegaEmbed('#{div_id}', {json.dumps(spec)})",
                            "                  .catch(console.error);",
                            "              </script>"
                        ])
                    except json.JSONDecodeError:
                        html_lines.append("              <p class='text-red-600'>Error loading visualization</p>")
                
                # Add visualization description
                if vis_text:
                    html_lines.extend([
                        "              <div class='mt-4'>",
                        # "                <h4 class='text-md font-semibold text-gray-700 mb-2'>Interpretation</h4>",
                        f"                <div class='text-gray-600'>{markdown_to_html(vis_text)}</div>",
                        "              </div>"
                    ])
                
                # # Add facts if available
                # if facts:
                #     extracted_facts = extract_facts_from_text(facts)
                #     if extracted_facts:
                #         html_lines.extend([
                #             "              <div class='mt-4'>",
                #             "                <h4 class='text-md font-semibold text-gray-700 mb-2'>Key Facts</h4>",
                #             "                <div class='bg-orange-50 rounded-lg p-3'>",
                #             "                  <ul class='text-sm text-gray-700 space-y-1'>"
                #         ])
                #         for fact in extracted_facts[:10]:  # Limit to first 10 facts
                #             html_lines.append(f"                    <li class='flex items-start'>")
                #             html_lines.append(f"                      <span class='text-orange-500 mr-2'>•</span>")
                #             html_lines.append(f"                      <span>{fact}</span>")
                #             html_lines.append(f"                    </li>")
                #         html_lines.extend([
                #             "                  </ul>",
                #             "                </div>",
                #             "              </div>"
                #         ])
                
                html_lines.append("            </div>")
                html_lines.append("          </div>")
        
        html_lines.extend([
            "        </div>",
            "      </div>",
            "    </section>"
        ])
    
    # Footer
    html_lines.extend([
        "  </main>",
        "  <footer class='bg-orange-700 text-white py-8 mt-12'>",
        "    <div class='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center'>",
        "      <p class='text-gray-300'>Generated by Automated Visualization Research Analysis</p>",
        "    </div>",
        "  </footer>",
        "</body>",
        "</html>"
    ])
    
    # Write out HTML file
    Path(output_path).write_text("\n".join(html_lines), encoding="utf-8")
    print(f"✓ HTML report generated: {output_path}")

def generate_html_from_state_file(state_file_path: str, output_path: str):
    """
    Generate HTML report from a state JSON file.
    
    Args:
        state_file_path: Path to the state JSON file
        output_path: Path to save the HTML report
    """
    try:
        # Load the state file
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # Generate HTML report
        generate_html_report(state_data, output_path)
        
        print(f"✓ HTML report generated from state file: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: State file not found: {state_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in state file: {e}")
    except Exception as e:
        print(f"Error generating HTML from state file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage
    state_file = "../../../outputs/vis_report/thread_20250819_134745/state.json"
    output_file = "../../../outputs/vis_report/thread_20250819_134745/report.html"
    
    if Path(state_file).exists():
        generate_html_from_state_file(state_file, output_file)
    else:
        print(f"State file not found: {state_file}")
        print("Please provide a valid state file path.")
