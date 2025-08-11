import re
import json
from pathlib import Path

def markdown_to_html(md: str) -> str:
    """Very-lightweight markdown → HTML for our narratives."""
    # horizontal rules
    html = re.sub(r'^---\s*$', r'<hr/>', md, flags=re.MULTILINE)
    # headings
    html = re.sub(r'^###\s*(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    # bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    # paragraphs
    parts = [p.strip() for p in html.split('\n\n') if p.strip()]
    return '\n'.join(f'<p>{p}</p>' for p in parts)

def generate_html_report(output_state: dict, output_path: str):
    """
    Builds an interactive HTML report from output_state["message"],
    treating each ```json ...``` block as a Vega-Lite spec,
    interleaving it with narrative converted from simple Markdown.
    """
    content = output_state["message"]
    fence_re = re.compile(r'```json\s*\n(.*?)```', re.DOTALL)
    parts = []
    last_end = 0

    # 1. Split into narrative vs. JSON spec blocks
    for m in fence_re.finditer(content):
        narrative = content[last_end:m.start()].strip()
        if narrative:
            parts.append(("html", markdown_to_html(narrative)))
        json_code = m.group(1).strip()
        try:
            spec = json.loads(json_code)
            parts.append(("vega", spec))
        except json.JSONDecodeError:
            # fallback: show code block
            parts.append(("html", f'<pre><code>{json_code}</code></pre>'))
        last_end = m.end()

    # tail narrative
    tail = content[last_end:].strip()
    if tail:
        parts.append(("html", markdown_to_html(tail)))

    # 2. Build the HTML document
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Interactive Vega-Lite Report</title>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>",
        "  <style>",
        "    body {",
        "      font-family: 'Segoe UI', sans-serif;",
        "      background: #f8f9fa;",
        "      width: 90%;",
        "      max-width: 960px;",
        "      margin: 2rem auto;",
        "      padding: 1rem;",
        "      color: #333;",
        "    }",
        "    h1 {",
        "      text-align: center;",
        "      border-bottom: 3px solid #003366;",
        "      padding-bottom: 0.5rem;",
        "      color: #003366;",
        "      font-size: 2rem;",
        "    }",
        "    h3 {",
        "      color: #1a5276;",
        "      margin-top: 2rem;",
        "      font-size: 1.5rem;",
        "    }",
        "    .block {",
        "      background: white;",
        "      padding: 1rem 1.5rem;",
        "      border-radius: 0.5rem;",
        "      box-shadow: 0 2px 6px rgba(0,0,0,0.1);",
        "      margin-bottom: 2rem;",
        "    }",
        "    .vega-embed {",
        "      width: 100%;",
        "    }",
        "   .chart-wrapper {",
        "      display: flex;",
        "      justify-content: center;",
        "    }",
        "    .footer {",
        "      text-align: center;",
        "      font-size: 0.875rem;",
        "      color: #888;",
        "      margin-top: 3rem;",
        "    }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Auto-generated Visual Report</h1>",
    ]

    # 3. Interleave narrative and charts
    vis_counter = 0
    for kind, data in parts:
        if kind == "html":
            html_lines.append(data)
        else:  # vega spec
            div_id = f"vis{vis_counter}"
            html_lines.append(f"  <div id='{div_id}' class='vega-embed'></div>")
            spec_json = json.dumps(data)
            html_lines.extend([
                "  <script>",
                f"    vegaEmbed('#{div_id}', {spec_json})",
                "      .catch(console.error);",
                "  </script>",
            ])
            vis_counter += 1


    html_lines.extend([
        "</body>",
        "</html>"
    ])

    # 4. Write out in html file
    Path(output_path).write_text("\n".join(html_lines), encoding="utf-8")
