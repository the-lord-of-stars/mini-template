import plotly.graph_objects as go
import json
import os

def test_plotly_render():
    """Test Plotly chart rendering"""
    
    # Create a simple test chart
    fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2]))
    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=600,
        margin=dict(l=60, r=30, t=60, b=40),
        title="Test Chart"
    )
    
    # Convert to JSON
    fig_json = fig.to_json()
    print(f"Chart JSON length: {len(fig_json)}")
    print(f"Chart JSON preview: {fig_json[:200]}...")
    
    # Create HTML with the chart
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Test Plotly Chart</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .chart-container {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Test Plotly Chart</h1>
        <div class="chart-container">
            <div id="test-chart" style="width: 100%; height: 500px;"></div>
        </div>
        
        <script>
            Plotly.newPlot('test-chart', {fig_json});
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open("test_plotly_chart.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Test HTML file created: test_plotly_chart.html")
    print("📄 Open this file in your browser to see if the chart renders correctly")

if __name__ == "__main__":
    test_plotly_render()
