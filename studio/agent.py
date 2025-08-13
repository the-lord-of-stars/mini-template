from module_report import decode_output_fixed_v2
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

    def initialize_state_from_csv(self, file_path: str, file_url: str, user_query: str) -> InputState:
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

    def decode_output(self, output_state):
        # html_content = generate_report_html(output_state)
        if 'analysis_result' in output_state:
            spec = extract_vega_lite_spec_from_analysis_result(output_state['analysis_result'])
            if spec:
                output_state['analysis_result']['vega_lite_spec'] = spec
        decode_output_fixed_v2(self, output_state)


    def process(self):
        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")

        user_query = "Based on the provided research publication record dataset, what are the most meaningful analysis tasks that can be performed? Consider trends, topics, and authors."
        file_path = "./dataset.csv"
        file_url = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
        print(f"Agent: Starting processing for query: '{user_query}' with file: '{file_path}'")

        # initialize the state & read the dataset
        input_state = self.initialize_state_from_csv(file_path, file_url, user_query)

        # invoke the workflow
        output_state: OutputState = self.workflow.invoke(input_state)

        # print("------Output State-----")
        # print(output_state)

        # decode the output
        self.decode_output(output_state)

        return output_state

def extract_vega_lite_spec_from_analysis_result(analysis_result):
    messages = analysis_result.get('messages', [])
    for msg in messages:
        content = getattr(msg, 'content', '')
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            json_text = match.group(1)
            try:
                spec = json.loads(json_text)
                return spec
            except json.JSONDecodeError:
                print("Failed to decode Vega-Lite JSON spec")
    return None
