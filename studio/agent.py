import pandas as pd

from process.output.report_html import generate_html_report
from studio.components.workflow import create_workflow
from studio.process.preprocess.inspect_data import output_preprocess


class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()

    def initialize_state_from_csv(self) -> dict:
        # The dataset should be first input to the agentic configuration, and it should be generalizable to any dataset
        path = "./dataset.csv"
        df = pd.read_csv(path)
        state = output_preprocess(df)
        return state

    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report

        # generate_pdf_report(output, "output.pdf")
        generate_html_report(output, "output.html")

    def process(self):
        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        state = self.initialize_state_from_csv()

        # invoke the workflow
        output_state = self.workflow.invoke(state)
        print(output_state)

        # flatten the output
        def _flatten(value):
            return getattr(value, "content", value)
        result = {k: _flatten(v) for k, v in output_state.items()}

        # decode the output
        self.decode_output(result)

        # return the result
        return result