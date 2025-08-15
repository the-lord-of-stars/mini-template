from graph import create_workflow
from state import InputState, OutputState, State
import json
import re
from report_html import generate_html_report
from memory import shared_memory
from datetime import datetime

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()
        png_data = self.workflow.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_data)

    # def initialize_state(self, file_path: str, file_url: str, user_query: str) -> InputState:
    def initialize_state(self) -> dict:
        """
        Prepares the initial input state for the workflow.
        """
        # if not file_path:
        #     raise ValueError("File path must be provided to initialize_state_from_csv.")

        # initial_state: InputState = {
        #     "file_path": file_path,
        #     "dataset_url": file_url,
        #     "dataset_info": "",
        #     "user_query": user_query,
        #     "messages": [],
        #     "topic": "evolution of research on sensemaking",
        #     "iteration_count": 0,  # Initialize iteration counter
        #     "max_iterations": 2,  # Set maximum iterations (adjust as needed)
        #     "should_continue": True,  # Initialize to continue
        #     "iteration_history": []  # yuhan: this is not used
        # }
        state = {
            "topic": "evolution of research on sensemaking",
            "iteration_count": 0,  # Initialize iteration counter
            "max_iterations": 2,  # Set maximum iterations (adjust as needed)
            "should_continue": True,  # Initialize to continue
            "iteration_history": []  # yuhan: this is not used
        }
        return state

    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report

        # generate_pdf_report(output, "output.pdf")
        output_path = f"outputs/simple_iteration/{shared_memory.thread_id}/output.html"
        generate_html_report(output, output_path, shared_memory)
        output_path = f"output.html"
        generate_html_report(output, output_path, shared_memory)
        print(f"Visualization report generated: {output_path}")


    def generate_thread_id(self) -> str:
        """Generate a unique thread ID"""
        return f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # def process(self, thread_id: str = None):
    #     if self.workflow is None:
    #         raise RuntimeError("Agent not initialised. Call initialize() first.")
    #
    #     # Generate thread_id if not provided
    #     if thread_id is None:
    #         thread_id = self.generate_thread_id()
    #         shared_memory.set_thread_id(thread_id)
    #         print(f"Generated new thread_id: {thread_id}")
    #
    #     user_query = "Based on the provided research publication record dataset, tell me."
    #     file_path = "./dataset.csv"
    #     file_url = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
    #     print(f"Agent: Starting processing for query: '{user_query}' with file: '{file_path}'")
    #
    #     # initialize the state & read the dataset
    #     # input_state = self.initialize_state(file_path, file_url, user_query)
    #     # initialize the state & save to memory
    #     state = self.initialize_state()
    #     shared_memory.save_state(state)
    #
    #     # invoke the workflow
    #     # output_state: OutputState = self.workflow.invoke(input_state, config={"configurable": {"thread_id": thread_id}})
    #     # invoke the workflow with generated thread_id
    #     try:
    #         output_state = self.workflow.invoke(state, config={"configurable": {"thread_id": thread_id}})
    #     except Exception as e:
    #         # output_state = self.workflow.get_latest_state()
    #         output_state = shared_memory.get_state()
    #     print(output_state)
    #
    #     # print("------Output State-----")
    #     # print(output_state)
    #
    #     # return output_state
    #
    #     # flatten the output
    #     def _flatten(value):
    #         return getattr(value, "content", value)
    #
    #     result = {k: _flatten(v) for k, v in output_state.items()}
    #
    #     # decode the output
    #     self.decode_output(result)
    #
    #     return result
    def process(self, thread_id: str = None, use_workflow_synthesise: bool = False):
        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")

        # Generate thread_id if not provided
        if thread_id is None:
            thread_id = self.generate_thread_id()
            shared_memory.set_thread_id(thread_id)
            print(f"Generated new thread_id: {thread_id}")

        # initialize the state & save to memory
        input_state = self.initialize_state()
        shared_memory.save_state(input_state)

        # invoke the workflow with generated thread_id
        try:
            output_state = self.workflow.invoke(input_state, config={"configurable": {"thread_id": thread_id}})
        except Exception as e:
            # output_state = self.workflow.get_latest_state()
            output_state = shared_memory.get_state()
        
        # Save the final state to memory
        shared_memory.save_state(output_state)
        
        facts = output_state['facts']
        print(output_state)

        # flatten the output
        def _flatten(value):
            return getattr(value, "content", value)

        result = {k: _flatten(v) for k, v in output_state.items()}

        # decode the output (for backward compatibility)
        if not use_workflow_synthesise:
            self.decode_output(result)

        # return the result
        return result
