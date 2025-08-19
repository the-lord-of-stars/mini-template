from graph import create_workflow
from state import InputState, OutputState, State
import json
import re
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

    def initialize_state(self) -> dict:
        """
        Prepares the initial input state for the workflow.
        """

        state = {
            # "topic": "who are the researchers in sensemaking research",
            "topic": "research on automated visualization",
            "iteration_count": 0,  # Initialize iteration counter
            "max_iterations": 2,  # Set maximum iterations (adjust as needed) - counting starts from "iteration_count"+1
            "should_continue": True,  # Initialize to continue
            "iteration_history": []  # yuhan: this is not used
        }
        return state


    def generate_thread_id(self) -> str:
        """Generate a unique thread ID"""
        return f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
            print(f"Workflow error: {e}")
            # Get the latest state from memory history
            history = shared_memory.get_history()
            if history:
                output_state = history[-1]
            else:
                output_state = input_state
        
        # Save the final state to memory
        shared_memory.save_state(output_state)

        # return the result
        return output_state
