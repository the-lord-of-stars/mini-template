from helpers import get_llm


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
            "file_url": file_url,
            "file_path": file_path,
            "output_path": output_path
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
        file_url = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
        file_path = "./dataset.csv"
        output_path = f"outputs/{shared_memory.thread_id}/output.html"

        input_state = self.initialize_state(file_url, file_path, output_path)
        shared_memory.save_state(input_state)

        # invoke the workflow with generated thread_id
        try:
            output_state = self.workflow.invoke(input_state, config={"configurable": {"thread_id": thread_id}})
        except Exception as e:
            # output_state = self.workflow.get_latest_state()
            output_state = shared_memory.get_state()
        
        # Save the final state to memory
        shared_memory.save_state(output_state)
    
    return output_state

def create_workflow():

    builder = StateGraph(State)

    # --- Add the nodes ---
   
    # --- Add the edges ---
    
    # Compile the graph
    return builder.compile()
    # return builder.compile(recursion_limit=50)


