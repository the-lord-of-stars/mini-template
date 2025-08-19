# Internal Notes

## Run
Run in this directory:
```bash
source ../mini-template-evn/bin/activate
pip install -r requirements.txt  # if running for the first time
python run.py
```

### Customize Settings
You can customize analysis parameters in the `initialize_state()` method in `agent.py`:

#### Topic and Iteration Configuration
```python
def initialize_state(self) -> dict:
    state = {
        "topic": "evolution of visualization for sensemaking",  # Modify this to set your analysis topic        
        "iteration_count": 0,        # Starting iteration counter (keep as 0)
        "max_iterations": 2,         # Maximum number of iterations (adjust as needed)
        "should_continue": True,     # Continue analysis flag (keep as True)
        "iteration_history": []      # Iteration history (keep as empty list)
    }
    return state
```

#### Parameter Reference

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `topic` | Research topic for analysis | Customize based on your research interest |
| `max_iterations` | Maximum analysis iterations | default for 2 iterations (avoid over-analysis) |
| `iteration_count` | Starting iteration count | Keep as 0 |
| `should_continue` | Analysis continuation flag | Keep as True |
| `iteration_history` | Historical records | Keep as empty list |

#### Important Notes
- Setting `max_iterations` too high may result in lengthy analysis times
- The node `fact` executes code in python environment with a `timeout` limit. Default setting is 120 seconds, however, might not be sufficient. A simple retry mechanism is set up but not enabled. 