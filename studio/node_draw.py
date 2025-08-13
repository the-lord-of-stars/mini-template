import json

import pandas as pd

from studio.helpers import get_llm
from studio.state import State


def react_analysis_node(state: State) -> State:
    llm = get_llm()

    # pick top 5 priority tasks
    tasks = state.get("analysis_tasks")
    tasks = sorted(tasks, key=lambda x: x["priority"], reverse=True)[:5]

    for task in tasks:
        objective = task["objective"]
        description = task["description"]
        time_scope = task["time_scope"]
        target_columns = task["target_columns"]
        for suggested_op in task["suggested_ops"]:
            op_name = suggested_op["name"]
            op_description = suggested_op["description"]




# test
if __name__ == "__main__":
    state = State(messages=[])

    # read ./analysis_tasks.json file and set it to state
    with open("analysis_tasks.json", "r", encoding="utf-8") as f:
        analysis_tasks = json.load(f)
    state.update({"analysis_tasks":analysis_tasks})

    # read ./dataset.csv file and set it to states
    df = pd.read_csv("./dataset.csv", encoding='utf-8')
    state.update({"dataframe":df})

    react_analysis_node(state)