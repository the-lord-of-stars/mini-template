from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from datetime import datetime

from agents.vis_report.memory import memory

from agents.vis_report.planner.state import State
from agents.vis_report.planner.agent import create_workflow as create_planner_workflow
from agents.vis_report.planner.agent import Agent as PlannerAgent


def create_workflow():
    return create_planner_workflow()


class Agent(PlannerAgent):
    def __init__(self) -> None:
        super().__init__()
        self.workflow = create_workflow()
