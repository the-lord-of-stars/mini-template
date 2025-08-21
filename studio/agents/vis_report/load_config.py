from typing_extensions import TypedDict
import yaml

class Config(TypedDict):
    dataset: str
    topic: str
    target_audience: str
    domain_knowledge: str

config_path = "config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)