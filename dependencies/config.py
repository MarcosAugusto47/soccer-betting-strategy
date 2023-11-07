from dataclasses import dataclass
import yaml


@dataclass
class Config:
    metadata_path: str
    odds_path: str
    strategy_result_path: str
    
    max_vector_length: int


def load_config(path: str) -> Config:
    with open(path) as config_file:
        return Config(**yaml.safe_load(config_file))
