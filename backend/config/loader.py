import yaml
from pathlib import Path
from typing import Union

def load_yaml_config(file_path: Union[str, Path])-> dict:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)





