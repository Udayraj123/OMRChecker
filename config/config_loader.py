import yaml
import os

def load_config(config_path="config/default_config.yaml"):
    """
    Load YAML config file and return it as a dictionary.
    Raises an error if the config file is missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)
