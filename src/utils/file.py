import json


def load_json(path, **rest):
    with open(path, "r") as f:
        loaded = json.load(f, **rest)
    return loaded
