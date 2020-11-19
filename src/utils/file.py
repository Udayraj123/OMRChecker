# docstring, snake_case
"""
 OMRChecker
 Designed and Developed by-
 Udayraj Deshmukh
 https://github.com/Udayraj123
"""
import json


def loadJson(path, **rest):
    with open(path, "r") as f:
        loaded = json.load(f, **rest)
    return loaded
