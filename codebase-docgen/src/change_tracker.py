import difflib

def show_changes(original, updated):
    diff = difflib.unified_diff(original.splitlines(), updated.splitlines(), lineterm="")
    return "\n".join(diff)