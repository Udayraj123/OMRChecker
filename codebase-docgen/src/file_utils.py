import os

def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, "w") as file:
        file.write(content)

def save_changes(file_name, changes):
    os.makedirs("saved_changes", exist_ok=True)
    with open(f"saved_changes/{file_name}_changes.txt", "w") as file:
        file.write(changes)