import json
import os

def load_json(file_path):
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data