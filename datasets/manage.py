import os
import json

from typing import List

DATASETS_DIR = 'datasets'
EXPT_DIR = 'Expt1'
EXCLUDE_DIR = ['__pycache__']

def get_subject_names() -> List[str]:
    DATA_DIR = os.path.join(DATASETS_DIR, EXPT_DIR)
    print(DATA_DIR)
    return list(filter(lambda x: os.path.isdir(os.path.join(DATA_DIR, x)) and x not in EXCLUDE_DIR, os.listdir(DATA_DIR)))

def get_datapack_names(subject_name) -> List[str]:
    subject_dir = os.path.join(DATASETS_DIR, os.path.join(EXPT_DIR, subject_name))
    return list(map(lambda x: os.path.join(subject_dir, x), [x for x in os.listdir(subject_dir) if not x.startswith('.')]))
    
def load_datapack(datapack_name: str) -> dict:
    with open(datapack_name, 'r', encoding='utf-8') as f:
        return json.load(f)

def set_expt(expt_name: str) -> None:
    global EXPT_DIR
    EXPT_DIR = expt_name