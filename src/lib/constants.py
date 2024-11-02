
"""
'constants.py' is used for global variables
"""

import os
from pathlib import Path

base_path = Path(os.getenv("HOME"))
constant_py_file_path = Path(__file__).resolve()

lib_dir_path = constant_py_file_path.parent
src_dir_path = lib_dir_path.parent
ricc_dir_path = src_dir_path.parent

dataset_dir_path = ricc_dir_path / "dataset"
graph_dir_path = dataset_dir_path / "graph"



# Temporary,, don't know what these constants does
buffer = 0.8
sampling_size = 100
num_unlabel = 0

theta = 0.5
weight = 0.01
iteration = 6
threshold = 0


FN_nodes = []
