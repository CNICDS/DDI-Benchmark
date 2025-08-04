import os

from itertools import product
import subprocess

dataset_path = "/home/wuxiao/llm-ddi/ddi/DESC_MOL-DDIE/datasets/drugbank2025"
dataset_split_type = ["S0", "S1", "S2"]
seeds = [1, 12, 123]

for dataset in product(dataset_split_type, seeds):
    dataset_split_type = dataset[0]
    seed = dataset[1]
    
    dataset_name = f"{dataset_split_type}" if dataset_split_type != "S0" else dataset_split_type
    output_path = f"/home/wuxiao/llm-ddi/ddi/DESC_MOL-DDIE/fingerprint/datasets2025/{dataset_name}"
    command = f"python preprocessor.py {dataset_path}/{dataset_name} none 1 {output_path}"
    subprocess.run(command, shell=True, check=True)