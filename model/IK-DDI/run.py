import os
from itertools import product
import subprocess

config_file = "upload/model/configs/main_drugbank.yaml"
dataset_split_type = ["S0", "S1", "S2"]
seeds = [11, 14, 224]

for dataset in product(dataset_split_type, seeds):
    dataset_split_type = dataset[0]
    seed = dataset[1]
    
    dataset_name = dataset_split_type
    command = f"python main.py --seed {seed} --dataset_name {dataset_name}"
    subprocess.run(command, shell=True, check=True)