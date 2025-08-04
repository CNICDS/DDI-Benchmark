import os
from itertools import product
import subprocess

config_file = "upload/model/configs/main_drugbank.yaml"
dataset_split_type = ["S0","S1","S2"]
seeds = [11, 14, 224]

for dataset in product(dataset_split_type, seeds):
    dataset_split_type = dataset[0]
    seed = dataset[1]
    
    dataset_name = f"{dataset_split_type}" if dataset_split_type != "S0" else dataset_split_type
    command = f"CUDA_VISIBLE_DEVICES=2 python drugbank.py --drugbank_root upload/data/3DGT-DDI/drugbank2025/ --drugbank_path drugbank2025.csv --batch_size 16 --epochs 10 --lr 2e-5 --weight_decay 1e-2 --num_class 2 --cutoff 10.0 --num_layers 6 --hidden_channels 128 --num_filters 128 --num_gaussians 197 --g_out_channels 32 --seed {seed} --dataset_name {dataset_name}"
    subprocess.run(command, shell=True, check=True)