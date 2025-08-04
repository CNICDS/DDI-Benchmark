import os
from itertools import product
import subprocess

dataset_split_type = ["S0","S1","S2"]
seeds = [11, 14, 224]

for dataset in product(dataset_split_type, seeds):
    dataset_split_type = dataset[0]
    seed = dataset[1]
    
    dataset_name = f"{dataset_split_type}" 
    output_dir = f"upload/model/DESC_MOL-DDIE/output/{dataset_name}_{seed}"
    command = f"CUDA_VISIBLE_DEVICES=3 python run_ddie.py \
    --task_name MRPC \
    --model_type bert \
    --data_dir upload/data/DESC_MOL-DDIE/drugbank/{dataset_name} \
    --model_name_or_path upload/model/pretrained_models/scibert_scivocab_uncased \
    --per_gpu_train_batch_size 32 \
    --num_train_epochs 3. \
    --dropout_prob .1 \
    --weight_decay .01 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --max_seq_length 128 \
    --use_cnn \
    --conv_window_size 5 \
    --pos_emb_dim 10 \
    --activation gelu \
    --desc_conv_window_size 3 \
    --desc_conv_output_size 20 \
    --molecular_vector_size 50 \
    --gnn_layer_hidden 5 \
    --gnn_layer_output 1 \
    --gnn_mode sum \
    --gnn_activation gelu \
    --seed {seed}\
    --fingerprint_dir upload/model/DESC_MOL-DDIE/fingerprint/datasets/{dataset_name} \
    --output_dir {output_dir}"
    # subprocess.run(command, shell=True, check=True)
    subprocess.run(command, shell=True, capture_output=False)
