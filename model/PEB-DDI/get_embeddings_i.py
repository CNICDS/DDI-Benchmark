from datetime import datetime
import time 
import argparse

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from get_args import config
import models_i
import custom_loss
from data_preprocessing_i import DrugDataset, DrugDataLoader
import warnings
import pickle
import os
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore',category=UserWarning)
from tqdm import tqdm
CUDA_LAUNCH_BLOCKING=1

######################### Parameters ######################
dataset_name = config['dataset_name']

import random
seed = config[dataset_name]['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

pkl_name = config[dataset_name]["inductive_pkl_dir"]
result_name = config[dataset_name]['result']
params = config['params']
lr = params['lr']
n_epochs = 100
batch_size = params['batch_size']
weight_decay = params['weight_decay']
neg_samples = params['neg_samples']
data_size_ratio = params['data_size_ratio']
device = 'cuda:1' if torch.cuda.is_available() and params['use_cuda'] else 'cpu'
# device = 'cpu'

print(dataset_name, params)
n_atom_feats = 55
rel_total = 200
kge_dim = 128
############################################################

###### Dataset
df_ddi_train = pd.read_csv(config[dataset_name]["induc_ddi_train"])
df_ddi_s1 = pd.read_csv(config[dataset_name]["induc_s1"])
df_ddi_s2 = pd.read_csv(config[dataset_name]["induc_s2"])

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)

print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
s1_data_loader = DrugDataLoader(s1_data, batch_size=batch_size *3,num_workers=2)
s2_data_loader = DrugDataLoader(s2_data, batch_size=batch_size *3,num_workers=2)


def extract_embeddings(model, test_loader, save_path_emb, save_path_label=None, device='cuda:0'):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pos_tri, neg_tri = batch  # 两个 9元组结构

            for tri, label_value in [(pos_tri, 1), (neg_tri, 0)]:
                if len(tri) != 8:
                    print(f"[ERROR] Unexpected tri length: {len(tri)} (expected 9)")
                    continue
                try:
                    # 将 tri 的每个元素移到 device
                    tri = [x.to(device) for x in tri]

                    # 提取嵌入：你需要确保 encode_pair() 接收完整结构
                    emb = model.encode_pair(tri)  # emb: [batch_size, emb_dim]

                    all_embeddings.append(emb.cpu().numpy())
                    all_labels.append(np.full((emb.shape[0],), label_value))
                except Exception as e:
                    print(f"[ERROR] Failed on label={label_value} tri: {e}")
                    continue

    if len(all_embeddings) == 0:
        print("[ERROR] No embeddings extracted.")
        return

    X = np.concatenate(all_embeddings, axis=0)
    np.save(save_path_emb, X)
    print(f"[INFO] Saved embeddings to {save_path_emb}")

    if save_path_label:
        y = np.concatenate(all_labels, axis=0)
        np.save(save_path_label, y)
        print(f"[INFO] Saved labels to {save_path_label}")


model = models_i.MVN_DDI([n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, rel_total, [32,32,32,32], [4, 4, 4, 4], 64, 0.06)
pkl_name = 'pkl/tw_s1s2_11-2.pkl'
state_dict = torch.load(pkl_name, map_location=device)
model.load_state_dict(state_dict)
model.eval()
# print(model)
model.to(device=device)
# # if __name__ == '__main__':
extract_embeddings(
    model,
    s1_data_loader,
    save_path_emb='saved_embeddings/s1_embeddings.npy',
    save_path_label='saved_embeddings/s1_labels.npy',
    device=device  # 你之前设置的 device='cuda:6'
)
extract_embeddings(
    model,
    s2_data_loader,
    save_path_emb='saved_embeddings/s2_embeddings.npy',
    save_path_label='saved_embeddings/s2_labels.npy',
    device=device  # 你之前设置的 device='cuda:6'
)