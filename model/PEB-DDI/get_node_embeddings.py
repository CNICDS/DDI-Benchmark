from datetime import datetime
import time 
import argparse
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import models_t
import custom_loss
from data_preprocessing_t import DrugDataset, DrugDataLoader
from get_args import config
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore',category=UserWarning)
import pickle
import os
from tqdm import tqdm

dataset_name = config['dataset_name']

import random
seed = config[dataset_name]['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

######################### Parameters ######################
dataset_name = config['dataset_name']
pkl_name = config[dataset_name]["transductive_pkl_dir"]
result_name = config[dataset_name]['result']
params = config['params']
lr = params['lr']
n_epochs = params['n_epochs']
batch_size = params['batch_size']
weight_decay = params['weight_decay']
neg_samples = params['neg_samples']
data_size_ratio = params['data_size_ratio']
device = 'cuda:1' if torch.cuda.is_available() and params['use_cuda'] else 'cpu'
print(dataset_name, params)
n_atom_feats = 55
rel_total = 200
kge_dim = 128
######################### Dataset ######################

# df_ddi_train = pd.read_csv(config[dataset_name]["trans_ddi_train"])
df_ddi_test = pd.read_csv(config[dataset_name]["trans_ddi_test"])
# df_ddi_valid= pd.read_csv(config[dataset_name]["trans_ddi_valid"])

# train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
# val_tup = [(h, t, r) for h, t, r in zip(df_ddi_valid['d1'], df_ddi_valid['d2'], df_ddi_valid['type'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

# train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
# val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)

print(f" testing with {len(test_data)}")

# train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
# val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3,num_workers=2)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=2)



def extract_embeddings(model, test_loader, save_path_emb, save_path_label=None, device='cuda:0'):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pos_tri, neg_tri = batch  # 两个 9元组结构

            for tri, label_value in [(pos_tri, 1)]:
                if len(tri) != 9:
                    print(f"[ERROR] Unexpected tri length: {len(tri)} (expected 9)")
                    continue
                try:
                    # 将 tri 的每个元素移到 device
                    tri = [x.to(device) for x in tri]

                    # 提取嵌入：你需要确保 encode_pair() 接收完整结构
                    emb = model.encode_node(tri)  # emb: [batch_size, emb_dim]

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
    print(X.shape)
    print(f"[INFO] Saved embeddings to {save_path_emb}")
 





model = models_t.MVN_DDI([n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, rel_total, [64,64,64,64], [2, 2, 2, 2],64, 0.0)
pkl_name = 'pkl/tw_s0_11.pkl'
state_dict = torch.load(pkl_name, map_location=device)
model.load_state_dict(state_dict)
model.eval()
# print(model)
model.to(device=device)
# # if __name__ == '__main__':

extract_embeddings(
    model,
    test_data_loader,
    save_path_emb='saved_embeddings/tw_nodeembeddings_0.npy',
    device=device  # 你之前设置的 device='cuda:6'
)

