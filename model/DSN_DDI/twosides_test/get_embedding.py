from datetime import datetime
import time 
import argparse
import torch
import random
import os

from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader
import warnings
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore',category=UserWarning)
random.seed(11)
######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')


parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=0.1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='s0_11_tw.pkl')
parser.add_argument('--result_name', type=str, default='./model/DSN_DDI/drugbank_test/result/s0_226_db.txt')

args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
pkl_name = args.pkl_name
result_name = args.result_name
os.makedirs(os.path.dirname(result_name), exist_ok=True)

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
device = 'cuda:1' 
#if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################
df_ddi_train = pd.read_csv('./data/twosides/train_s0.csv')
# df_ddi_val = pd.read_csv('./data/twosides/valid_s2.csv')
df_ddi_test = pd.read_csv('./data/twosides/test_s0.csv')

train_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'], df_ddi_train['Neg samples'])]
# val_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'], df_ddi_val['Neg samples'])]
test_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'], df_ddi_train['Neg samples'])]

train_data = DrugDataset(train_tup)
# val_data = DrugDataset(val_tup)
test_data = DrugDataset(test_tup)


print(f"testing with {len(test_data)}")

# train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
# val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3,num_workers=2)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=2)



def extract_embeddings(model, test_loader, save_path_emb, save_path_label=None, device=device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pos_tri, neg_tri = batch

            for tri, label_value in [(pos_tri, 1), (neg_tri, 0)]:
                h_data, t_data, rels, b_graph = tri
                h_data = h_data.to(device)
                t_data = t_data.to(device)
                rels = rels.to(device)
                b_graph = b_graph.to(device)

                try:
                    emb = model.encode_pair((h_data, t_data, rels, b_graph))  # shape: [batch_size, emb_dim]
                    all_embeddings.append(emb.cpu().numpy())
                    all_labels.append(np.full((emb.shape[0],), label_value))  # shape: [batch_size,]
                except Exception as e:
                    print(f"[ERROR] Failed on label={label_value} tri: {e}")
                    continue

    if len(all_embeddings) == 0:
        print("[ERROR] No embeddings extracted.")
        return

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)

    np.save(save_path_emb, X)
    print(f"[INFO] Saved embeddings to {save_path_emb}")
    if save_path_label:
        np.save(save_path_label, y)
        print(f"[INFO] Saved labels to {save_path_label}")




model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
loss = custom_loss.SigmoidLoss()
# print(model)
model.to(device=device)

# # if __name__ == '__main__':
test_model = torch.load(pkl_name, map_location=device)
extract_embeddings(test_model, test_data_loader,
                     save_path_emb='saved_embeddings/test_embeddings.npy',
                     save_path_label='saved_embeddings/test_labels.npy',device=device)


