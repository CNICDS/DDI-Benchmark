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
rel_total = 197
kge_dim = 128
######################### Dataset ######################

df_ddi_train = pd.read_csv(config[dataset_name]["trans_ddi_train"])
df_ddi_test = pd.read_csv(config[dataset_name]["trans_ddi_test"])
df_ddi_valid= pd.read_csv(config[dataset_name]["trans_ddi_valid"])

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
val_tup = [(h, t, r) for h, t, r in zip(df_ddi_valid['d1'], df_ddi_valid['d2'], df_ddi_valid['type'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3,num_workers=2)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=2)

def do_compute(batch, device, model):
        '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''
        probas_pred, ground_truth = [], []
        pos_tri, neg_tri = batch
        
        pos_tri = [tensor.to(device=device) for tensor in pos_tri]
        # print('p_score')
        p_score = model(pos_tri)
        probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        ground_truth.append(np.ones(len(p_score)))

        neg_tri = [tensor.to(device=device) for tensor in neg_tri]
        # print('n_score')
        n_score = model(neg_tri)
        probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        ground_truth.append(np.zeros(len(n_score)))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)
        # print('batch ok')

        return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap


def test(test_loader, model):
    """
    评估模型并可选地保存混淆矩阵到 cm_save_path。
    """
    probas_all, labels_all = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            _, _, probas_pred, ground_truth = do_compute(batch, device, model)
            probas_all.append(probas_pred)
            labels_all.append(ground_truth)
    
    probas_all = np.concatenate(probas_all, axis=0)
    labels_all  = np.concatenate(labels_all,  axis=0)
    
    # --------- 评估指标 ---------
    test_acc, test_auc_roc, test_f1, test_precision, \
    test_recall, test_int_ap, test_ap = do_compute_metrics(probas_all, labels_all)

    # --------- 计算并保存混淆矩阵 ---------
        # 多标签：先二值化，再展平
    preds  = (probas_all > 0.5).astype(int).flatten()
    truths = labels_all.flatten()
    cm = confusion_matrix(truths, preds)
    cm_save_path= f'PEB-DDI/cm/{dataset_name}_cm.pkl'
    os.makedirs(os.path.dirname(cm_save_path), exist_ok=True)
    with open(cm_save_path, 'wb') as f:
        pickle.dump(cm, f)
    
    # --------- 打印 / 保存结果 ---------
    result  = (
        '============================== Test Result ==============================\n'
        f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},'
        f'test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}\n'
        f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},'
        f'test_ap: {test_ap:.4f}\n'
    )
    print('\n' + result)


model = models_t.MVN_DDI([n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, rel_total, [64,64,64,64], [2, 2, 2, 2],64, 0.0)
pkl_name = 'pkl/db2025_s0_11.pkl'
state_dict = torch.load(pkl_name, map_location=device)
model.load_state_dict(state_dict)
model.eval()
# print(model)
model.to(device=device)
# # if __name__ == '__main__':

test(test_data_loader,model)


