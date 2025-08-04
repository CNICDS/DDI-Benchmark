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
warnings.filterwarnings('ignore',category=UserWarning)
random.seed(226)
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
parser.add_argument('--pkl_name', type=str, default='s1s2_11_tw.pkl')
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
device = 'cuda:0' 
#if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################

df_ddi_train = pd.read_csv('./data/twosides/train_s2.csv')
df_ddi_val = pd.read_csv('./data/twosides/valid_s2.csv')
df_ddi_test = pd.read_csv('./data/twosides/test_s2.csv')

train_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'], df_ddi_train['Neg samples'])]
val_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'], df_ddi_val['Neg samples'])]
test_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'], df_ddi_train['Neg samples'])]

train_data = DrugDataset(train_tup)
val_data = DrugDataset(val_tup)
test_data = DrugDataset(test_tup)


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
        p_score = model(pos_tri)
        probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        ground_truth.append(np.ones(len(p_score)))

        neg_tri = [tensor.to(device=device) for tensor in neg_tri]
        n_score = model(neg_tri)
        probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        ground_truth.append(np.zeros(len(n_score)))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

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


def test(test_data_loader,model):
    model.eval()
    test_probas_pred, test_ground_truth = [], []

    with torch.no_grad():
        for batch in test_data_loader:
            # do_compute() 内部需保证把张量移至 device
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)

    # 拼接整个测试集结果
    test_probas_pred = np.concatenate(test_probas_pred)
    test_ground_truth = np.concatenate(test_ground_truth)

    # ---------- 主要评价指标 ----------
    test_acc, test_auc_roc, test_f1, test_precision,test_recall,test_int_ap, test_ap = do_compute_metrics(test_probas_pred, test_ground_truth)

    # ---------- 计算并保存混淆矩阵 ----------
    test_pred_binary = (test_probas_pred > 0.5).astype(int)
    cm = confusion_matrix(test_ground_truth, test_pred_binary)
    cm_path = f'confusion_matrix.pkl'
    with open(cm_path, "wb") as f:
        pickle.dump(cm, f)

    # ---------- 输出与写文件 ----------
    print('\n')
    print('============================== Test Result ==============================')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')
    print(f'\t\tConfusion matrix saved to: {cm_path}')

model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
loss = custom_loss.SigmoidLoss()
# print(model)
model.to(device=device)

# # if __name__ == '__main__':
test_model = torch.load(pkl_name)
test(test_data_loader,test_model)


