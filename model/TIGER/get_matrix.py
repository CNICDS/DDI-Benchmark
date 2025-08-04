#UniDDI

import os
import argparse
import gc
import torch
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import json
import copy
from utils import *
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from model.tiger import TIGER
from torch_geometric.utils import degree
from torch.utils.data.distributed import DistributedSampler
from data_process import smile_to_graph, read_smiles, read_interactions, generate_node_subgraphs, read_network
from sklearn.model_selection import StratifiedKFold, KFold
from train_eval import train, test, eval
from sklearn.metrics import confusion_matrix
import random
import torch.nn.functional as F
import pickle
def init_args(user_args=None):

    parser = argparse.ArgumentParser(description='TIGER')

    parser.add_argument('--model_name', type=str, default='tiger')

    parser.add_argument('--dataset', type=str, default="drugbank_new2")

    parser.add_argument('--folds', type=int, default=1)
    parser.add_argument('--layer', type=int, default=2)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--model_episodes', type=int, default=100)
    parser.add_argument('--extractor', type=str, default="randomWalk") ##option [khop-subtree, randomWalk, probability]
    parser.add_argument('--graph_fixed_num', type=int, default=1)
    parser.add_argument('--khop', type=int, default=2)
    parser.add_argument('--fixed_num', type=int, default=32)

    # Graphormer
    parser.add_argument("--d_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_smiles_degree", type=int, default=300)
    parser.add_argument("--max_graph_degree", type=int, default=600)
    parser.add_argument("--dropout", type=float, default=0.2)

    # coeff
    parser.add_argument('--sub_coeff', type=float, default=0.1)
    parser.add_argument('--mi_coeff', type=float, default=0.1)

    parser.add_argument('--s_type', type=str, default='random')
    parser.add_argument('--load_model_path',type=str,default='best_save/tiger/drugbank_new2/randomWalk/11/0.64803/TIGER.pt')
    args = parser.parse_args()

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def k_fold(data, kf, folds, y):

    test_indices = []
    train_indices = []

    if len(y):
        for _, idx in kf.split(torch.zeros(len(data)), y):
            test_indices.append(idx)
    else:
        for _, idx in kf.split(data):
            test_indices.append(idx)

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(data), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def split_fold(folds, dataset, labels, scenario_type='random'):

    test_indices, train_indices, val_indices = [], [], []

    if scenario_type == 'random':##这是根据interactions在划分的数据集，也就是根据interactions的label进行的数据集划分
        skf = StratifiedKFold(folds, shuffle=True, random_state=2023)
        train_indices, test_indices, val_indices = k_fold(dataset, skf, folds, labels)

    return train_indices, test_indices, val_indices

def load_data(args):

    ##这个逻辑是这样的:先读出ddi中所有drug的数量，并不一定是全部的drug

    dataset = args.dataset

    data_path = "dataset/" + dataset + "/"

    ligands = read_smiles(os.path.join(data_path, "drug_smiles.txt"))


    # smiles to graphs
    print("load drug smiles graphs!!")
    smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph(data_path, ligands)

    print("load networks !!")
    num_node, network_edge_index, network_rel_index, num_rel = read_network(data_path + "networks.txt")

    print("load DDI samples!!")
    interactions_label, all_contained_drgus = read_interactions(os.path.join(data_path, "ddi.txt"), smile_graph)
    print("type:", type(interactions_label))
    print("shape:", np.array(interactions_label).shape)
    print("content:", interactions_label[:3])
    interactions = interactions_label[:, :2]
    labels = interactions_label[:, 3]


    print("generate subgraphs!!")
    drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(dataset, all_contained_drgus,
                                                                                  network_edge_index, network_rel_index,
                                                                                  num_rel, args)

    data_sta = {
        'num_nodes': num_node + 1,
        'num_rel_mol': num_rel_mol_update + 1,
        'num_rel_graph': num_rel_update + 1,
        'num_interactions': len(interactions),
        'num_drugs_DDI': len(all_contained_drgus),
        'max_degree_graph': max_smiles_degree + 1,
        'max_degree_node': int(max_subgraph_degree)+1
    }

    print(data_sta)

    return interactions, labels, smile_graph, drug_subgraphs, data_sta

def save(save_dir, args, train_log, test_log):
    args.device = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "/args.json", 'w') as f:
        json.dump(args.__dict__, f)
    with open(save_dir + '/test_results.json', 'w') as f:
        json.dump(test_log, f)
    with open(save_dir + '/train_log.json', 'w') as f:
        json.dump(train_log, f)

def save_results(save_dir, args, results_list):
    acc = []
    auc = []
    aupr = []
    f1 = []

    for r in results_list:
        acc.append(r['acc'])
        auc.append(r['auc'])
        aupr.append(r['aupr'])
        f1.append(r['f1'])

    acc = np.array(acc)
    auc = np.array(auc)
    aupr = np.array(aupr)
    f1 = np.array(f1)

    results = {
        'acc':[np.mean(acc),np.std(acc)],
        'auc':[np.mean(auc),np.std(auc)],
        'aupr': [np.mean(aupr), np.std(aupr)],
        'f1': [np.mean(f1), np.std(f1)],
    }

    args = vars(args)
    args.update(results)

    with open(save_dir + args['extractor'] + '_all_results.json', 'a+') as f:
        json.dump(args, f)


def init_model(args, dataset_statistics):
    if args.model_name == 'tiger':
        model = TIGER(max_layer=args.layer,
                      num_features_drug = 67,
                      num_nodes=dataset_statistics['num_nodes'],
                      num_relations_mol=dataset_statistics['num_rel_mol'],
                      num_relations_graph=dataset_statistics['num_rel_graph'],
                      output_dim=args.d_dim,
                      max_degree_graph=dataset_statistics['max_degree_graph'],
                      max_degree_node = dataset_statistics['max_degree_node'],
                      sub_coeff=args.sub_coeff,
                      mi_coeff=args.mi_coeff,
                      dropout=args.dropout,
                      device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))


    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    return model, optimizer

def load_split_txt(file_path):
    data_list = []
    labels = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            drug1, drug2, relation, label = line.strip().split(',')
            data_list.append((drug1, drug2, int(relation)))  # 你原本的data结构中可能就是这种tuple
            labels.append(int(label))

    return data_list, labels

def main(args=None):
    seed = 11
    print(f"Seed: {seed}")
    if args is None:
        args = init_args()

    # 设置设备
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    setup_seed(seed)

    # 加载数据
    data, labels, smile_graph, node_graph, dataset_statistics = load_data(args)
    test_data_raw, test_labels = load_split_txt(os.path.join('dataset', args.dataset, 'test.txt'))
    test_data = DTADataset(x=test_data_raw, y=test_labels, sub_graph=node_graph, smile_graph=smile_graph)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 加载模型
    assert args.load_model_path is not None, "请指定要加载的模型路径：args.load_model_path"
    model, _ = init_model(args, dataset_statistics)
    model.load_state_dict(torch.load(args.load_model_path, map_location=device))
    model.to(device)
    model.eval()

    print("==============Start Testing==============")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, ncols=80):
            data_mol1 = data[0].to(device)
            data_drug1 = data[1].to(device)
            data_mol2 = data[2].to(device)
            data_drug2 = data[3].to(device)

            predicts, _ = model(data_mol1, data_drug1, data_mol2, data_drug2)

            # pred = torch.argmax(predicts, dim=1)
            label = data_mol1.y

            all_preds.append(predicts.cpu())
            all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds)
    all_preds = (all_preds > 0.5).long()  # 将概率转为 0 或 1
    all_labels = torch.cat(all_labels)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    save_path = os.path.join(os.path.dirname(args.load_model_path), 'confusion_matrix.pt')
    with open(save_path, 'wb') as f:
        pickle.dump(cm, f)
    print("SAVED")

    print(f"混淆矩阵已保存至: {save_path}")

if __name__ == "__main__":
    main()
