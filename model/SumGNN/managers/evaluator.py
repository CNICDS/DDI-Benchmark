import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import  cohen_kappa_score, accuracy_score
from tqdm import tqdm

class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def print_attn_weight(self):
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                s = r_labels_pos.cpu().numpy().tolist()
                # print(s)
                #if s[0] in [ 0,  6, 12, 16, 17, 18, 21, 22, 30, 34, 35, 37, 40, 41, 43, 44, 45, 47, 49, 50, 51, 54, 55, 58, 61, 64, 65, 77, 80, 83, 85] or s[1] in [ 0,  6, 12, 16, 17, 18, 21, 22, 30, 34, 35, 37, 40, 41, 43, 44,45, 47, 49, 50, 51, 54, 55, 58, 61, 64, 65, 77, 80, 83, 85]:
                if 19 in s:
                    print(s, targets_pos)
                    score_pos = self.graph_classifier(data_pos)
                    s = score_pos.detach().cpu().numpy()
                    # with open('Drugbank/result.txt', 'a') as f:
                    #     f.write()

    def print_result(self):
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.eval()
        pos_labels = []
        pos_argscores = []
        pos_scores = []
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_pos = self.graph_classifier(data_pos)
                label_ids = r_labels_pos.to('cpu').numpy()
                pos_labels += label_ids.flatten().tolist()
                pos_argscores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 
                print( torch.max(score_pos, dim=1, out=None))
                pos_scores += torch.max(score_pos, dim=1)[0].cpu().flatten().tolist() 
                # s = r_labels_pos.cpu().numpy().tolist()
                # # print(s)
                # #if s[0] in [ 0,  6, 12, 16, 17, 18, 21, 22, 30, 34, 35, 37, 40, 41, 43, 44, 45, 47, 49, 50, 51, 54, 55, 58, 61, 64, 65, 77, 80, 83, 85] or s[1] in [ 0,  6, 12, 16, 17, 18, 21, 22, 30, 34, 35, 37, 40, 41, 43, 44,45, 47, 49, 50, 51, 54, 55, 58, 61, 64, 65, 77, 80, 83, 85]:
                # if 19 in s:
                #     print(s, targets_pos)
                #     score_pos = self.graph_classifier(data_pos)
                #     s = score_pos.detach().cpu().numpy()
        with open('Drugbank/results.txt', 'w') as f:
            for (x,y,z) in zip(pos_argscores, pos_labels, pos_scores):
                f.write('%d %d %d\n'%(x, y, z))


    
    @staticmethod
    def _to_onehot(labels, num_classes=None):
        labels = np.array(labels)
        if num_classes is None:
            num_classes = np.max(labels) + 1
        onehot = np.zeros((labels.size, num_classes))
        onehot[np.arange(labels.size), labels] = 1
        return onehot

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        dataloader = DataLoader(
            self.data, 
            batch_size=self.params.batch_size, 
            shuffle=False, 
            num_workers=self.params.num_workers, 
            collate_fn=self.params.collate_fn
        )

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                score_pos = self.graph_classifier(data_pos)

                label_ids = r_labels_pos.to('cpu').numpy()
                preds = torch.argmax(score_pos, dim=1).cpu().flatten().tolist()

                pos_labels += label_ids.flatten().tolist()
                pos_scores += preds

        # 基本指标
        f1_macro = metrics.f1_score(pos_labels, pos_scores, average='macro')
        acc_macro = metrics.balanced_accuracy_score(pos_labels, pos_scores)
        kappa = metrics.cohen_kappa_score(pos_labels, pos_scores)
        f1 = metrics.f1_score(pos_labels, pos_scores, average=None)

        # 需要 one-hot 编码计算 auc/aupr
        try:
            num_classes = max(max(pos_labels), max(pos_scores)) + 1
            labels_onehot = self._to_onehot(pos_labels, num_classes)
            scores_onehot = self._to_onehot(pos_scores, num_classes)

            auc_macro = metrics.roc_auc_score(labels_onehot, scores_onehot, average='macro', multi_class='ovr')
            aupr_macro = metrics.average_precision_score(labels_onehot, scores_onehot, average='macro')
        except ValueError:
            auc_macro = -1
            aupr_macro = -1

        # 可选保存预测文件
        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {
            'f1_macro': f1_macro,
            'acc_macro': acc_macro,
            'auc_macro': auc_macro,
            'aupr_macro': aupr_macro,
            'kappa': kappa
        }, {
            'f1': f1
        }



    # def eval(self, save=False):
    #     pos_scores = []
    #     pos_labels = []
    #     neg_scores = []
    #     neg_labels = []
    #     y_pred = []
    #     label_matrix = []
    #     dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

    #     self.graph_classifier.eval()
    #     with torch.no_grad():
    #         for b_idx, batch in enumerate(dataloader):

    #             data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
    #             # print([self.data.id2relation[r.item()] for r in data_pos[1]])
    #             # pdb.set_trace()
    #             score_pos = self.graph_classifier(data_pos)
    #             #score_neg = self.graph_classifier(data_neg)

    #             # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
    #             label_ids = r_labels_pos.to('cpu').numpy()
    #             pos_labels += label_ids.flatten().tolist()
    #             #y_pred = y_pred + F.softmax(output, dim = -1)[:, -1].cpu().flatten().tolist()
    #             #outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    #             pos_scores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 

    #             # pred = F.softmax(score_pos, dim = -1).detach().cpu().numpy()
    #             # label_mat = np.zeros(pred.shape)
    #             # label_mat[np.arange(label_mat.shape[0]), label_ids] = 1
    #             # y_pred.append(pred)
    #             # label_matrix.append(label_mat)

    #     # acc = metrics.accuracy_score(labels, preds)
    #     auc = metrics.f1_score(pos_labels, pos_scores, average='macro')
    #     auc_pr = metrics.f1_score(pos_labels, pos_scores, average='micro')
    #     f1 = metrics.f1_score(pos_labels, pos_scores, average=None)
    #     kappa = metrics.cohen_kappa_score(pos_labels, pos_scores)

    #     # y_pred = np.vstack(y_pred)
    #     # label_matrix = np.vstack(label_matrix)
    #     # #print(y_pred.T[0])
    #     # auprc = [average_precision_score(y_l, y_p) for (y_l, y_p) in zip(label_matrix.T ,  y_pred.T) if np.sum(y_l)>=2]
    #     # auroc = [roc_auc_score(y_l, y_p) for (y_l, y_p) in zip(label_matrix.T ,  y_pred.T) if np.sum(y_l)>=2]
        
    #     #print(s)
    #     if save:
    #         pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
    #         with open(pos_test_triplets_path) as f:
    #             pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
    #         pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
    #         with open(pos_file_path, "w") as f:
    #             for ([s, r, o], score) in zip(pos_triplets, pos_scores):
    #                 f.write('\t'.join([s, r, o, str(score)]) + '\n')

    #         neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
    #         with open(neg_test_triplets_path) as f:
    #             neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
    #         neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
    #         with open(neg_file_path, "w") as f:
    #             for ([s, r, o], score) in zip(neg_triplets, neg_scores):
    #                 f.write('\t'.join([s, r, o, str(score)]) + '\n')

    #     return {'auc': auc, 'microf1': auc_pr, 'k':kappa}, {'f1': f1}

class Evaluator_ddi2():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []

        y_pred = []
        y_label = []
        outputs = []

        pred_class = {}
        num_classes = None

        dataloader = DataLoader(
            self.data,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            collate_fn=self.params.collate_fn
        )

        self.graph_classifier.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                score_pos = self.graph_classifier(data_pos)

                pred = torch.sigmoid(score_pos)
                preds = pred.detach().to('cpu').numpy()
                labels = r_labels_pos.detach().to('cpu').numpy()

                if num_classes is None:
                    num_classes = preds.shape[1]

                labels = np.eye(num_classes)[labels.astype(int)]
                targets_pos = targets_pos.detach().to('cpu').numpy()

                for (label_ids, pred, label_t) in zip(labels, preds, targets_pos):
                    for i, (l, p) in enumerate(zip(label_ids, pred)):
                        if l == 1:
                            if i in pred_class:
                                pred_class[i]['pred'] += [p]
                                pred_class[i]['l'] += [label_t]
                                pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                            else:
                                pred_class[i] = {
                                    'pred': [p],
                                    'l': [label_t],
                                    'pred_label': [1 if p > 0.5 else 0]
                                }

        # 计算 per-class 指标
        roc_auc = [roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
        prc_auc = [average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
        accs = [accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]

        # 计算 ap@50
        all_preds = []
        all_labels = []
        for l in pred_class:
            all_preds.extend(pred_class[l]['pred'])
            all_labels.extend(pred_class[l]['l'])

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        if len(all_preds) >= 50:
            top_50_idx = np.argsort(all_preds)[-50:]
            ap_at_50 = np.mean(all_labels[top_50_idx])
        else:
            ap_at_50 = np.mean(all_labels)

        if save:
            pos_test_triplets_path = os.path.join(
                self.params.main_dir, 'dupload/data/SumGNN/{}/S0/{}.txt'.format(self.params.dataset, self.data.file_name)
            )
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(
                self.params.main_dir, 'upload/data/SumGNN/{}/S0/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name)
            )
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(
                self.params.main_dir, 'upload/data/SumGNN/{}/S0/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name)
            )
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(
                self.params.main_dir, 'upload/data/SumGNN/{}/S0/grail_neg_{}_{}_predictions.txt'.format(
                    self.params.dataset, self.data.file_name, self.params.constrained_neg_prob
                )
            )
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {
            'acc_macro': np.mean(accs),
            'auc_macro': np.mean(roc_auc),
            'aupr_macro': np.mean(prc_auc),
            'ap@50': ap_at_50
        }, {
            "auc_all": roc_auc,
            "aupr_all": prc_auc,
            "acc_all": accs
        }