import os
import numpy as np
import torch
import random
from sklearn import metrics
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score,cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
import pickle
from sklearn.metrics import confusion_matrix
GLOBAL_SEED=11  ##11 14
GLOBAL_WORKER_ID=None


def init_fn(worker_id): 
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    seed = GLOBAL_SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

class Evaluator_multiclass():
    """
    Drugbank
    """
    def __init__(self, params, classifier, data,is_test=False):
        self.params = params
        self.graph_classifier = classifier
        self.data = data
        self.global_graph = data.global_graph
        self.move_batch_to_device = move_batch_to_device_dgl
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.is_test = is_test
        self.eval_times = 0
        self.current_epoch = 0

    def eval(self):
        self.eval_times += 1
        self.current_epoch += 1

        dataloader = DataLoader(
            self.data,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=init_fn
        )

        all_preds = []
        all_labels = []

        self.graph_classifier.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                data, r_labels, _ = self.move_batch_to_device(batch, self.params.device, multi_type=1)
                logits = self.graph_classifier(data)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(r_labels.cpu().numpy().tolist())

        # 多分类基础指标
        macro_acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        kappa = cohen_kappa_score(all_labels, all_preds)

        # AUC/AUPR 所需概率
        all_probs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                data, r_labels, _ = self.move_batch_to_device(batch, self.params.device, multi_type=1)
                logits = self.graph_classifier(data)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        all_probs = np.vstack(all_probs)
        all_labels_np = np.array(all_labels)
        num_classes = all_probs.shape[1]
        label_binarized = label_binarize(all_labels_np, classes=list(range(num_classes)))

        # 防止 AUC 报错：只在多于1类时计算
        try:
            if len(np.unique(all_labels_np)) < 2:
                raise ValueError("Only one class in y_true")
            macro_auc = roc_auc_score(label_binarized, all_probs, average='macro', multi_class='ovr')
            macro_aupr = average_precision_score(label_binarized, all_probs, average='macro')
        except ValueError as e:
            print(f"⚠️ AUC/AUPR skipped due to ValueError: {e}")
            macro_auc = -1
            macro_aupr = -1

        return {
            'acc_macro': macro_acc,
            'f1_macro': macro_f1,
            'f1_micro': micro_f1,
            'kappa': kappa,
            'auc_macro': macro_auc,
            'aupr_macro': macro_aupr,
        }, {
            'all_labels': all_labels,
            'all_preds': all_preds,
            'all_probs': all_probs
        }

    def get_cm(self, save_path='confusion_matrix.pkl'):
        """
        计算混淆矩阵并保存为pkl格式。
        """
        dataloader = DataLoader(
            self.data,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=init_fn
        )

        all_preds = []
        all_labels = []

        self.graph_classifier.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating confusion matrix"):
                data, r_labels, _ = self.move_batch_to_device(batch, self.params.device, multi_type=1)
                logits = self.graph_classifier(data)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(r_labels.cpu().numpy().tolist())

        cm = confusion_matrix(all_labels, all_preds)

        # 保存为 .pkl 文件
        with open(save_path, 'wb') as f:
            pickle.dump(cm, f)

        print(f"✅ Confusion matrix saved to: {save_path}")
        return cm



    # def eval(self):
    #     self.eval_times += 1
    #     scores = []
    #     labels = []
    #     self.current_epoch += 1
    #     dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn,worker_init_fn=init_fn)

    #     self.graph_classifier.eval()
    #     with torch.no_grad():
    #         for b_idx, batch in tqdm(enumerate(dataloader)):
    #             data, r_labels, polarity= self.move_batch_to_device(batch, self.params.device,multi_type=1)
    #             score = self.graph_classifier(data)


    #             label_ids = r_labels.to('cpu').numpy()
    #             labels += label_ids.flatten().tolist()
    #             scores += torch.argmax(score, dim=1).cpu().flatten().tolist() 

    #     auc = metrics.f1_score(labels, scores, average='macro')
    #     auc_pr = metrics.f1_score(labels, scores, average='micro')
    #     f1 = metrics.f1_score(labels, scores, average=None)
    #     kappa = metrics.cohen_kappa_score(labels, scores)
    #     return {'auc': auc, 'auc_pr': auc_pr, 'k':kappa}, {'f1': f1}

# class Evaluator_multilabel():
#     """
#     BioSNAP
#     """
#     def __init__(self, params, classifier, data):
#         self.params = params
#         self.graph_classifier = classifier
#         self.data = data
#         self.global_graph = data.global_graph
#         self.move_batch_to_device = move_batch_to_device_dgl
#         self.collate_fn = collate_dgl
#         self.num_workers = params.num_workers
    

class Evaluator_multilabel():
    """
    BioSNAP
    """
    def __init__(self, params, classifier, data):
        self.params = params
        self.graph_classifier = classifier
        self.data = data
        self.global_graph = data.global_graph
        self.move_batch_to_device = move_batch_to_device_dgl
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers

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
                self.params.main_dir, 'upload/data/KnowDDI/data/{}/S0/{}.txt'.format(self.params.dataset, self.data.file_name)
            )
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(
                self.params.main_dir, 'upload/data/KnowDDI/data/{}/S0/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name)
            )
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(
                self.params.main_dir, 'upload/data/KnowDDI/data/{}/S0/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name)
            )
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(
                self.params.main_dir, 'upload/data/KnowDDI/data/{}/S0/grail_neg_{}_{}_predictions.txt'.format(
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


    # def eval(self):
    #     pred_class = {}
    #     dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn,worker_init_fn=init_fn)
        
    #     self.graph_classifier.eval()
    #     with torch.no_grad():
    #         for batch in tqdm(dataloader):
    #             data, r_labels, polarity = self.move_batch_to_device(batch, self.params.device,multi_type=2)
    #             score_pos = self.graph_classifier(data)

    #             m = nn.Sigmoid()
    #             pred = m(score_pos)
    #             labels = r_labels.detach().to('cpu').numpy() # batch * 200
    #             preds = pred.detach().to('cpu').numpy() # batch * 200
    #             polarity = polarity.detach().to('cpu').numpy()
    #             for (label, pred, pol) in zip(labels, preds, polarity):
    #                 for i, (l, p) in enumerate(zip(label, pred)):
    #                     if l == 1:
    #                         if i in pred_class:
    #                             pred_class[i]['pred'] += [p]
    #                             pred_class[i]['pol'] += [pol] 
    #                             pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
    #                         else:
    #                             pred_class[i] = {'pred':[p], 'pol':[pol], 'pred_label':[1 if p > 0.5 else 0]}
                                
    #     roc_auc = [ roc_auc_score(pred_class[l]['pol'], pred_class[l]['pred']) for l in pred_class]
    #     prc_auc = [ average_precision_score(pred_class[l]['pol'], pred_class[l]['pred']) for l in pred_class]
    #     ap =  [accuracy_score(pred_class[l]['pol'], pred_class[l]['pred_label']) for l in pred_class]
    #     return {'auc': np.mean(roc_auc), 'auc_pr': np.mean(prc_auc), 'f1': np.mean(ap)}, {"auc_all":roc_auc,"aupr_all":prc_auc, "f1_all":ap}
