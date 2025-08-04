import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import json
from torch.nn.utils import clip_grad_norm_


class Trainer():
    def __init__(self, params, graph_classifier, train, train_evaluator = None, valid_evaluator=None, test_evaluator = None):
        self.graph_classifier = graph_classifier
        self.train_evaluator=train_evaluator
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train
        self.test_evaluator = test_evaluator
        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        if params.dataset == 'drugbank':
            self.criterion = nn.CrossEntropyLoss()
        elif params.dataset == 'BioSNAP':
            self.criterion = nn.BCELoss(reduce=False) 
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def load_model(self):
        self.graph_classifier.load_state_dict(torch.load("my_resnet.pth"))

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []
        # 初始化 num_classes
        num_classes = None

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        bar = tqdm(enumerate(dataloader))
        for b_idx, batch in bar:
            #data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
            
            self.optimizer.zero_grad()
            score_pos = self.graph_classifier(data_pos)
            if self.params.dataset == 'drugbank':
                loss = self.criterion(score_pos, r_labels_pos)
            elif self.params.dataset == 'BioSNAP':
                m = nn.Sigmoid()
                score_pos = m(score_pos)
                targets_pos = targets_pos.unsqueeze(1)
                num_classes = score_pos.size(1)
                r_labels_pos = F.one_hot(r_labels_pos, num_classes=num_classes).float()
                # r_labels_pos = r_labels_pos[:, :score_pos.size(1)]
                loss_train = self.criterion(score_pos, r_labels_pos * targets_pos)
                loss = torch.sum(loss_train * r_labels_pos)
                # loss_train = self.criterion(score_pos, r_labels_pos * targets_pos)
                # loss = torch.sum(loss_train * r_labels_pos)            
            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            bar.set_description('epoch: ' + str(b_idx+1) + '/ loss_train: ' + str(loss.cpu().detach().numpy()))
    
            # except RuntimeError:
            #     print(data_pos, r_labels_pos, targets_pos)
            #    print('-------runtime error--------')
            #    continue
            with torch.no_grad():
                total_loss += loss.item()
                if self.params.dataset != 'BioSNAP':
                    
                    label_ids = r_labels_pos.to('cpu').numpy()
                    all_labels += label_ids.flatten().tolist()
                    #y_pred = y_pred + F.softmax(output, dim = -1)[:, -1].cpu().flatten().tolist()
                    #outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
                    all_scores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 
            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result, save_dev_data = self.valid_evaluator.eval()
                test_result, save_test_data = self.test_evaluator.eval()
                logging.info('\033[95m Eval Performance:' + str(result) + 'in ' + str(time.time() - tic)+'\033[0m')
                logging.info('\033[93m Test Performance:' + str(test_result) + 'in ' + str(time.time() - tic)+'\033[0m')
                if result['acc_macro'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['acc_macro']
                    self.not_improved_count = 0
                    if self.params.dataset != 'BioSNAP':
                        logging.info('\033[93m Test Performance Per Class:' + str(save_test_data) + 'in ' + str(time.time() - tic)+'\033[0m')
                    else:
                        with open('experiments/%s/result.json'%(self.params.experiment_name), 'a') as f:
                            f.write(json.dumps(save_test_data))
                            f.write('\n')
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['acc_macro']
        weight_norm = sum(map(lambda x: torch.norm(x), model_params))
        if self.params.dataset != 'BioSNAP':
            f1_macro = metrics.f1_score(all_labels, all_scores, average='macro')
            auc_pr = metrics.f1_score(all_labels, all_scores, average='micro')

            return total_loss/b_idx, f1_macro, auc_pr, weight_norm
        else:
            return total_loss/b_idx, 0, 0, weight_norm


    def train(self):
        self.reset_training_state()

        # ⏱ 记录训练整体时间
        total_start_time = time.time()

        # 🚀 显存峰值监控初始化
        torch.cuda.reset_peak_memory_stats(self.params.device)

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            
            loss, f1_macro, auc_pr, weight_norm = self.train_epoch()

            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training f1_macro: {f1_macro}, training auc_pr: {auc_pr}, best validation f1_macro: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed:.2f}s')

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

        # ✅ 训练结束后记录训练时间
        train_end_time = time.time()

        # ✅ 评估验证集
        eval_start = time.time()
        valid_result, valid_details = self.valid_evaluator.eval()
        eval_end = time.time()

        # ✅ 评估测试集
        test_start = time.time()
        test_result, test_details = self.test_evaluator.eval()
        test_end = time.time()

        # ✅ 总结各种时间
        total_end_time = time.time()
        training_time = train_end_time - total_start_time
        eval_time = eval_end - eval_start
        test_time = test_end - test_start
        total_time = total_end_time - total_start_time
        peak_gpu_memory_MB = torch.cuda.max_memory_allocated(self.params.device) / 1024 / 1024

        # ✅ 打印最终信息
        logging.info(f"✅ Training Time: {training_time:.2f}s")
        logging.info(f"✅ Validation Eval Time: {eval_time:.2f}s")
        logging.info(f"✅ Test Eval Time: {test_time:.2f}s")
        logging.info(f"✅ Total Time: {total_time:.2f}s")
        logging.info(f"🚀 Peak GPU Memory Usage: {peak_gpu_memory_MB:.2f} MB")

    # def train(self):
    #     self.reset_training_state()

    #     for epoch in range(1, self.params.num_epochs + 1):
    #         time_start = time.time()
            
    #         loss, auc, auc_pr, weight_norm = self.train_epoch()

    #         time_elapsed = time.time() - time_start
    #         logging.info(f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

    #         # if self.valid_evaluator and epoch % self.params.eval_every == 0:
    #         #     result = self.valid_evaluator.eval()
    #         #     logging.info('\nPerformance:' + str(result))
            
    #         #     if result['auc'] >= self.best_metric:
    #         #         self.save_classifier()
    #         #         self.best_metric = result['auc']
    #         #         self.not_improved_count = 0

    #         #     else:
    #         #         self.not_improved_count += 1
    #         #         if self.not_improved_count > self.params.early_stop:
    #         #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
    #         #             break
    #         #     self.last_metric = result['auc']

    #         if epoch % self.params.save_every == 0:
    #             torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def case_study(self):
        self.reset_training_state()
        test_result, save_test_data = self.test_evaluator.print_result()

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
