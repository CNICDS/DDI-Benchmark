
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import time
import torch

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import csv
import os
import random
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,  average_precision_score, cohen_kappa_score
import torch.utils.data as Data
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import logging
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import UndefinedMetricWarning
from collections import Counter
# 替代写法，自动 fallback
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


def get_drugpair_info(dir,list,drugs):
    with open(dir) as raw_input_file:
        data = csv.reader(raw_input_file, delimiter=',')
        header=next(data)
        for p, r in data:
            list.append([eval(p), eval(r)])
            if eval(p)[0] not in drugs:
                drugs.append(eval(p)[0])
            if eval(p)[1] not in drugs:
                drugs.append(eval(p)[1])
        return list,drugs
        


def feature_vector(feature_path):

    feature = {}
    with open(feature_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头

        for drug_str, emb_str in reader:
            # 尝试把 drug_str eval 成实际类型（有些文件里是 "'ID'" 这种格式）
            try:
                raw = eval(drug_str)
            except:
                raw = drug_str

            # if key in drugs:
            #     # emb_str 里通常是一个列表文本，eval 变成 Python list
            #     feature[key] = eval(emb_str)
            key = f'{raw}'
            feature[key] = eval(emb_str)

    return feature


def load_dataset_from_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过header
        for p, label in reader:
            data.append((eval(p), int(label)))
    return data

def load_all_splits(train_path, valid_path, test_path):
    train_data = load_dataset_from_csv(train_path)
    valid_data = load_dataset_from_csv(valid_path)
    test_data = load_dataset_from_csv(test_path)

    trainX = [x[0] for x in train_data]
    trainY = [x[1] for x in train_data]

    validX = [x[0] for x in valid_data]
    validY = [x[1] for x in valid_data]

    testX = [x[0] for x in test_data]
    testY = [x[1] for x in test_data]

    return trainX, trainY, validX, validY, testX, testY


def create_log_id(dir_path):
    log_count=0
    file_path=os.path.join(dir_path,'log{:d}'.format(log_count))
    while os.path.exists(file_path):
        log_count+=1
        file_path=os.path.join(dir_path,'log{:d}'.format(log_count))
    return log_count


def logging_config(folder=None, name=None, level=logging.DEBUG,console_level=logging.DEBUG,no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder

def early_stopping(recall_list, stopping_steps,min_epoch):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps and min_epoch>60 :
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop

class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()

        self.fea_dim = 197
        self.fc1 = nn.Sequential(nn.Linear(800, 2048), nn.LayerNorm(2048), nn.Dropout(0.2), nn.ReLU(True))
        
        self.gru1 = nn.GRUCell(800, 800)
        self.gru2 = nn.GRUCell(800, 800)

        self.fc = nn.Sequential(nn.Linear(2048, 800), nn.LayerNorm(800), nn.Dropout(0.3), nn.ReLU(True))
        self.fce = nn.Sequential(nn.Linear(400, 400), nn.LayerNorm(400), nn.Dropout(0.1), nn.ReLU(True))
        self.fc_1 = nn.Sequential(nn.Linear(1200, 800), nn.LayerNorm(800), nn.Dropout(0.2), nn.ReLU(True))

        self.softmax=nn.Softmax(dim=1)

        self.fc2 = nn.Sequential(nn.Linear(2048, self.fea_dim))

    def fusion(self, batch_data):
        device = next(self.fc1[0].parameters()).device
        # --------- 新增：准备默认全 0 向量 ---------
        # 假设 drug_emb1、drug_emb2、drug_emb3、drug_emb4 都不为空
        default1 = [0.] * len(next(iter(drug_emb1.values())))
        default2 = [0.] * len(next(iter(drug_emb2.values())))
        default3 = [0.] * len(next(iter(drug_emb3.values())))
        default4 = [0.] * len(next(iter(drug_emb4.values())))
        # --------------------------------------------
        emb1_1, emb1_2 = [], []
        emb2= []
        emb3_1, emb3_2 = [], []
        emb4_1, emb4_2 = [], []
        for i in batch_data:
            # emb1_1.append(drug_emb1[i[0]])
            # emb1_2.append(drug_emb1[i[1]])
            emb1_1.append(drug_emb1.get(i[0], default1))
            emb1_2.append(drug_emb1.get(i[1], default1))
            # emb2.append([*drug_emb2[i[0]],*drug_emb2[i[1]]])
            # emb3_1.append(drug_emb3[i[0]])
            # emb3_2.append(drug_emb3[i[1]])
            # emb4_1.append(drug_emb4[i[0]])
            # emb4_2.append(drug_emb4[i[1]])
            emb2.append([*drug_emb2.get(i[0], default2), *drug_emb2.get(i[1], default2)])
            emb3_1.append(drug_emb3.get(i[0], default3))
            emb3_2.append(drug_emb3.get(i[1], default3))
            emb4_1.append(drug_emb4.get(i[0], default4))
            emb4_2.append(drug_emb4.get(i[1], default4))
        
        emb1_1t = torch.tensor(emb1_1, dtype=torch.float32, device=device)
        emb1_2t = torch.tensor(emb1_2, dtype=torch.float32, device=device)
        emb2t = torch.tensor(emb2, dtype=torch.float32, device=device)
        emb3_1t = torch.tensor(emb3_1, dtype=torch.float32, device=device)
        emb3_2t = torch.tensor(emb3_2, dtype=torch.float32, device=device)
        emb4_1t = torch.tensor(emb4_1, dtype=torch.float32, device=device)
        emb4_2t = torch.tensor(emb4_2, dtype=torch.float32, device=device)

        size=emb1_1t.size(0)
        ft_1=torch.cat((emb1_1t,emb3_1t,emb4_1t),0).to(device)
        ft_2=torch.cat((emb1_2t,emb3_2t,emb4_2t),0).to(device)
        ft_1=self.fce(ft_1)
        ft_2=self.fce(ft_2)

        ft1_1=torch.cat((ft_1[:size],ft_2[:size]-ft_1[:size],ft_1[size:2*size]),1)#n*1200
        ft1_2=torch.cat((ft_2[:size],ft_2[:size]-ft_2[:size],ft_2[size:2*size]),1)#n*1200

        ft2_1 = torch.cat((ft_1[:size], ft_2[:size] - ft_1[:size], ft_1[2*size:]), 1)#n*1200
        ft2_2 = torch.cat((ft_2[:size], ft_2[:size] - ft_2[:size], ft_2[2*size:]), 1)#n*1200

        ft1=ft1_1+ft1_2
        ft2=ft2_1+ft2_2
        ft1 = self.fc_1(ft1)
        ft2=self.fc_1(ft2)#n*1200->n*800

        sf = self.fc(emb2t.to(device))
        ft=torch.stack((ft1,ft2),1)#n*2*800
        ft_0=torch.sum(ft,dim=1)#n*800
        ft_1=self.gru1(ft_0,ft1)
        ft_2=self.gru1(sf,ft_1)
        ft_3=self.gru2(ft_0,ft2)
        ft_4=self.gru2(sf,ft_3)
        ft3=torch.stack((ft_2,ft_4),1)
        attention=self.softmax(ft3)
        feature=torch.sum(ft*attention,dim=1)#n*2*800->n*800
        out=self.fc2(self.fc1(feature))
        return out
        

    def train_DDI_data(self, mode, train_data):
        x = self.fusion(train_data)
        return x

    def test_DDI_data(self, mode, test_data):
        x = self.fusion(test_data)
        sm = nn.Softmax(dim=1)
        pre = sm(x)
        return pre

    def forward(self, mode, *input):
        if mode == 'train':
            return self.train_DDI_data(mode, *input)
        if mode == 'test':
            return self.test_DDI_data(mode, *input)


def calc_metrics(y_true, y_pred, pred_score):

    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    y_true_bi=F.one_hot(y_true.to(torch.int64),num_classes=pred_score.shape[1])
    
    labels_present = set(y_true.tolist())
    # 只用出现类别对应的列来计算AUC
    indices = list(labels_present)
    y_true_sub = y_true_bi[:, indices]
    pred_score_sub = pred_score[:, indices]

    auc_ = roc_auc_score(y_true_sub, pred_score_sub)
    aupr = average_precision_score(y_true_sub, pred_score_sub)


    return acc, macro_precision, macro_recall, macro_f1, kappa, auc_, aupr
    

def pred_tru(loader_test, model):
    with torch.no_grad():
        for i, data in enumerate(loader_test):
            test_x_map = data[0]
            test_x = []
            for k in range(len(test_x_map[0])):
                dp = (test_x_map[0][k], test_x_map[1][k])
                test_x.append(dp)

            if i == 0:
                test_y = data[1]
            else:
                test_y = torch.cat((test_y, data[1]), 0)
            
            out1 = model('test', test_x)
            if i == 0:
                out = out1
            else:
                out = torch.cat((out, out1), 0)
    return out, test_y

def evaluate(loader_test, model):
    model.eval()

    with torch.no_grad():
        out, test_y = pred_tru(loader_test, model)

        prediction = torch.max(out, 1)[1]
        # prediction = prediction.cuda().data.cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        out = out.cuda().data.cpu().numpy()

        acc, macro_precision, macro_recall, macro_f1, kappa, auc_, aupr = calc_metrics(test_y, prediction, out)
        return macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr
        

def pos_weight():
    data1 = []
    with open(filename[0]) as f2:
        data2 = csv.reader(f2)
        header=next(data2)
        for i, j in data2:
            data1.append(eval(j))
    data3 = torch.Tensor(data1)
    num = data3.size(0)
    posn = torch.sum(data3, 0)
    numn = torch.full_like(posn, num)
    pos_weight = torch.div(numn - posn, posn).to('cuda')
    return pos_weight

def Train(trainX, trainY, validX, validY, testX, testY,batch_size=2000,n_epoch=100):
    seed = 226
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = 'log\\'
    logging_config(folder=save_dir, name=f's0_{seed}_db_new', no_console=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    acc_list = []
    kappa_list=[]
    auc_list = []
    aupr_list = []

    time0 = time.time()
        #'''
    model=M2()
    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    
    trainset = [[trainX[j], trainY[j]] for j in range(len(trainX))]
    loader_train = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    validset = [[validX[j], validY[j]] for j in range(len(validX))]
    loader_valid = Data.DataLoader(dataset=validset, batch_size=batch_size, shuffle=False)

    testset = [[testX[j], testY[j]] for j in range(len(testX))]
    loader_test = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    best_epoch = -1
    val_list = []
    for epoch in range(1,n_epoch+1):
        model.train()

        ddi_total_loss=0
        for step,tdata in enumerate(loader_train):
            iter=step+1
            time2=time.time()
            train_x_map = tdata[0]
            train_y=tdata[1]               

            if use_cuda:
                train_y = train_y.to(device)
            train_x = []
            for ii in range(len(train_x_map[0])):
                dp = (train_x_map[0][ii], train_x_map[1][ii])
                train_x.append(dp)
            out=model('train',train_x)

            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(out, train_y.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ddi_total_loss+=loss.item()
            if (iter%100)==0:
                logging.info('DDI Training: Epoch {:04d} Iter {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'
                        .format(epoch, iter, time.time() - time2, loss.item(), ddi_total_loss / iter))
        scheduler.step()

        time3 = time.time()

        macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr = evaluate(loader_valid, model)
        logging.info(
            'DDI Evaluation:Total Time {:.1f}s | Macro Precision {:.4f} | Macro Recall {:.4f} | Macro F1 {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
                .format(time.time() - time3, macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr))
        val_list.append(macro_precision+macro_recall+macro_f1+acc+ kappa+ auc_+ aupr)
        best_acc, should_stop = early_stopping(val_list, 10,epoch)

        if val_list.index(best_acc) == len(val_list) - 1:
            PATH = f's0_{seed}_db_new.pth'
            torch.save(model.state_dict(), PATH)
            best_epoch = epoch

    #test:
    # PATH = 'best_model_epoch.pth'
        
    model.load_state_dict(torch.load(PATH))
    model.to(device)
        
    time5 = time.time()
    macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr = evaluate(loader_test, model)
    logging.info(
            'DDI Test:Total Time {:.1f}s | Macro Precision {:.4f} | Macro Recall {:.4f} | Macro F1 {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
                .format(time.time() - time5, macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr))


if __name__=='__main__':


    filename = ['./MCFF-MTDDI-main/Multi-Class-data1/db2025/data.csv',
                     'input-features/drug initial embedding representations.csv','input-features/Morgan fingerprint vectors.csv',
                    'input-features/drug subgraph mean representations.csv','input-features/drug subgraph frequency representations.csv']
    
    train_csv = './MCFF-MTDDI-main/Multi-Class-data1/db2025/train_s0.csv'
    valid_csv = './MCFF-MTDDI-main/Multi-Class-data1/db2025/valid_s0.csv'
    test_csv = './MCFF-MTDDI-main/Multi-Class-data1/db2025/test_s0.csv'

    drug_emb1 = feature_vector(filename[1])
    drug_emb2 = feature_vector(filename[2])
    drug_emb3 = feature_vector(filename[3])
    drug_emb4 = feature_vector(filename[4])

    trainX, trainY, validX, validY, testX, testY = load_all_splits(train_csv, valid_csv, test_csv)
    
    Train(trainX, trainY, validX, validY, testX, testY,batch_size=1024, n_epoch=100)
