
import argparse
import pandas as pd
import torch
import torch.utils.data as Data
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
# from gensim.models import KeyedVectors

import linecache
import re
from string import punctuation
import nltk
from nltk.tokenize import sent_tokenize
import torch.optim as optim
import math
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig
import os
import random

# import xlrd
# import time

# nltk.download('punkt_tab')

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

config_path = 'upload/model/pretrained_models/biobert-base-cased-v1.2/config.json'
model_path = 'upload/model/pretrained_models/biobert-base-cased-v1.2'
vocab_path = 'upload/model/pretrained_models/biobert-base-cased-v1.2/vocab.txt'

tokenizer = BertTokenizer.from_pretrained(vocab_path)

PAD, CLS = '[PAD]', '[CLS]'
sequence_length = 128
BATCH_SIZE = 32

sentences = []
sentences_id = []
label = []
atten_masks = []

e1_pos = []
e2_pos = []
dist_ra_e1 = []
dist_ra_e2 = []

article_num_list = []
sentence_num_list = []

text_DB = []

e1_DB_text_id = []
e2_DB_text_id = []
atten_masks_DB_e1 = []
atten_masks_DB_e2 = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def attention_masks(ids):
    id_mask = [int(i > 0) for i in ids]
    return id_mask

def pos(x):
    if x < -sequence_length:
        return 0
    if sequence_length >= x >= -sequence_length:
        return x + sequence_length + 1
    if x > sequence_length:
        return (sequence_length + 1) * 2

def load_data(file):
    total_line = len(open(file).readlines())

    for i in range(total_line):
        line = linecache.getline(file, i + 1)
        line = line.strip()
        list_1 = [j.start() for j in re.finditer('=', line)]
        list_2 = [j.start() for j in re.finditer('\$', line)]
        type = line[list_1[2] + 1: list_2[2]]

        if type.lower() == 'false':
            label.append(0.0)
        elif type.lower() == 'effect':
            label.append(1.0)
        elif type.lower() == 'advise':
            label.append(2.0)
        elif type.lower() == 'mechanism':
            label.append(3.0)
        elif type.lower() == 'int':
            label.append(4.0)
        else:
            label.append(0.0)

        article_num = int(line[list_1[-2] + 1: list_2[-2]])
        sentence_num = int(line[list_1[-1] + 1: list_2[-1]])

        article_temp = []
        article_temp.append(article_num)
        article_num_list.append(article_temp * 128)

        sentence_temp = []
        sentence_temp.append(sentence_num)
        sentence_num_list.append(sentence_temp * 128)

        sentence_raw = line[list_1[3] + 1: list_2[3]]
        sentence_raw = sentence_raw.lower()

        dicts = {x: ' ' for x in punctuation}
        punc_table = str.maketrans(dicts)
        sentence_raw = sentence_raw.translate(punc_table)
        sentence_split = nltk.word_tokenize(sentence_raw)
        sentence_split_new = [x for x in sentence_split if x != '']
        sentence_content = ' '.join(sentence_split_new)

        token = tokenizer.tokenize(sentence_content)
        sentences.append(token)

        pos_1 = sentence_split_new.index('e1')
        pos_2 = sentence_split_new.index('e2')

        e1_pos.append(pos_1)
        e2_pos.append(pos_2)
        dis_curr_line_1 = []
        dis_curr_line_2 = []
        for j in range(0, sequence_length):
            e_r_1 = j - pos_1
            e_r_2 = j - pos_2
            dis_curr_line_1.append(pos(e_r_1))
            dis_curr_line_2.append(pos(e_r_2))
        dist_ra_e1.append(dis_curr_line_1)
        dist_ra_e2.append(dis_curr_line_2)

        input_ids = tokenizer.encode(token)
        len_ids = len(input_ids)

        if sequence_length > len_ids:
            for j in range(sequence_length - len_ids):
                input_ids.append(0)
        elif sequence_length < len_ids:
            for k in range(len_ids - sequence_length):
                del(input_ids[-1])

        sentences_id.append(input_ids)

        input_mask = attention_masks(input_ids)
        atten_masks.append(input_mask)

def load_data_csv(file):
    df = pd.read_csv(file)

    for idx, row in df.iterrows():
        type = row['type']

        # if type.lower() == 'false':
        #     label.append(0.0)
        # elif type.lower() == 'effect':
        #     label.append(1.0)
        # elif type.lower() == 'advise':
        #     label.append(2.0)
        # elif type.lower() == 'mechanism':
        #     label.append(3.0)
        # elif type.lower() == 'int':
        #     label.append(4.0)
        # else:
        #     label.append(0.0)
        label.append(int(type))

        article_num = int(row['article_num'])
        sentence_num = int(row['sentence_num'])

        article_temp = []
        article_temp.append(article_num)
        article_num_list.append(article_temp * 128)

        sentence_temp = []
        sentence_temp.append(sentence_num)
        sentence_num_list.append(sentence_temp * 128)

        # if "#Drug1" not in row['sentence']:
        #     row['sentence'] = row['sentence'] + ' #Drug1'
        # if "#Drug2" not in row['sentence']:
        #     row['sentence'] = row['sentence'] + ' #Drug2'

        # sentence_raw = row['sentence'].replace('#Drug1', 'e1').replace('#Drug2', 'e2')
        sentence_raw = "e1: " + str(row['e1 smiles']).lower() + "; e2:" + str(row['e1 smiles']).lower()
        sentence_raw = sentence_raw.lower().replace(row['e1'].lower(), 'e1').replace(row['e2'].lower(), 'e2')

        dicts = {x: ' ' for x in punctuation}
        punc_table = str.maketrans(dicts)
        sentence_raw = sentence_raw.translate(punc_table)
        sentence_split = nltk.word_tokenize(sentence_raw)
        sentence_split_new = [x for x in sentence_split if x != '']
        sentence_content = ' '.join(sentence_split_new)

        token = tokenizer.tokenize(sentence_content)
        sentences.append(token)

        pos_1 = sentence_split_new.index('e1')
        pos_2 = sentence_split_new.index('e2')

        e1_pos.append(pos_1)
        e2_pos.append(pos_2)
        dis_curr_line_1 = []
        dis_curr_line_2 = []
        for j in range(0, sequence_length):
            e_r_1 = j - pos_1
            e_r_2 = j - pos_2
            dis_curr_line_1.append(pos(e_r_1))
            dis_curr_line_2.append(pos(e_r_2))
        dist_ra_e1.append(dis_curr_line_1)
        dist_ra_e2.append(dis_curr_line_2)

        input_ids = tokenizer.encode(token)
        len_ids = len(input_ids)

        if sequence_length > len_ids:
            for j in range(sequence_length - len_ids):
                input_ids.append(0)
        elif sequence_length < len_ids:
            for k in range(len_ids - sequence_length):
                del(input_ids[-1])

        sentences_id.append(input_ids)

        input_mask = attention_masks(input_ids)
        atten_masks.append(input_mask)

def gen_dataloader():
    for i in range(0, len(sentences_id)):
        if len(sentences_id[i]) > sequence_length:
            for k in range(len(sentences_id[i]) - sequence_length):
                del(sentences_id[i][-1])
                del(atten_masks[i][-1])

    all_index_ten = torch.LongTensor(sentences_id)
    dist_ra_e1_ten = torch.LongTensor(dist_ra_e1)
    dist_ra_e2_ten = torch.LongTensor(dist_ra_e2)
    article_num_ten = torch.LongTensor(article_num_list)
    sentence_num_ten = torch.LongTensor(sentence_num_list)
    real_input = torch.cat((all_index_ten, dist_ra_e1_ten, dist_ra_e2_ten, article_num_ten, sentence_num_ten), axis=1)
    attention_tokens = torch.LongTensor(atten_masks)
    real_target = torch.LongTensor(label)
    torch_dataset = Data.TensorDataset(real_input, attention_tokens, real_target)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    return loader

def data_unpack(cat_data, N):
    list_x = torch.split(cat_data, [N, N, N, N, N], 1)
    x = list_x[0]
    e_r_1 = list_x[1]
    e_r_2 = list_x[2]
    article_lable = list_x[3]
    sentence_lable = list_x[4]

    return x, e_r_1, e_r_2, article_lable, sentence_lable

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Model(nn.Module):
    def __init__(self, flag_class=86):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dist_embedding = nn.Embedding(1000, 10)
        self.dist_embedding.weight.data.uniform_(-1e-3, 1e-3)

        kernel_sizes = [5]

        self.hidden_size = 404
        dropout_1 = 0.5
        dropout_2 = 0.1
        embedding_dim = 808
        flag_class = flag_class
        hidden_size = 768

        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, bidirectional=True)
        self.conv_list = nn.ModuleList([nn.Conv1d(self.hidden_size * 2,
                                                  hidden_size, w, padding=(w - 1) // 2) for w in kernel_sizes])
        self.classifier = nn.Linear(len(kernel_sizes) * hidden_size, flag_class)
        self.dropout_1 = nn.Dropout(dropout_1)
        self.dropout_2 = nn.Dropout(dropout_2)

    def forward(self, x_in, e_r_1_in, e_r_2_in, article_lable_in, sentence_lable_in, masks):
        dist_1_emb = self.dist_embedding(e_r_1_in)
        dist_2_emb = self.dist_embedding(e_r_2_in)
        article_lable_emb = self.dist_embedding(article_lable_in)
        sentence_lable_emb = self.dist_embedding(sentence_lable_in)

        out = self.bert(x_in, attention_mask=masks)
        input_0 = out[0]
        x_cat = torch.cat((input_0, dist_1_emb, dist_2_emb, article_lable_emb, sentence_lable_emb), 2)
        input_1 = x_cat.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(2, len(x_in), self.hidden_size).cuda())
        cell_state = Variable(torch.zeros(2, len(x_in), self.hidden_size).cuda())
        output_1, (final_hidden_state, final_cell_state) = self.lstm(input_1, (hidden_state, cell_state))
        output_2 = output_1.permute(1, 0, 2)

        conv_outputs = []
        for c in self.conv_list:
            conv_output = c(output_2.transpose(1, 2))
            conv_output = gelu(conv_output)
            conv_output, _ = torch.max(conv_output, -1)
            conv_outputs.append(conv_output)
        pooled_output = torch.cat(conv_outputs, 1)
        pooled_output = self.dropout_2(pooled_output)

        logits = self.classifier(pooled_output)

        return logits
        del x_in
        del e_r_1_in
        del e_r_2_in
        del article_lable_in
        del sentence_lable_in
        del masks
        del e1_text
        del e2_text
        del e1_text_mask
        del e2_text_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--dataset_name', type=str, default='S0', help='dataset name')

    args = parser.parse_args()
    setup_seed(args.seed)
    file_name = f'upload/data/IK-DDI/drugbank2025/{args.dataset_name}/train_ddi.csv'
    # file_name = f'/home/lipengjiang/Project/ddi/datasets/new/drugbank2025/{args.dataset_name.lower()}.csv'
    load_data_csv(file_name)
    loader = gen_dataloader()

    model = Model(flag_class=197)
    model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, eps=1e-8)

    model.train()

    # Training
    for epoch in range(3):
        print('\nepoch: {}'.format(epoch + 1))
        print('\nepoch: ' + str(epoch + 1))
        print('*' * 10)
        print('\n************')
        running_loss = 0
        sum_train = 0
        corrects_train = 0
        size_train = 0

        for i, (real_input, attention_tokens, real_target) in enumerate(loader):
            optimizer.zero_grad()

            real_target = real_target.cuda()
            real_input = real_input.cuda()
            attention_tokens = attention_tokens.cuda()

            x, e_r_1, e_r_2, article_lable, sentence_lable = data_unpack(real_input, sequence_length)
            out = model(x, e_r_1, e_r_2, article_lable, sentence_lable, attention_tokens)
            corrects_train_cur = (torch.max(out, 1)[1].view(real_target.size()) == real_target).sum()
            corrects_train += corrects_train_cur

            size_train = size_train + len(real_input)
            loss = criterion(out, real_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            sum_train += 1

        accuracy_train = 100.0 * corrects_train / size_train
        loss_average = running_loss / sum_train
        print('epoch %d' % (i), '结束，loss平均为：%f' % (running_loss / sum_train), 'acc = %f' % (accuracy_train))
        print('\nepoch' + str(epoch + 1))
        print(' end, avergae loss is:' + str(loss_average))
        print(' acc = ' + str(accuracy_train))

    print('\n')

    print("\n\nTotal number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    del loader
    del real_input
    del attention_tokens
    del corrects_train_cur
    del article_num_list
    del sentence_num_list

    # Test
    corrects, avg_loss = 0, 0
    corrects_test = 0
    sum_test = 0

    sentences = []
    sentences_id = []
    label = []
    atten_masks = []

    e1_pos = []
    e2_pos = []
    dist_ra_e1 = []
    dist_ra_e2 = []

    article_num_list = []
    sentence_num_list = []

    file_name = f'upload/data/IK-DDI/drugbank2025/{args.dataset_name}/test_ddi.csv'
    # file_name = f'/home/lipengjiang/Project/ddi/datasets/new/drugbank2025/{args.dataset_name.lower()}_test.csv'
    load_data_csv(file_name)
    loader = gen_dataloader()

    # 测试阶段开始
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    for i, (input_test_unit, attention_test, target_test_unit) in enumerate(loader):
        target_test_unit = target_test_unit.cuda()
        input_test_unit = input_test_unit.cuda()
        attention_test = attention_test.cuda()

        x, e_r_1, e_r_2, article_lable, sentence_lable = data_unpack(input_test_unit, sequence_length)
        out_test = model(x, e_r_1, e_r_2, article_lable, sentence_lable, attention_test)

        if i == 0:
            out_test_matrix = out_test
            out_lable_matrix = target_test_unit
        elif i > 0:
            out_test_matrix = torch.cat((out_test_matrix, out_test), 0)
            out_lable_matrix = torch.cat((out_lable_matrix, target_test_unit), 0)

    y_pred = torch.max(out_test_matrix, 1)[1].view(out_lable_matrix.size())
    y_true = out_lable_matrix

    y_pred = y_pred.cuda().cpu()
    y_true = y_true.cuda().cpu()

    acc = accuracy_score(y_true, y_pred)
    micro_p = precision_score(y_true, y_pred, average='micro')
    micro_r = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    # save_path = f"/home/lipengjiang/Project/ddi/IK-DDI/log_drugbank2025/{args.dataset_name}/{args.seed}.log"
    save_path = f"upload/model/IK-DDI/log2025/{args.dataset_name}/{args.seed}.log"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(' micro_p = ' + str(micro_p) + 'micro_r' + str(micro_r) + 'micro_f1' + str(micro_f1))
    print('\nmicro_p = ' + str(micro_p) + 'micro_r' + str(micro_r) + 'micro_f1' + str(micro_f1))

    macro_p = precision_score(y_true, y_pred, average='macro')
    macro_r = recall_score(y_true, y_pred, average='macro')
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)

    kappa = metrics.cohen_kappa_score(y_true, y_pred)

    print(' macro_p = ' + str(macro_p) + 'macro_r = ' + str(macro_r) + 'macro_f1 = ' + str(macro_f1))
    print('\nmacro_p = ' + str(macro_p) + 'macro_r = ' + str(macro_r) + 'macro_f1 = ' + str(macro_f1))

    # target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    print(classification_report(y_true, y_pred, digits=4))

    with open(save_path, 'a') as file:
        file.write('micro_p = ' + str(micro_p) + 'micro_r = ' + str(micro_r) + 'micro_f1 = ' + str(micro_f1) + '\n')
        file.write('macro_p = ' + str(macro_p) + 'macro_r = ' + str(macro_r) + 'macro_f1 = ' + str(macro_f1) + '\n')
        file.write('accuracy = ' + str(acc) + '\n')
        file.write('kappa = ' + str(kappa) + '\n')
        file.write('classification_report:\n')
        file.write(str(classification_report(y_true, y_pred, digits=4)))
        # file_1.close()

    # print(str(classification_report(y_true, y_pred, target_names=target_names, digits=4)))
    # file_1.close()
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    # 转换为DataFrame
    df_report = pd.DataFrame(report_dict).transpose()

    # 保存为 CSV 文件
    df_report.to_csv(save_path.replace('.log', '_report.csv'), index=False)
    # checkpoint_path = f"/home/lipengjiang/Project/ddi/IK-DDI/log_drugbank2025/{args.dataset_name}/{args.seed}.pth"
    checkpoint_path = f"upload/model/IK-DDI/log2025/{args.dataset_name}/{args.seed}.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    conf_matrix = confusion_matrix(y_true, y_pred)
    pd.DataFrame(conf_matrix).to_csv(save_path.replace('.log', '_confusion_matrix.csv'), index=False)

    print("Done!")
