# -*- coding: utf-8 -*-
# @Time    : 2023/5/28 下午6:23
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : tiger.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.nn import BCEWithLogitsLoss, Linear
import math
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree
from .GraphTransformer import GraphTransformer
import os



class NodeFeatures(torch.nn.Module):
    def __init__(self, degree, feature_num, embedding_dim, layer=2, type='graph'):
        super(NodeFeatures, self).__init__()

        # todo
        self.embedding_dim = embedding_dim  # 将embedding_dim保存为类的实例属性  
        
        if type == 'graph': ##代表有feature num
            self.node_encoder = Linear(feature_num, embedding_dim)
        else:
            self.node_encoder = torch.nn.Embedding(feature_num, embedding_dim)

        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim, padding_idx=0)  ##将度的值映射成embedding
        self.apply(lambda module: init_params(module, layers=layer))

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.degree_encoder.reset_parameters()

    def forward(self, data):
        # todo
        # 检查edge_index是否为空  
        if data.edge_index.numel() == 0:  
            # 返回一个零向量或其他默认值，取决于模型的需求  
            # 这里假设你需要同样大小的零向量作为默认输出，可以根据输入x的尺寸返回零向量  
            node_feature = torch.zeros((data.x.size(0), self.embedding_dim), dtype=data.x.dtype, device=data.x.device)  
            return node_feature  

        row, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=data.x.dtype)
        node_feature = self.node_encoder(data.x)
        node_feature += self.degree_encoder(x_degree.long())

        return node_feature

class TIGER(torch.nn.Module):
    def __init__(self, max_layer = 6, num_features_drug = 78, num_nodes = 200, num_relations_mol = 10, num_relations_graph = 10, output_dim=64, max_degree_graph=100, max_degree_node=100, sub_coeff = 0.2, mi_coeff = 0.5, dropout=0.2, device = 'cuda'):
        super(TIGER, self).__init__()

        print("TIGER Loaded")
        #todo
        self.output_dim = output_dim

        self.device = device

        self.layers = max_layer
        self.num_features_drug = num_features_drug

        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node

        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout

        self.mol_atom_feature = NodeFeatures(degree=max_degree_graph, feature_num=num_features_drug, embedding_dim=output_dim, type='graph')
        self.drug_node_feature = NodeFeatures(degree=max_degree_node, feature_num=num_nodes, embedding_dim=output_dim, type='node')

        ##学习的模块
        self.mol_representation_learning = GraphTransformer(layer_num = max_layer, embedding_dim = output_dim, num_heads = 4, num_rel = num_relations_mol, dropout= dropout, type='graph')
        self.node_representation_learning = GraphTransformer(layer_num = max_layer, embedding_dim = output_dim, num_heads = 4, num_rel = num_relations_graph, dropout=dropout, type='node')
        ##Net用统一的代码就可以了，用type指示是哪种类型的学习，或者分开两个模块，然后两个模块里面集合一些公共的模块

        self.fc1 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()

    def to(self, device):

        self.mol_atom_feature.to(device)
        self.drug_node_feature.to(device)

        self.mol_representation_learning.to(device)
        self.node_representation_learning.to(device)

        self.fc1.to(device)
        self.fc2.to(device)

        self.disc.to(device)
        self.b_xent.to(device)

    def reset_parameters(self):

        self.mol_atom_feature.reset_parameters()
        self.drug_node_feature.reset_parameters()

        self.mol_representation_learning.reset_parameters()
        self.node_representation_learning.reset_parameters()


    def forward(self, drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph):

        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        ##在获得了特征之后，就是学习相应的表示，一个是节点级别的表示，一个是图级别的表示！！
        mol1_graph_embedding, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        # todo
        # 用零向量填充空输入  
        def safe_representation_learning(node_feature, subgraph):  
            if node_feature.numel() == 0 or subgraph.edge_index.numel() == 0:  
                # 从mol_graph_embedding获取batch_size  
                batch_size = mol1_graph_embedding.size(0)  
                embedding_dim = self.output_dim # 使用模型定义的hidden_dim  
                empty_embedding = torch.zeros((batch_size, embedding_dim),   
                                        device=mol1_graph_embedding.device)  
                empty_attn = torch.zeros((batch_size, 1),   
                                    device=mol1_graph_embedding.device)  
                return empty_embedding, empty_embedding, empty_attn  
            return self.node_representation_learning(node_feature, subgraph)

        drug1_node_embedding, drug1_sub_embedding, drug1_attn = safe_representation_learning(drug1_node_feature, drug1_subgraph)  
        drug2_node_embedding, drug2_sub_embedding, drug2_attn = safe_representation_learning(drug2_node_feature, drug2_subgraph)  
        # 确保维度匹配后再concat  
        assert drug1_node_embedding.size(0) == mol1_graph_embedding.size(0), \
           f"Batch size mismatch: {drug1_node_embedding.size(0)} vs {mol1_graph_embedding.size(0)}" 
        # drug1_node_embedding, drug1_sub_embedding, drug1_attn = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
        # drug2_node_embedding, drug2_sub_embedding, drug2_attn = self.node_representation_learning(drug2_node_feature, drug2_subgraph)

        # todo 
        # print(f"drug1_node_embedding shape: {drug1_node_embedding.shape}")  
        # print(f"mol1_graph_embedding shape: {mol1_graph_embedding.shape}")

        drug1_embedding = self.fc1(torch.concat([drug1_node_embedding,mol1_graph_embedding],dim=-1))
        drug2_embedding = self.fc1(torch.concat([drug2_node_embedding, mol2_graph_embedding], dim=-1))

        score = self.fc2(torch.concat([drug1_embedding, drug2_embedding], dim=-1))

        loss_s_m = self.loss_MI(self.MI(drug1_embedding, mol1_atom_embedding)) + self.loss_MI(self.MI(drug2_embedding, mol2_atom_embedding))
        loss_s_d = self.loss_MI(self.MI(drug1_embedding, drug1_sub_embedding)) + self.loss_MI(self.MI(drug2_embedding, drug2_sub_embedding))


        predicts_drug = F.log_softmax(score, dim=-1)
        loss_label = F.nll_loss(predicts_drug, drug1_mol.y.view(-1))

        loss = loss_label + self.mol_coeff* loss_s_m + self.mi_coeff * loss_s_d

        return torch.exp(predicts_drug)[:,1], loss

    def MI(self, graph_embeddings, sub_embeddings):  
        # 检查 sub_embeddings 是否为空  
        if len(sub_embeddings) == 0:  
            # 创建一个全零的 sub_embeddings，占位符，形状为 [batch_size, embedding_dim]  
            batch_size, embedding_dim = graph_embeddings.size(0), graph_embeddings.size(1)  
            sub_embeddings = [torch.zeros((1, embedding_dim), device=graph_embeddings.device) for _ in range(batch_size)]  
        else:  
            # 确保每个 sub_embeddings 都是 2D 张量  
            sub_embeddings = [sub.unsqueeze(0) if sub.dim() == 1 else sub for sub in sub_embeddings]  

        # 找到当前批次中 sub_embeddings 的最大长度  
        max_length = max(sub.size(0) for sub in sub_embeddings) if sub_embeddings else 1  

        padded_sub_embeddings = []  
        for sub in sub_embeddings:  
            current_length, embedding_dim = sub.size(0), sub.size(1)  
            if current_length < max_length:  
                # 填充 zeros，保持维度为 [max_length, embedding_dim]  
                padding = torch.zeros((max_length - current_length, embedding_dim), device=sub.device)  
                padded_sub = torch.cat([sub, padding], dim=0)  
            else:  
                # 如果长度超过最大长度，裁剪到 max_length  
                padded_sub = sub[:max_length, :]  
            padded_sub_embeddings.append(padded_sub)  

        # 堆叠所有填充后的 sub_embeddings，形状为 [batch_size, max_length, embedding_dim]  
        sub = torch.stack(padded_sub_embeddings, dim=0)  

        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)  
        idx[len(idx) // 2] = idx[len(idx) // 2 + 1]  
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))  

        c_0_list, c_1_list = [], []  
        for c_0, c_1, sub_embed in zip(graph_embeddings, shuffle_embeddings, padded_sub_embeddings):  
            c_0_list.append(c_0.expand_as(sub_embed))  # pos  
            c_1_list.append(c_1.expand_as(sub_embed))  # neg  

        # 连接 pos 和 neg 的 embedding  
        c_0 = torch.cat(c_0_list, dim=0)  
        c_1 = torch.cat(c_1_list, dim=0)  

        return self.disc(sub, c_0, c_1)
    # def MI(self, graph_embeddings, sub_embeddings):
    #     # todo
    #     if len(sub_embeddings) == 0:  # 检查 sub_embeddings 是否为空  
    #         # 如果为空，创建一个适当大小的空张量  
    #         batch_size, embedding_dim = graph_embeddings.size(0), graph_embeddings.size(1)  
    #         sub_embeddings = [torch.zeros((batch_size, embedding_dim), device=graph_embeddings.device)]

    #     idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
    #     idx[len(idx) // 2] = idx[len(idx) // 2 + 1]
    #     shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
    #     c_0_list, c_1_list = [], []
    #     for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
    #         c_0_list.append(c_0.expand_as(sub)) ##pos
    #         c_1_list.append(c_1.expand_as(sub)) ##neg

    #     # c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
    #     # todo
    #     # Ensure sub_embeddings is treated as a list  
    #     sub = torch.cat([sub.unsqueeze(0) for sub in sub_embeddings], dim=0)  
    #     c_0 = torch.cat(c_0_list, dim=0)  
    #     c_1 = torch.cat(c_1_list, dim=0)  

    #     return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits):

        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(self.device)

        return self.b_xent(logits.view([1,-1]), lbl.view([1, -1]))

    def save(self, path):
        save_path = os.path.join(path, self.__class__.__name__+'.pt')
        torch.save(self.state_dict(), save_path)
        return save_path

    # UniDDI
    def encode_pair(self, drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph):
        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        ##在获得了特征之后，就是学习相应的表示，一个是节点级别的表示，一个是图级别的表示！！
        mol1_graph_embedding, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        # todo
        # 用零向量填充空输入  
        def safe_representation_learning(node_feature, subgraph):  
            if node_feature.numel() == 0 or subgraph.edge_index.numel() == 0:  
                # 从mol_graph_embedding获取batch_size  
                batch_size = mol1_graph_embedding.size(0)  
                embedding_dim = self.output_dim # 使用模型定义的hidden_dim  
                empty_embedding = torch.zeros((batch_size, embedding_dim),   
                                        device=mol1_graph_embedding.device)  
                empty_attn = torch.zeros((batch_size, 1),   
                                    device=mol1_graph_embedding.device)  
                return empty_embedding, empty_embedding, empty_attn  
            return self.node_representation_learning(node_feature, subgraph)

        drug1_node_embedding, drug1_sub_embedding, drug1_attn = safe_representation_learning(drug1_node_feature, drug1_subgraph)  
        drug2_node_embedding, drug2_sub_embedding, drug2_attn = safe_representation_learning(drug2_node_feature, drug2_subgraph)  
        # 确保维度匹配后再concat  
        assert drug1_node_embedding.size(0) == mol1_graph_embedding.size(0), \
           f"Batch size mismatch: {drug1_node_embedding.size(0)} vs {mol1_graph_embedding.size(0)}" 
        # drug1_node_embedding, drug1_sub_embedding, drug1_attn = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
        # drug2_node_embedding, drug2_sub_embedding, drug2_attn = self.node_representation_learning(drug2_node_feature, drug2_subgraph)

        # todo 
        # print(f"drug1_node_embedding shape: {drug1_node_embedding.shape}")  
        # print(f"mol1_graph_embedding shape: {mol1_graph_embedding.shape}")

        drug1_embedding = self.fc1(torch.concat([drug1_node_embedding,mol1_graph_embedding],dim=-1))
        drug2_embedding = self.fc1(torch.concat([drug2_node_embedding, mol2_graph_embedding], dim=-1))
        pair_embedding = torch.cat([drug1_embedding, drug2_embedding], dim=-1)  # shape: [B, 2*dim]

        return pair_embedding
    def encode_node(self, drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph):
        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        ##在获得了特征之后，就是学习相应的表示，一个是节点级别的表示，一个是图级别的表示！！
        mol1_graph_embedding, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        # todo
        # 用零向量填充空输入  
        def safe_representation_learning(node_feature, subgraph):  
            if node_feature.numel() == 0 or subgraph.edge_index.numel() == 0:  
                # 从mol_graph_embedding获取batch_size  
                batch_size = mol1_graph_embedding.size(0)  
                embedding_dim = self.output_dim # 使用模型定义的hidden_dim  
                empty_embedding = torch.zeros((batch_size, embedding_dim),   
                                        device=mol1_graph_embedding.device)  
                empty_attn = torch.zeros((batch_size, 1),   
                                    device=mol1_graph_embedding.device)  
                return empty_embedding, empty_embedding, empty_attn  
            return self.node_representation_learning(node_feature, subgraph)

        drug1_node_embedding, drug1_sub_embedding, drug1_attn = safe_representation_learning(drug1_node_feature, drug1_subgraph)  
        drug2_node_embedding, drug2_sub_embedding, drug2_attn = safe_representation_learning(drug2_node_feature, drug2_subgraph)  
        # 确保维度匹配后再concat  
        assert drug1_node_embedding.size(0) == mol1_graph_embedding.size(0), \
           f"Batch size mismatch: {drug1_node_embedding.size(0)} vs {mol1_graph_embedding.size(0)}" 
        # drug1_node_embedding, drug1_sub_embedding, drug1_attn = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
        # drug2_node_embedding, drug2_sub_embedding, drug2_attn = self.node_representation_learning(drug2_node_feature, drug2_subgraph)

        # todo 
        # print(f"drug1_node_embedding shape: {drug1_node_embedding.shape}")  
        # print(f"mol1_graph_embedding shape: {mol1_graph_embedding.shape}")

        drug1_embedding = self.fc1(torch.concat([drug1_node_embedding,mol1_graph_embedding],dim=-1))
        drug2_embedding = self.fc1(torch.concat([drug2_node_embedding, mol2_graph_embedding], dim=-1))

        return drug1_embedding


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c: 1, 512; h_pl: 1, 2708, 512; h_mi: 1, 2708, 512
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)
        # todo 
        # 检查 h_pl 和 h_mi 是否为空  
        if h_pl.numel() == 0 or h_mi.numel() == 0:  
            # 如果任何输入是空子图则返回全零 logits  
            return torch.zeros(0, device=c.device)  
        
        # c_x = c
        # todo
        # 确保 c_x 的维度与 h_pl 和 h_mi 匹配  
        # c_x = c.unsqueeze(1)  # 扩展到 (1, 1, 512)  

        # 确保 c_x 的维度与 h_pl 和 h_mi 匹配  
        if len(c.shape) == 4:  
            # 如果 c 是 4D，将其重新排列为 2D  
            # 原形状 [1, 128, 101, 64] -> [128*101, 64]  
            c_x = c.view(-1, c.shape[-1])  # [128*101, 64]  
        elif len(c.shape) == 3:  
            if c.shape[1] == 1:  
                # 如果第二个维度是1，例如 [128, 1, 64]，移除该维度  
                c_x = c.squeeze(1)  # [128, 64]  
            else:  
                # 否则，将其重新排列为 2D  
                c_x = c.view(-1, c.shape[-1])  # 根据具体情况调整  
        elif len(c.shape) == 2:  
            # 如果 c 是 2D，例如 [128, 64]  
            c_x = c  
        else:  
            raise ValueError(f"Unsupported c shape: {c.shape}")  
        
        # print(f"c_x shape: {c_x.shape}")  
        # print(f"h_pl shape: {h_pl.shape}")  
        # print(f"h_mi shape: {h_mi.shape}")

        # 确保 c_x 的第一个维度与 h_pl 和 h_mi 匹配  
        if c_x.shape[0] != h_pl.shape[0]:  
            if c_x.shape[0] * c_x.shape[1] == h_pl.shape[0]:  
                # 假设 c_x 是 [128, 101, 64] 或类似形状  
                c_x = c_x.view(-1, c_x.shape[-1])  # [12928, 64]  
            else:  
                raise ValueError(f"c_x 的第一个维度 {c_x.shape[0]} 不能与 h_pl 的第一维 {h_pl.shape[0]} 匹配")   

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # logits = torch.cat((sc_1, sc_2), 0)
        # todo
        # 将 sc_1 和 sc_2 拼接  
        logits = torch.cat((sc_1, sc_2), dim=0)  # 输出形状: (2, 2708, 1)  

        return logits



def init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)