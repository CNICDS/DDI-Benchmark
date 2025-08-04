import torch
from planetoid import Planetoid
from autoddi.model.gnn_model import GnnModel
import numpy as np
# ==== 1. 参数配置 ====
data_name = "drugbank_S0_2013"                   # 你的数据集名
save_suffix = "_search"              # 与训练时一致
fold = "fold0"
model_id = 0                         # 搜索得到的模型编号
arch = [8, 'MFConv', 'GATConv', 'SAGPool', 'global_add']# 搜索出来的结构
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== 2. 加载图数据 ====
print(f"Loading graph data for {data_name} with fold {fold} and save_suffix {save_suffix}")
graph = Planetoid(data_name=data_name, fold=fold, save_suffix=save_suffix)
print(f"Graph data loaded: {graph}")
# ==== 3. 构造模型并加载权重 ====
print(f"Constructing model with architecture: {arch}")
def get_emb(batch,device,model):
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    pos_emb = model.encode_pair(pos_tri)

    return pos_emb


model = GnnModel(
    arch,
    graph.num_features,
    graph.num_labels,
    graph.rel_total,
    graph.data_name
).to(device)

# model = model_load(graph.data_logger_save, model_id, fold).to(device)
print(f"Loading model from {graph.data_logger_save} with model_id {model_id} and fold {fold}")
model = torch.load("logger/drugbank_S0_2013_search/seed11/model_drugbank_S0_2013_search_num1_fold0.pkl").to(device)
model.eval()

# ==== 4. 收集测试集预测 ====
print("Collecting predictions from test data loader")
emb_list, label_list = [], []
with torch.no_grad():
    for batch in graph.test_data_loader:
        # batch: 调用 do_compute 来获取预测概率
        pos_emb = get_emb(batch, device, model)
        emb_list.append(pos_emb.cpu().numpy())
        label_list.append(np.ones(len(pos_emb)))
emb_arr=np.concatenate(emb_list, axis=0)
label_arr=np.concatenate(label_list, axis=0)

print("Embeddings and labels collected.")
save_emb = 'db_nodeembeddings_0.npy'
np.save(save_emb,   emb_arr)
print(f'✅ Embeddings saved to {save_emb}, shape = {emb_arr.shape}')