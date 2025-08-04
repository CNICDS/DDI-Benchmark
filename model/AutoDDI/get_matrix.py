import torch
from planetoid import Planetoid
from autoddi.model.gnn_model import GnnModel
from autoddi.model.logger import gnn_architecture_performance_save, model_save, model_load
from autoddi.estimation import do_compute, eval_performance
from sklearn.metrics import confusion_matrix
import data_preprocessing
import numpy as np
import pickle
# ==== 1. 参数配置 ====
data_name = "twosides_S2"                   # 你的数据集名
save_suffix = "_search"              # 与训练时一致
fold = "fold0"
model_id = 0                         # 搜索得到的模型编号
arch = [8, 'GeneralConv', 'GATConv', 'PANPool', 'global_add']# 搜索出来的结构
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ==== 2. 加载图数据 ====
print(f"Loading graph data for {data_name} with fold {fold} and save_suffix {save_suffix}")

graph = Planetoid(data_name=data_name, fold=fold, save_suffix=save_suffix)
print(f"Graph data loaded: {graph}")
# ==== 3. 构造模型并加载权重 ====
print(f"Constructing model with architecture: {arch}")
model = GnnModel(
    arch,
    graph.num_features,
    graph.num_labels,
    graph.rel_total,
    graph.data_name
).to(device)

# model = model_load(graph.data_logger_save, model_id, fold).to(device)
print(f"Loading model from {graph.data_logger_save} with model_id {model_id} and fold {fold}")
model = torch.load("logger/twosides_S2_search/seed11/model_twosides_S2_search_num1_fold0.pkl").to(device)
model.eval()

# ==== 4. 收集测试集预测 ====
print("Collecting predictions from test data loader")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in graph.test_data_loader:
        # batch: 调用 do_compute 来获取预测概率
        _, _, probas_pred, ground_truth = do_compute(batch, device, model)

        preds = (probas_pred > 0.5).astype(int)  # 二分类阈值，或改为 argmax 用于多分类
        all_preds.append(preds)
        all_labels.append(ground_truth)

# ==== 5. 拼接预测结果 ====
print(type(all_labels), type(all_preds))
print(all_labels[:5])
print(all_preds[:5])

all_labels_flat = np.concatenate(all_labels).astype(int)

# 将所有预测拼接，并转为一维整数
all_preds_flat = np.concatenate(all_preds).reshape(-1).astype(int)

# === 计算混淆矩阵并保存 ===
print("Calculating confusion matrix")
cm = confusion_matrix(all_labels_flat, all_preds_flat)

with open("cm_tw_2.pkl", "wb") as f:
    pickle.dump(cm, f)