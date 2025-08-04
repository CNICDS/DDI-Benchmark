from autoddi.auto_model import AutoModel
from set_config import data_name, save_suffix, search_parameter, gnn_parameter
from planetoid import Planetoid
import warnings
import numpy as np
import random
import torch

def set_seed(seed):
        print('seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
seed=226
set_seed(seed)
print(seed)
print(data_name)
warnings.filterwarnings('ignore',category=UserWarning)

graph = Planetoid(data_name, save_suffix=save_suffix)

AutoModel(graph, search_parameter, gnn_parameter, save_suffix)
print(seed)
print(data_name)