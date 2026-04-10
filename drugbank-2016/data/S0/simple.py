import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取文件
# 假设文件是 tab 分隔，且每个文件结构相同：drug1, drug2, reaction_type
file1 = pd.read_csv('test_ddi.txt', header=None, sep=" ", names=['drug1', 'drug2', 'reaction_type'])
file2 = pd.read_csv('train_ddi.txt', header=None, sep=" ", names=['drug1', 'drug2', 'reaction_type'])
file3 = pd.read_csv('valid_ddi.txt', header=None, sep=" ", names=['drug1', 'drug2', 'reaction_type'])

# 2. 合并文件
df = pd.concat([file1, file2, file3], ignore_index=True)
# 保存合并后的 DataFrame 到新的 CSV 文件
df.to_csv('merged_file.csv', index=False)  # index=False 防止保存行索引
