import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import ast

class CSVTextCodeData(Dataset):
    def __init__(self, data_path, split="train", drop_rate=0.6, full_data=False):
        super(CSVTextCodeData, self).__init__()
        
        # 1. 读取包含数据的 CSV 文件
        self.df = pd.read_csv(data_path)
        
        # 假设 CSV 中有一列名为 'split'，用来区分 'train', 'valid', 'test'
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            
        self.split = split
        self.drop_rate = drop_rate
        self.full_data = full_data
        
        # 2. 设定特征的维度
        # sentence-BERT 和 CodeBERT 提取出的特征向量维度通常都是 768
        self.orig_dims = [768, 768] 
        
        # 3. 设定输入序列的长度 (seq_len)
        # 这里需要填入你通过特征提取模型处理数据时设定的 max_length
        # 假设文本截断长度为 50，代码截断长度为 100
        self.seq_lens = [50, 100] 

    def get_dim(self):
        return self.orig_dims
    
    def get_seq_len(self):
        return self.seq_lens

    def __len__(self):
        # 返回当前数据集切片的大小
        return len(self.df)
    
    def get_missing_mode(self):
        """
        双模态缺失逻辑：
        0: 缺描述文本
        1: 缺代码
        2: 都不缺失
        """
        if self.full_data:
            return 2
        if random.random() < self.drop_rate:
            # 以一定的 drop_rate 随机抛弃其中一个模态
            return random.randint(0, 1)
        else:
            return 2

    def __getitem__(self, idx):
        # 4. 从 CSV 获取当前行的数据
        row = self.df.iloc[idx]
        
        # --- 数据读取核心逻辑 ---
        # 注意：这里的模型底层 Transformer 需要的是特征矩阵，而不是纯文本。
        # 假设你的 CSV 中存储了用 sentence-BERT 和 CodeBERT 提前提取好的 `.npy` 或 `.pt` 文件路径：
        # text_feat = np.load(row['text_feat_path']) 
        # code_feat = np.load(row['code_feat_path'])
        
        # 如果你还没提前提取，你需要在这里动态调用预训练模型进行前向计算（会严重拖慢训练速度，强烈建议提前提取好存下来）。
        
        # 下面这两行是用随机张量占位，防止你跑不通。
        # 等你把真实的特征提取逻辑写好后，请替换掉这两行。
        text_feat = np.random.randn(self.seq_lens[0], self.orig_dims[0]) 
        code_feat = np.random.randn(self.seq_lens[1], self.orig_dims[1])
        
        # 5. 转为 PyTorch 张量
        L_feat = torch.tensor(text_feat).float()  # 代表描述文本特征 (Language)
        C_feat = torch.tensor(code_feat).float()  # 代表代码特征 (Code)
        
        # 6. 读取标签（假设 CSV 中的标签列名为 'label'）
        label = torch.tensor(row['label']).float()
        
        # 组合输入和缺失标记
        X = (L_feat, C_feat)
        missing_code = self.get_missing_mode()

        return X, label, missing_code