# -*- coding: utf-8 -*-
"""
@author: iopenzd
"""
import torch
import torch.nn as nn
import math
import transformer

# absolute PE 为输入序列添加位置信息
class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__() # 定义位置编码器
        max_len = max(5000, seq_len) # 取输入序列长度seq_len和5000的最大值，用于创建位置编码矩阵的大小
        self.dropout = nn.Dropout(p=dropout) #对位置编码进行随机失活
        pe = torch.zeros(max_len, d_model) # 一个形状为(max_len, d_model)的零填充矩阵，用于存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 一个形状为(max_len, 1)的张量，表示位置编码的位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #一个形状为(d_model/2,)的张量，用于计算位置编码的频率因子
        pe[:, 0::2] = torch.sin(position * div_term)
        #通过使用正弦和余弦函数，将位置编码矩阵pe的偶数列填充为正弦值，奇数列填充为余弦值。这样就创建了一个能够表示序列中每个位置的位置编码矩阵

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0: -1]

        pe = pe.unsqueeze(0).transpose(0, 1) #将位置编码矩阵pe进行形状变换，使其变为形状为(1, max_len, d_model)的张量，并进行转置操作
        self.register_buffer('pe', pe) # 将位置编码矩阵pe注册为模型的缓冲区，以便在模型的前向传播中使用

    # Input:  seq_len x batch_size x dim,
    # Output: seq_len, batch_size, dim
    # 前向传播过程中，将位置编码矩阵pe的前seq_len行与输入张量x进行相加，以将位置编码添加到输入张量中的每个位置
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# contextual PE 
# class PositionalEncoding(nn.Module):
#     def __init__(self, seq_len, d_model, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout=nn.Dropout(p=dropout)
#         self.pe = nn.Conv1d(d_model, d_model, kernel_size =5, padding = 'same')
#     def forward(self, x):
#         x = x + self.pe(x)
#         return self.dropout(x)

# relative PE    
# class PositionalEncoding(nn.Module):
#     def __init__(self, seq_len, d_model, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.d_model = d_model
#         self.seq_len = seq_len

#         pe = self.generate_positional_encoding()
#         self.register_buffer('pe', pe)

#     def generate_positional_encoding(self):
#         position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
#         rel_pos = position / div_term

#         sin_rel_pos = torch.sin(rel_pos)
#         cos_rel_pos = torch.cos(rel_pos)

#         pe = torch.zeros(self.seq_len, self.d_model)
#         pe[:, 0::2] = sin_rel_pos
#         pe[:, 1::2] = cos_rel_pos

#         return pe.unsqueeze(0)

#     def forward(self, x):
#         pe = self.pe[:, :x.size(1), :]  # Adjust the positional encoding size to match the batch size of x
#         x = x + pe
#         return self.dropout(x)
    

class Permute(torch.nn.Module): # 将输入张量的维度进行置换
    def forward(self, x):
        return x.permute(1, 0)


class clsTransformerModel(nn.Module):

    def __init__(self, task_type, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_re, nhid_task, nlayers, dropout=0.1):
        super(clsTransformerModel, self).__init__()

        self.trunk_net = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.BatchNorm1d(batch)
        )

        encoder_layers = transformer.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers, device)

        self.batch_norm = nn.BatchNorm1d(batch)

        # Reconstruction Layers
        self.re_net = nn.Sequential(
            nn.Linear(emb_size, nhid_re),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_re, nhid_re),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_re, input_size),
        )


        if task_type == 'classification':
            # Classification Layers
            self.class_net = nn.Sequential(
                nn.Linear(emb_size, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Dropout(p=0.3),
                nn.Linear(nhid_task, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Dropout(p=0.3),
                nn.Linear(nhid_task, nclasses) # nclasses用于预测类别
            )

    def forward(self, x, task_type):
        x = self.trunk_net(x.permute(1, 0, 2))
        x, attn = self.transformer_encoder(x)
        x = self.batch_norm(x)
        # x : seq_len x batch x emb_size

        if task_type == 'reconstruction':
            output = self.re_net(x).permute(1, 0, 2)
        elif task_type == 'classification':
            output = self.class_net(x[-1])
        return output, attn


