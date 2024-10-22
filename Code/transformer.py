# -*- coding: utf-8 -*-
"""
@author: iopenzd
"""

import torch, copy
import torch.nn as nn
import torch.nn.functional as F

# 获取激活函数
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


#复制指定模块的实例，并返回一个模块列表
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    r"""Users may modify or implement in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()

        # d_model is emb_size 嵌入维度
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):    # 设置模型状态，检查状态中是否存在激活函数activation
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required). 到编码器层的序列
            src_mask: the mask for the src sequence (optional). SRC序列的掩码
            src_key_padding_mask: the mask for the src keys per batch (optional).  每批SRC键的掩码
        Shape:
            see the docs in Transformer class.
        """

        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)  # attn_output, attn_output_weights
        src = src + self.dropout1(src2)    # add
        src = self.norm1(src)   # norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # ffn
        src = src + self.dropout2(src2) # add
        src = self.norm2(src)  # norm
        return src, attn


class TransformerEncoder(nn.Module):
    r"""  TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required). 实例
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8) # Transformer编码器层对象，用于构建编码器的多个层
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6) # num_layers表示编码器中的层数
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, device, norm=None):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None): # 向前传播
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).      给编码器的序列
            mask: the mask for the src sequence (optional).   SRC序列的掩码
            src_key_padding_mask: the mask for the src keys per batch (optional). 每批SRC键的掩码
        Shape:
            see the docs in Transformer class.
        """

        output = src
        # 创建一个用于存储注意力权重的变量
        attn_output = torch.zeros((src.shape[1], src.shape[0], src.shape[0]), device=self.device)  # batch, seq_len, seq_len 其中batch表示批量大小，seq_len表示序列长度

        # 对每个编码器层进行迭代
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_output += attn

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output # 返回编码结果和累加的注意力权重

