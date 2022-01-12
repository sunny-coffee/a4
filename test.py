# print(['a']+'b'*8)

# from nltk.util import pr
import torch
import torch.nn as nn
import numpy as np

# a = torch.tensor([0.0, 1.0, 2,0])
# print(a([0]))

# batch_size = 2
# max_length = 3
# hidden_size = 2
# n_layers = 4
# # 这个RNN由两个全连接层组成，对应的两个hidden_state的维度是2，输入向量维度是1
# rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)

# x = torch.FloatTensor([[1, 0, 0], [1, 2, 3]]).resize_(2, 3, 1)
# seq_lengths = np.array([1, 3])  # list of integers holding information about the batch size at each sequence step
# # print(x)

# order_idx = np.argsort(seq_lengths)[::-1]
# # print('order_idx:', str(order_idx))
# order_x = x[order_idx.tolist()]
# order_seq = seq_lengths[order_idx]
# print(order_x)

# pack = nn.utils.rnn.pack_padded_sequence(order_x, order_seq, batch_first=True)

# h0 = torch.randn(n_layers, batch_size, hidden_size)
# # forward
# out, final = rnn(pack, h0)
# print(out)
# print('='*80)
# print(out)
# unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
# print(unpacked)
# print('='*80)
# print(final)
# print('='*80)

a = torch.arange(10).reshape(5,2)

for i in torch.split(a, 1, dim=1):
    print(i)