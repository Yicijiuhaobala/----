import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    input : batch_size * seq_len * input_dim
        q : batch_size * input_dim * dim_k
        k : batch_size * input_dim * dim_k
        v : batch_size * input_dim * dim_v
    """

    def __init__(self, input_dim, dim_k, dim_v):
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v
        b = torch.bmm(Q, K.permute(0, 2, 1))
        a = torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.dim_k)
        attention = torch.bmm(self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.dim_k)), V)

        return attention


class MultiHeadSelfAttention(nn.Module):
    """
    input : batch_size * seq_len * input_dim
        q : batch_size * input_dim * dim_k
        k : batch_size * input_dim * dim_k
        v : batch_size * input_dim * dim_v
    """

    def __init__(self, input_dim, dim_k, dim_v, nums_head):
        super().__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)

        self.nums_head = nums_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.q(x).view(-1, x.shape[1], self.nums_head, self.dim_k // self.nums_head).permute(0, 2, 1, 3)
        K = self.k(x).view(-1, x.shape[1], self.nums_head, self.dim_k // self.nums_head).permute(0, 2, 1, 3)
        V = self.v(x).view(-1, x.shape[1], self.nums_head, self.dim_v // self.nums_head).permute(0, 2, 1, 3)

        a = self.softmax(torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.dim_k))
        # b = torch.bmm(Q, K.permute(0, 1, 3, 2))
        attention = torch.matmul(self.softmax(torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.dim_k)),
                                 V).transpose(-2, -1)  # [batch_size, n_head, seq_len, hidden_size // n_head]
        attention = attention.transpose(1, 2)  # [batch_size, seq_len, n_head, hidden_size // n_head]

        # output = attention.reshape(-1, x.shape[1], x.shape[2])  # [batch_size, seq_len, hidden_size]
        output = attention.reshape(x.shape[0], x.shape[1], -1)
        print(output.shape)
        # 或
        # attention = attention.permute(2, 0, 1, 3)
        # output = torch.cat([_ for _ in attention], dim=-1)

        return output


# attention = SelfAttention(10, 6, 8)
# result = attention(torch.randn([1, 5, 10]))
# print(result)

attention = MultiHeadSelfAttention(10, 6, 8, 2)
result = attention(torch.randn([1, 5, 10]))
print(result)

nn.Transformer()