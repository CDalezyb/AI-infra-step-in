import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OriginalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: float = 0.1):
        super().__init__()
        assert (
            d_model % num_head == 0
        ), f"d_model: {d_model} 必须能被 num_head: {num_head}整除"
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model / num_head
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = x.shape[0]
        Q = self.Wq(x).view(bs, -1, self.num_head, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(bs, -1, self.num_head, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(bs, -1, self.num_head, self.head_dim).transpose(1, 2)

        attention_mat = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            # 不要污染计算图，要使用 out-place操作
            attention_mat = attention_mat.mask_fill(mask == 0, float("-inf"))

        attention_scores = F.softmax(attention_mat, dim=-1)
        attention_scores = self.drop(attention_scores)

        output = (
            torch.matmul(attention_scores, V)
            .transpose(1, 2)
            .contiguous()
            .view(bs, -1, self.d_model)
        )

        output = self.Wo(output)

        return output, attention_scores
