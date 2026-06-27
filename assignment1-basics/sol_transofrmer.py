import torch.nn as nn
import torch
from torch.nn.init import trunc_normal_
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)  # 上采样到 fp32 计算，避免平方时溢出
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate up-project
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down-project
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # value up-project

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        return self.w2(x1 * torch.sigmoid(x1) * x3)  # SiLU(W1 x) ⊙ (W3 x) -> W2

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # cos & sin
        self.d_k = d_k
        thetas = theta ** (- 2 * torch.arange(0, d_k // 2, device=device) / d_k)
        angles = torch.arange(max_seq_len)[:, None] * thetas    # (max_seq_len, d_k // 2)
        self.register_buffer('cosine', torch.cos(angles))
        self.register_buffer('sine', torch.sin(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        cosine, sine = self.cosine[token_positions], self.sine[token_positions]
        y1 = x1 * cosine - x2 * sine
        y2 = x1 * sine + x2 * cosine
        return torch.stack([y1, y2], dim=-1).reshape(x.shape[:-1] + (self.d_k,))


if __name__ == "__main__":
    pass