import torch.nn as nn
import torch
from torch.nn.init import trunc_normal_
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def scaled_dot_product_attn(q, k, v, attn_mask=None):
    d_qk = q.size(-1)
    scaled_qk_dot = (q @ k.transpose(-2, -1)) / (d_qk ** 0.5)
    if attn_mask is not None:
        scaled_qk_dot = scaled_qk_dot.masked_fill(~attn_mask, -float("inf"))
    attn_scores = torch.softmax(scaled_qk_dot, dim=-1)
    attn_output = attn_scores @ v
    return attn_output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        in_shape = in_features.shape    # (..., seq_len, d_in)

        seq_len = in_shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        q = (self.q_proj(in_features)).view(in_shape[:-1] + (self.num_heads, self.d_head))   # (..., seq, nh, dh)
        k = (self.k_proj(in_features)).view(in_shape[:-1] + (self.num_heads, self.d_head))
        v = (self.v_proj(in_features)).view(in_shape[:-1] + (self.num_heads, self.d_head))
        attn_out = scaled_dot_product_attn(q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2), attn_mask=mask)   # (..., nh, seq, dh)

        out_shape = in_shape[:-1] + (-1,) 
        attn_out = attn_out.transpose(-3, -2).reshape(out_shape)
        return self.o_proj(attn_out)

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, theta: float, d_model: int, num_heads: int, max_seq_len: int, device=None):
        super().__init__()
        self.rope = RoPE(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_shape = in_features.shape    # (..., seq_len, d_in)

        seq_len = in_shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        q = (self.q_proj(in_features)).view(in_shape[:-1] + (self.num_heads, self.d_head))   # (..., seq, nh, dh)
        k = (self.k_proj(in_features)).view(in_shape[:-1] + (self.num_heads, self.d_head))
        v = (self.v_proj(in_features)).view(in_shape[:-1] + (self.num_heads, self.d_head))

        q = self.rope(q.transpose(-3, -2), token_positions)
        k = self.rope(k.transpose(-3, -2), token_positions)

        attn_out = scaled_dot_product_attn(q, k, v.transpose(-3, -2), attn_mask=mask)   # (..., nh, seq, dh)

        out_shape = in_shape[:-1] + (-1,) 
        return self.output_proj(attn_out.transpose(-3, -2).reshape(out_shape))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device = None):
        super().__init__()
        self.attn = MultiHeadSelfAttentionWithRoPE(theta, d_model, num_heads, max_seq_len, device=device)
        self.ln1 = RMSNorm(d_model, device=device)

        # ffn
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device)

        # others
        self.device = device

    def forward(self, in_features: torch.Tensor):
        token_positions = torch.arange(in_features.shape[-2], device=self.device)

        x = in_features
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, context_length: int, rope_theta: float, num_layers: int, device: torch.device = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)
    
    def forward(self, in_indices):
        embeds = self.token_embeddings(in_indices)
        output = self.layers(embeds)
        output = self.lm_head(self.ln_final(output))
        return output


# problem solution
class Solution:
    """GPT-2 XL 一次前向（序列长 = context_length）的矩阵乘 FLOPs。
    一个 (m×k)·(k×n) 矩阵乘 = 2·m·k·n FLOPs。"""
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400

    def p_3_6_a(self):
        T, d = self.context_length, self.d_model
        d_ff = self.d_ff

        num_param = 0
        num_param += 4 * (d * d)                # QKVO
        num_param += 2 * (d * d_ff)             # Feed-Forward
        num_param += 2 * d                      # RMS-norm
        num_param *= self.num_layers

        num_param += self.vocab_size * d        # vocab
        num_param += self.vocab_size * d        # lm head
        return num_param


    def p_3_6_b(self):
        T, d = self.context_length, self.d_model
        d_ff = self.d_ff

        # 每个 transformer block
        flops = 0
        flops += 2 * (3 * T * d * d)              # QKV 投影 (3 个 (T×d)·(d×d))
        flops += 2 * (T * T * d + T * T * d)      # 注意力: QK^T 与 attn·V
        flops += 2 * (T * d * d)                  # 注意力输出投影 output_proj
        flops += 2 * (2 * T * d * d_ff)           # 标准 FFN (上投影 d->d_ff + 下投影 d_ff->d), GPT-2 XL
        flops *= self.num_layers

        # lm_head: (T×d)·(d×vocab)
        flops += 2 * (T * d * self.vocab_size)
        return flops

    def p_3_6_c(self):
        return "FFN"


if __name__ == "__main__":
    sol = Solution()
    total = sol.p_3_6_b()
    print(f"Problem 3.6(b) — GPT-2 XL total matmul FLOPs (one forward pass)")
    print(f"  {total:,} FLOPs")
    print(f"  ≈ {total / 1e12:.3f} TFLOPs")