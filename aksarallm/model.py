"""
aksaraLLM - Transformer Model Architecture

A decoder-only transformer following modern LLM design:
- RMSNorm (pre-norm)
- Rotary Position Embeddings (RoPE)
- SwiGLU activation
- No bias terms
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import aksaraLLMConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the frequency tensor for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary embeddings to query and key tensors."""
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis[:xq.shape[1]]
    freqs_cis = freqs_cis[None, :, None, :]  # (1, seq_len, 1, head_dim//2)
    
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(self, config: aksaraLLMConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q_for_rope = q.transpose(1, 2)  # (B, T, n_heads, head_dim)
        k_for_rope = k.transpose(1, 2)
        q_for_rope, k_for_rope = apply_rotary_emb(q_for_rope, k_for_rope, freqs_cis)
        q = q_for_rope.transpose(1, 2)
        k = k_for_rope.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        
        return out


class SwiGLU(nn.Module):
    """SwiGLU activation function used in modern LLMs."""
    
    def __init__(self, config: aksaraLLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.n_inner, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.n_inner, bias=config.bias)
        self.down_proj = nn.Linear(config.n_inner, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(self, config: aksaraLLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class aksaraLLMModel(nn.Module):
    """
    aksaraLLM — A decoder-only transformer language model.
    
    Architecture:
    - RMSNorm (pre-norm)
    - Rotary Position Embeddings (RoPE)
    - SwiGLU activation
    - Weight-tied embeddings
    """
    
    def __init__(self, config: aksaraLLMConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.lm_head.weight
        
        # Precompute RoPE frequencies
        head_dim = config.n_embd // config.n_heads
        freqs_cis = precompute_freqs_cis(head_dim, config.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"aksaraLLM initialized: {n_params / 1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) token indices
            targets: (batch_size, seq_len) target token indices for loss
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar loss if targets provided
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        
        x = self.drop(self.token_emb(input_ids))
        
        freqs_cis = self.freqs_cis[:T]
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: (batch_size, seq_len) prompt tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            
        Returns:
            (batch_size, seq_len + max_new_tokens) generated tokens
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
