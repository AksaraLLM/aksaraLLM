"""
aksaraLLM - Transformer Model Architecture v2

A decoder-only transformer following modern LLM design:
- RMSNorm (pre-norm)
- Rotary Position Embeddings (RoPE)
- SwiGLU activation
- Grouped Query Attention (GQA) for memory efficiency
- Gradient checkpointing support
- No bias terms

Changelog v2 (2024):
- Added GQA (Grouped Query Attention) — 25% memory savings at scale
- Added gradient checkpointing — 60% VRAM savings for training
- Replaced manual attention with PyTorch SDPA (Flash Attention compatible)
- Added KV-cache support for faster inference
- Added config presets for 200M, 500M, 1B scales

Changelog v3:
- KV-cache is now actually implemented (was documented but not wired up):
  generate() does a single prefill pass over the prompt, then one
  single-token forward per step reusing cached K/V — O(n) total generation
  work instead of the previous O(n^2) full-reprocessing-every-step.
- generate() supports repetition_penalty (HF-standard convention).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import aksaraLLMConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    10% faster than LayerNorm — no mean subtraction needed.
    Used by LLaMA, Qwen, Mistral, and all modern LLMs.
    """

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


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0):
    """Apply rotary embeddings to query and key tensors.

    `freqs_cis` is the model's *full* precomputed buffer; `start_pos` is the
    absolute position of the first token in this call (0 for a normal
    full-sequence forward pass or KV-cache prefill, >0 for an incremental
    decode step where only the newest token(s) are being processed).
    """
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    T = xq.shape[1]
    freqs_cis = freqs_cis[start_pos:start_pos + T]
    freqs_cis = freqs_cis[None, :, None, :]  # (1, seq_len, 1, head_dim//2)

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query heads for GQA.

    If n_rep=1, this is standard MHA (no repeat needed).
    If n_rep=4, each KV head is shared among 4 query heads.
    """
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, head_dim)
        .reshape(B, n_kv_heads * n_rep, T, head_dim)
    )


class KVCache:
    """Growable per-layer key/value cache for incremental decoding.

    One instance is created per `generate()` call and threaded through every
    layer's forward pass. `update()` appends this step's newly-computed K/V
    for a given layer and returns the full (cached + new) K/V for that layer
    to attend over.
    """

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    @property
    def seq_len(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=2)
        return self.k[layer_idx], self.v[layer_idx]


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and optional GQA.

    Grouped Query Attention (GQA):
    - Standard MHA: n_kv_heads = n_heads (every head has own K,V)
    - GQA: n_kv_heads < n_heads (multiple Q heads share K,V)
    - MQA: n_kv_heads = 1 (all Q heads share one K,V)

    GQA saves ~25% memory with minimal quality loss.
    Used by LLaMA 2 70B, Mistral 7B, Qwen 2.5.
    """

    def __init__(self, config: aksaraLLMConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.n_kv_heads = getattr(config, 'n_kv_heads', config.n_heads)  # Backward compatible
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.n_embd // config.n_heads

        self.q_proj = nn.Linear(config.n_embd, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (using each token's *absolute* position — matters once start_pos > 0)
        q_for_rope = q.transpose(1, 2)  # (B, T, n_heads, head_dim)
        k_for_rope = k.transpose(1, 2)  # (B, T, n_kv_heads, head_dim)
        q_for_rope, k_for_rope = apply_rotary_emb(q_for_rope, k_for_rope, freqs_cis, start_pos)
        q = q_for_rope.transpose(1, 2)
        k = k_for_rope.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Repeat KV heads for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if kv_cache is not None:
            # T (new queries) and k.shape[2] (all cached + new keys) can
            # differ, so build an explicit causal mask rather than relying on
            # is_causal=True's implicit bottom-right alignment for L != S —
            # this is correct on every PyTorch 2.x version, not just newer
            # ones. True = "may attend".
            T_total = k.shape[2]
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device)
            k_pos = torch.arange(0, T_total, device=x.device)
            attn_mask = (k_pos[None, :] <= q_pos[:, None])[None, None, :, :]
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Fast path used by training/normal full-sequence forward passes
            # — unchanged from before, no cache bookkeeping overhead.
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        return out


class SwiGLU(nn.Module):
    """SwiGLU activation function used in modern LLMs.

    SwiGLU(x) = Swish(xW_gate) * (xW_up)
    ~2% more accurate than ReLU/GELU for same compute.
    Used by LLaMA, PaLM, Qwen.
    """

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

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, start_pos=start_pos, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class aksaraLLMModel(nn.Module):
    """
    aksaraLLM v2 — A decoder-only transformer language model.

    Architecture:
    - RMSNorm (pre-norm)
    - Rotary Position Embeddings (RoPE)
    - SwiGLU activation
    - Grouped Query Attention (GQA)
    - Weight-tied embeddings
    - Gradient checkpointing support
    """

    def __init__(self, config: aksaraLLMConfig):
        super().__init__()
        self.config = config
        self._gradient_checkpointing = False

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

        # Special scaled init for residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_kv_heads = getattr(config, 'n_kv_heads', config.n_heads)
        gqa_info = f", GQA {n_kv_heads}kv/{config.n_heads}q" if n_kv_heads != config.n_heads else ""
        print(f"aksaraLLM v2 initialized: {n_params / 1e6:.2f}M parameters"
              f" (L={config.n_layers}, H={config.n_heads}, D={config.n_embd}{gqa_info})")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save ~60% VRAM during training."""
        self._gradient_checkpointing = True
        print("🔧 Gradient checkpointing enabled (saves ~60% VRAM)")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

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
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token indices — the *new* tokens
                only when kv_cache is used (e.g. shape (B, 1) mid-generation),
                the full sequence otherwise.
            targets: (batch_size, seq_len) target token indices for loss
            kv_cache: optional KVCache to read from / append to (inference only)
            start_pos: absolute position of input_ids[:, 0] in the sequence —
                0 for a normal forward pass or KV-cache prefill, >0 for an
                incremental decode step.

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar loss if targets provided
        """
        B, T = input_ids.shape
        assert start_pos + T <= self.config.max_seq_len, \
            f"Context length {start_pos + T} exceeds max {self.config.max_seq_len}"

        x = self.drop(self.token_emb(input_ids))

        for i, layer in enumerate(self.layers):
            if self._gradient_checkpointing and self.training:
                # kv_cache is never used while training, so it's fine that
                # checkpoint() only forwards the positional args here.
                x = checkpoint(layer, x, self.freqs_cis, start_pos, use_reentrant=False)
            else:
                x = layer(x, self.freqs_cis, start_pos=start_pos, kv_cache=kv_cache, layer_idx=i)

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
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        With use_cache=True (default), does a single "prefill" forward pass
        over the prompt to populate a KV-cache, then one single-token forward
        per subsequent step reusing it — O(n) total work for n generated
        tokens instead of the O(n^2) you get from reprocessing the whole
        growing sequence at every step (use_cache=False, kept for
        correctness comparisons / environments where caching isn't wanted).

        Args:
            input_ids: (batch_size, seq_len) prompt tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            repetition_penalty: >1.0 discourages repeating already-generated
                tokens (HF-standard convention: divide positive logits,
                multiply negative logits, by the penalty)
            use_cache: use incremental KV-cache decoding (see above)

        Returns:
            (batch_size, seq_len + max_new_tokens) generated tokens
        """
        self.eval()

        input_ids = input_ids[:, -self.config.max_seq_len:]
        generated = input_ids

        kv_cache = KVCache(self.config.n_layers) if use_cache else None
        next_logits = None
        cur_pos = 0

        if use_cache:
            logits, _ = self(input_ids, kv_cache=kv_cache, start_pos=0)
            cur_pos = input_ids.shape[1]
            next_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            if use_cache:
                step_logits = next_logits
            else:
                idx_cond = generated[:, -self.config.max_seq_len:]
                logits, _ = self(idx_cond)
                step_logits = logits[:, -1, :]

            step_logits = step_logits.clone() / max(temperature, 1e-5)

            # Repetition penalty (HF-standard: divide positive logits,
            # multiply negative logits, so both cases reduce probability).
            if repetition_penalty != 1.0:
                for b in range(step_logits.shape[0]):
                    prev = generated[b].unique()
                    scores = step_logits[b, prev]
                    step_logits[b, prev] = torch.where(
                        scores < 0, scores * repetition_penalty, scores / repetition_penalty
                    )

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                step_logits[step_logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                step_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if use_cache:
                if cur_pos >= self.config.max_seq_len:
                    break  # exhausted the trained context window
                logits, _ = self(next_token, kv_cache=kv_cache, start_pos=cur_pos)
                cur_pos += 1
                next_logits = logits[:, -1, :]

        return generated
