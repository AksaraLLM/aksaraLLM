"""
aksaraLLM - Model Configuration v3

Adds:
- aksarallm-20b from-scratch preset (42L × 6144d × 48H-Q / 8H-KV, vocab 131072, 8k ctx).
- aksarallm-tiny preset for CPU-only dry-runs (no TPU/GPU needed).
- Configurable ``rope_theta`` (1e4 for small models, 1e6 for long-context 20B).
- `get_config(size: str)` convenience accessor.

Backwards compatible with v2 presets (nano … xlarge). GQA is supported on every
preset via ``n_kv_heads``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class aksaraLLMConfig:
    """Configuration for aksaraLLM model variants."""

    # ── Architecture ─────────────────────────────────────────────────
    vocab_size: int = 50257           # GPT-2 default; 131072 for the 20B
    max_seq_len: int = 256
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 0               # 0 → mirror ``n_heads`` (pure MHA)
    n_embd: int = 384
    n_inner: int = 1536               # SwiGLU intermediate (≈2.7×–4× n_embd)
    dropout: float = 0.1
    bias: bool = False
    rope_theta: float = 10000.0       # 1e6 for the 20B (8k ctx)
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    # ── Training defaults (scale-appropriate) ───────────────────────
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    max_steps: int = 5000
    warmup_steps: int = 200
    eval_interval: int = 250
    eval_steps: int = 20
    log_interval: int = 50
    save_interval: int = 1000
    gradient_accumulation_steps: int = 4
    grad_clip: float = 1.0

    # ── Data ─────────────────────────────────────────────────────────
    dataset_name: str = "roneneldan/TinyStories"

    # ── Paths ────────────────────────────────────────────────────────
    output_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        if self.n_kv_heads <= 0:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by "
            f"n_kv_heads ({self.n_kv_heads})"
        )

    # ── Derived ──────────────────────────────────────────────────────
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_heads

    @property
    def n_params(self) -> int:
        """Total parameter count (with weight tying).

        Matches the v2 arithmetic but also works for the 20B / tiny presets.
        """
        head_dim = self.head_dim

        # Token embedding (tied with lm_head → counted once).
        emb = self.vocab_size * self.n_embd if self.tie_embeddings else 2 * self.vocab_size * self.n_embd

        # Per-layer attention: Q + K + V + output.
        attn_q = self.n_embd * (self.n_heads * head_dim)
        attn_k = self.n_embd * (self.n_kv_heads * head_dim)
        attn_v = self.n_embd * (self.n_kv_heads * head_dim)
        attn_o = self.n_embd * self.n_embd
        attn = self.n_layers * (attn_q + attn_k + attn_v + attn_o)

        # SwiGLU MLP per layer: gate + up + down.
        mlp = self.n_layers * (3 * self.n_embd * self.n_inner)

        # RMSNorm: 2 per layer + 1 final.
        norm = (2 * self.n_layers + 1) * self.n_embd

        return emb + attn + mlp + norm


# ─────────────────────────────────────────────────────────────────────
#  PRESET REGISTRY
# ─────────────────────────────────────────────────────────────────────
CONFIGS: dict[str, aksaraLLMConfig] = {
    # ═══ Unit-test / CPU-only dry-run preset ════════════════════════
    "tiny": aksaraLLMConfig(
        vocab_size=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        n_embd=64,
        n_inner=128,
        max_seq_len=64,
        max_steps=3,
        warmup_steps=0,
        batch_size=2,
        learning_rate=1e-3,
        dropout=0.0,
    ),  # ~40k params — used by every --dry-run path.

    # ═══ Small Models (Kaggle / Colab Free) ══════════════════════════
    "nano": aksaraLLMConfig(
        n_layers=4, n_heads=4, n_embd=256, n_inner=1024,
        max_seq_len=256, max_steps=3000, batch_size=32,
    ),  # ~10M params

    "micro": aksaraLLMConfig(
        n_layers=6, n_heads=6, n_embd=384, n_inner=1536,
        max_seq_len=256, max_steps=5000, batch_size=32,
    ),  # ~26M params

    "mini": aksaraLLMConfig(
        n_layers=8, n_heads=8, n_embd=512, n_inner=2048,
        max_seq_len=512, max_steps=10000, batch_size=16,
    ),  # ~59M params

    # ═══ Medium Models (Kaggle T4x2 / Colab Pro) ═════════════════════
    "small": aksaraLLMConfig(
        n_layers=12, n_heads=12, n_embd=768, n_inner=3072,
        max_seq_len=512, max_steps=20000, batch_size=8,
    ),  # ~130M params

    "medium": aksaraLLMConfig(
        n_layers=16, n_heads=16, n_kv_heads=4,
        n_embd=1024, n_inner=2816, max_seq_len=1024,
        max_steps=50000, batch_size=4, learning_rate=3e-4, dropout=0.0,
    ),  # ~200M params

    # ═══ Large Models (Cloud A100) ═══════════════════════════════════
    "large": aksaraLLMConfig(
        n_layers=20, n_heads=20, n_kv_heads=4,
        n_embd=1280, n_inner=3456, max_seq_len=2048,
        max_steps=100000, batch_size=2, learning_rate=2e-4,
        dropout=0.0, gradient_accumulation_steps=16,
    ),  # ~500M params

    "xlarge": aksaraLLMConfig(
        n_layers=24, n_heads=16, n_kv_heads=4,
        n_embd=2048, n_inner=5504, max_seq_len=2048,
        max_steps=200000, batch_size=1, learning_rate=1.5e-4,
        dropout=0.0, gradient_accumulation_steps=32,
    ),  # ~1B params

    # ═══ Flagship — from-scratch 20B (TPU v5p pod) ═══════════════════
    # Numbers locked by the project brief. See REPORT.md §3 for the
    # Chinchilla / compute / memory analysis that backs them.
    "20b": aksaraLLMConfig(
        vocab_size=131072,
        n_embd=6144,
        n_inner=16384,
        n_layers=42,
        n_heads=48,
        n_kv_heads=8,           # 6× GQA ratio
        max_seq_len=8192,
        rope_theta=1_000_000.0, # long-context RoPE θ
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
        # Training defaults matched to a TPU v5p-256 pod with 2M-token GBS.
        batch_size=1,
        gradient_accumulation_steps=512,
        learning_rate=3e-4,
        weight_decay=0.1,
        max_steps=200_000,
        warmup_steps=2000,
        grad_clip=1.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────
#  ACCESSORS
# ─────────────────────────────────────────────────────────────────────
def get_config(size: str) -> aksaraLLMConfig:
    """Look up a preset by name.

    Accepts aliases like ``"aksarallm-20b"`` or ``"20B"`` in addition to the
    bare keys in :data:`CONFIGS`.
    """
    key = size.lower().strip()
    if key.startswith("aksarallm-"):
        key = key[len("aksarallm-"):]
    if key not in CONFIGS:
        available = ", ".join(sorted(CONFIGS.keys()))
        raise KeyError(f"Unknown config '{size}'. Available: {available}")
    return CONFIGS[key]


# ─────────────────────────────────────────────────────────────────────
#  CHAT-TEMPLATE + SPECIAL-TOKEN CONSTANTS (shared by tokenizer + inference)
# ─────────────────────────────────────────────────────────────────────
# From-scratch 20B uses its own non-ChatML template. These are the canonical
# strings — do not diverge between tokenizer training, SFT data generation,
# and inference.
SPECIAL_TOKENS: list[str] = [
    "[BOS]", "[EOS]", "[PAD]", "[UNK]",
    "[SYS]", "[/SYS]",
    "[INST]", "[/INST]",
]

# Default system prompt baked into inference and SFT identity reinforcement.
DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, "
    "sopan, dan membantu. Jawab dengan jelas, jujur, dan ringkas."
)

# Jinja-style chat template (Hugging Face tokenizers compatible). Keep in
# sync with `aksarallm/tokenizer_utils.py::apply_chat_template`.
AKSARA_CHAT_TEMPLATE: str = (
    "{% if messages[0]['role'] == 'system' %}"
    "[SYS]{{ messages[0]['content'] }}[/SYS]"
    "{% set messages = messages[1:] %}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "[INST]{{ message['content'] }}[/INST]"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}[EOS]"
    "{% endif %}"
    "{% endfor %}"
)
