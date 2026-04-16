"""
aksaraLLM - Model Configuration v2

Supports scales from 10M (nano) to 1B+ (xlarge).
Added GQA (Grouped Query Attention) support for memory-efficient scaling.
"""
from dataclasses import dataclass, field


@dataclass
class aksaraLLMConfig:
    """Configuration for aksaraLLM model variants."""
    
    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    max_seq_len: int = 256
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 0      # KV heads for GQA (0 = same as n_heads = standard MHA)
    n_embd: int = 384
    n_inner: int = 1536      # 4 * n_embd (use 2.7x for SwiGLU-optimal)
    dropout: float = 0.1
    bias: bool = False
    
    # Training
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
    
    # Data
    dataset_name: str = "roneneldan/TinyStories"
    
    # Paths
    output_dir: str = "checkpoints"
    
    def __post_init__(self):
        # Default n_kv_heads to n_heads (standard MHA) if not set
        if self.n_kv_heads <= 0:
            self.n_kv_heads = self.n_heads
        # Validate GQA config
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
    
    @property
    def n_params(self) -> int:
        """Estimate total parameters (accurate for v2 architecture)."""
        head_dim = self.n_embd // self.n_heads
        
        # Token embedding (tied with lm_head, counted once)
        emb = self.vocab_size * self.n_embd
        
        # Attention per layer: Q + K + V + Output projection
        attn_q = self.n_embd * (self.n_heads * head_dim)      # Q proj
        attn_k = self.n_embd * (self.n_kv_heads * head_dim)   # K proj (GQA)
        attn_v = self.n_embd * (self.n_kv_heads * head_dim)   # V proj (GQA)
        attn_o = self.n_embd * self.n_embd                     # Output proj
        attn = self.n_layers * (attn_q + attn_k + attn_v + attn_o)
        
        # SwiGLU MLP per layer: gate + up + down
        mlp = self.n_layers * (3 * self.n_embd * self.n_inner)
        
        # RMSNorm: 2 per layer + 1 final
        norm = (2 * self.n_layers + 1) * self.n_embd
        
        return emb + attn + mlp + norm


# Pre-defined configurations
CONFIGS = {
    # ═══ Small Models (Kaggle/Colab Free Tier) ═══
    "nano": aksaraLLMConfig(
        n_layers=4,
        n_heads=4,
        n_embd=256,
        n_inner=1024,
        max_seq_len=256,
        max_steps=3000,
        batch_size=32,
    ),  # ~10M params
    
    "micro": aksaraLLMConfig(
        n_layers=6,
        n_heads=6,
        n_embd=384,
        n_inner=1536,
        max_seq_len=256,
        max_steps=5000,
        batch_size=32,
    ),  # ~26M params
    
    "mini": aksaraLLMConfig(
        n_layers=8,
        n_heads=8,
        n_embd=512,
        n_inner=2048,
        max_seq_len=512,
        max_steps=10000,
        batch_size=16,
    ),  # ~59M params
    
    # ═══ Medium Models (Kaggle T4x2 / Colab Pro) ═══
    "small": aksaraLLMConfig(
        n_layers=12,
        n_heads=12,
        n_embd=768,
        n_inner=3072,
        max_seq_len=512,
        max_steps=20000,
        batch_size=8,
    ),  # ~130M params
    
    "medium": aksaraLLMConfig(
        n_layers=16,
        n_heads=16,
        n_kv_heads=4,       # GQA: 4 KV heads shared among 16 Q heads
        n_embd=1024,
        n_inner=2816,       # 2.75x (SwiGLU optimal)
        max_seq_len=1024,
        max_steps=50000,
        batch_size=4,
        learning_rate=3e-4,
        dropout=0.0,        # No dropout for pre-training
    ),  # ~200M params
    
    # ═══ Large Models (Cloud GPU: A100) ═══
    "large": aksaraLLMConfig(
        n_layers=20,
        n_heads=20,
        n_kv_heads=4,       # GQA
        n_embd=1280,
        n_inner=3456,       # 2.7x
        max_seq_len=2048,
        max_steps=100000,
        batch_size=2,
        learning_rate=2e-4,
        dropout=0.0,
        gradient_accumulation_steps=16,
    ),  # ~500M params
    
    "xlarge": aksaraLLMConfig(
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,       # GQA
        n_embd=2048,
        n_inner=5504,       # 2.7x
        max_seq_len=2048,
        max_steps=200000,
        batch_size=1,
        learning_rate=1.5e-4,
        dropout=0.0,
        gradient_accumulation_steps=32,
    ),  # ~1B params
}
