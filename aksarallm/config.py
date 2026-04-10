"""
aksaraLLM - Model Configuration
"""
from dataclasses import dataclass


@dataclass
class aksaraLLMConfig:
    """Configuration for aksaraLLM model variants."""
    
    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    max_seq_len: int = 256
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384
    n_inner: int = 1536  # 4 * n_embd
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
    
    @property
    def n_params(self) -> int:
        """Estimate total parameters."""
        # Embedding + positional
        emb = self.vocab_size * self.n_embd + self.max_seq_len * self.n_embd
        # Attention: Q, K, V, out projection per layer
        attn = self.n_layers * (4 * self.n_embd * self.n_embd)
        # MLP: up + down projection per layer
        mlp = self.n_layers * (2 * self.n_embd * self.n_inner)
        # LayerNorm
        ln = self.n_layers * (4 * self.n_embd)
        # Output head (tied with embedding)
        return emb + attn + mlp + ln


# Pre-defined configurations
CONFIGS = {
    "nano": aksaraLLMConfig(
        n_layers=4,
        n_heads=4,
        n_embd=256,
        n_inner=1024,
        max_seq_len=256,
        max_steps=3000,
        batch_size=32,
    ),
    "micro": aksaraLLMConfig(
        n_layers=6,
        n_heads=6,
        n_embd=384,
        n_inner=1536,
        max_seq_len=256,
        max_steps=5000,
        batch_size=32,
    ),
    "mini": aksaraLLMConfig(
        n_layers=8,
        n_heads=8,
        n_embd=512,
        n_inner=2048,
        max_seq_len=512,
        max_steps=10000,
        batch_size=16,
    ),
    "small": aksaraLLMConfig(
        n_layers=12,
        n_heads=12,
        n_embd=768,
        n_inner=3072,
        max_seq_len=512,
        max_steps=20000,
        batch_size=8,
    ),
}
