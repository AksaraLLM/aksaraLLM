"""
aksaraLLM — A truly open-source Large Language Model
"""
from .config import aksaraLLMConfig, CONFIGS
from .tokenizer_utils import AksaraTokenizer

try:
    # aksaraLLMModel needs torch, which tokenizer-only workflows
    # (e.g. aksara-tokenizer's trainer script) shouldn't be forced to install.
    from .model import aksaraLLMModel
except ImportError:
    aksaraLLMModel = None

__version__ = "0.1.0"
__all__ = ["aksaraLLMConfig", "aksaraLLMModel", "CONFIGS", "AksaraTokenizer"]
