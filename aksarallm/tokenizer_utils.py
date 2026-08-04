"""
aksaraLLM — Tokenizer Utilities

Thin wrapper around a byte-level BPE tokenizer (HuggingFace ``tokenizers``
library) used across the AksaraLLM pipeline — trained by
``aksara-tokenizer/scripts/train_tokenizer_20b.py`` and consumed by the
pretraining, SFT/DPO, and inference scripts.

The vocabulary always reserves a block of special tokens matching the
AksaraLLM chat template:

    [BOS], [EOS], [PAD], [UNK], [SYS], [/SYS], [INST], [/INST]

...plus a reasoning-trace pair, so SFT data can include an explicit
chain-of-thought the model produces (and can be budget-limited or hidden
from the user at inference time) before its final answer — the same
"thinking" idea behind o1/R1/extended-thinking-style models:

    [THINK], [/THINK]

See ``aksarallm.reasoning.generate_with_thinking`` for the inference-side
helper that uses these. These are registered as ``special_tokens`` with the
BPE trainer so they get reserved vocabulary slots regardless of corpus
frequency, and survive save/load round-trips.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

SPECIAL_TOKENS = [
    "[BOS]", "[EOS]", "[PAD]", "[UNK]", "[SYS]", "[/SYS]", "[INST]", "[/INST]",
    "[THINK]", "[/THINK]",
]

TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


class AksaraTokenizer:
    """Byte-level BPE tokenizer for AksaraLLM.

    Wraps a :class:`tokenizers.Tokenizer` (available as ``self.tokenizer``
    for lower-level access, e.g. ``tokenizer.token_to_id(...)``).
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train_bpe_from_iterator(
        cls,
        iterator: Iterable[str],
        vocab_size: int = 131_072,
        min_frequency: int = 2,
    ) -> "AksaraTokenizer":
        """Train a fresh byte-level BPE tokenizer from an iterator of text."""
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        return cls(tokenizer)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pretrained(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(out_dir, TOKENIZER_FILE))
        config = {
            "model_type": "aksarallm_bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
        }
        with open(os.path.join(out_dir, TOKENIZER_CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str) -> "AksaraTokenizer":
        tokenizer_path = path if os.path.isfile(path) else os.path.join(path, TOKENIZER_FILE)
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"No {TOKENIZER_FILE} found at {path}")
        return cls(Tokenizer.from_file(tokenizer_path))

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def fertility_report(self, text: str) -> dict:
        """Report tokens-per-word ("fertility") for a sample of text.

        Lower fertility means the tokenizer represents the language more
        efficiently (fewer tokens per word = cheaper training/inference).
        """
        words = text.split()
        n_tokens = len(self.encode(text))
        fertility = n_tokens / max(len(words), 1)
        report = {
            "text": text,
            "n_words": len(words),
            "n_tokens": n_tokens,
            "fertility": fertility,
        }
        print(
            f"[AksaraTokenizer] {len(words)} words -> {n_tokens} tokens "
            f"(fertility={fertility:.2f} tokens/word)"
        )
        return report
