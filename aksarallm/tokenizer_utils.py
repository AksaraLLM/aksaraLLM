"""
aksaraLLM - Tokenizer utilities

A thin wrapper around a Hugging Face ``tokenizers`` BPE model that:

1. Trains / loads a BPE tokenizer with the AksaraLLM **from-scratch** special
   tokens — ``[BOS] [EOS] [PAD] [UNK] [SYS] [/SYS] [INST] [/INST]``.
2. Applies the canonical ``AKSARA_CHAT_TEMPLATE`` (see ``config.py``) without
   pulling in the full ``transformers`` dependency stack. This keeps the
   tokenizer usable both inside and outside of ``transformers``-based
   training/eval pipelines.
3. Saves / loads to a directory in the standard HF layout (``tokenizer.json``
   + ``tokenizer_config.json``) so ``transformers.AutoTokenizer`` and the
   ``llama.cpp`` GGUF converter can consume the output unchanged.

The ChatML-based tokenizer used by the older ``aksarallm-1.5b-v2`` (Qwen
lineage) is deliberately **not** touched by this module — see ``REPORT.md``
§1 for the rationale.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import AKSARA_CHAT_TEMPLATE, DEFAULT_SYSTEM_PROMPT, SPECIAL_TOKENS


# ──────────────────────────────────────────────────────────────────────
#  Chat rendering (stand-alone, no tokenizer required)
# ──────────────────────────────────────────────────────────────────────
def render_chat(
    messages: Sequence[dict],
    add_generation_prompt: bool = False,
    system_prompt: str | None = None,
) -> str:
    """Render a list of ``{"role": ..., "content": ...}`` messages.

    Mirrors :data:`AKSARA_CHAT_TEMPLATE` exactly. Using raw Python here
    instead of Jinja keeps this function usable in tight training loops and
    in environments without ``jinja2`` installed.

    Args:
        messages: list of ``{"role": "system"|"user"|"assistant",
            "content": str}`` entries.
        add_generation_prompt: if True, *do not* emit the trailing ``[EOS]``
            for the last assistant turn (there is no assistant turn to close);
            caller is expected to use this when asking the model to generate.
        system_prompt: if the first message is not a system message and
            ``system_prompt`` is provided, one is injected at the start.
    """
    msgs = list(messages)
    if system_prompt is not None and (not msgs or msgs[0].get("role") != "system"):
        msgs = [{"role": "system", "content": system_prompt}, *msgs]

    out: list[str] = []
    if msgs and msgs[0].get("role") == "system":
        out.append(f"[SYS]{msgs[0]['content']}[/SYS]")
        msgs = msgs[1:]

    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(f"[INST]{content}[/INST]")
        elif role == "assistant":
            out.append(f"{content}[EOS]")
        else:
            raise ValueError(f"Unsupported chat role: {role!r}")

    rendered = "".join(out)
    if add_generation_prompt and rendered.endswith("[EOS]"):
        rendered = rendered[: -len("[EOS]")]
    return rendered


# ──────────────────────────────────────────────────────────────────────
#  BPE tokenizer wrapper
# ──────────────────────────────────────────────────────────────────────
@dataclass
class AksaraTokenizer:
    """Thin facade around ``tokenizers.Tokenizer``.

    This class is intentionally *not* a subclass of
    ``transformers.PreTrainedTokenizerBase``; exporting to that format is a
    one-liner via :meth:`save_pretrained` which writes ``tokenizer.json``
    plus a minimal ``tokenizer_config.json``.
    """

    # The underlying ``tokenizers.Tokenizer`` instance.
    tokenizer: object  # tokenizers.Tokenizer — avoid import cost at module load

    @classmethod
    def train_bpe(
        cls,
        files: Sequence[str],
        vocab_size: int = 131072,
        min_frequency: int = 2,
        initial_alphabet: Sequence[str] | None = None,
    ) -> AksaraTokenizer:
        """Train a new byte-level BPE tokenizer on a corpus of text files.

        Args:
            files: list of plain-text or JSONL file paths (line-delimited).
            vocab_size: target vocabulary size (131072 for the 20B).
            min_frequency: drop merges that appear fewer than this many times.
            initial_alphabet: optional list of single-character strings to
                force into the vocab (Indonesian-specific punctuation, e.g.).
        """
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=list(initial_alphabet) if initial_alphabet else pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train(list(files), trainer=trainer)

        return cls(tokenizer=tok)

    @classmethod
    def train_bpe_from_iterator(
        cls,
        iterator: Iterable[str],
        vocab_size: int = 131072,
        min_frequency: int = 2,
    ) -> AksaraTokenizer:
        """In-memory variant of :meth:`train_bpe`. Useful for tests."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(iterator, trainer=trainer)
        return cls(tokenizer=tok)

    # ── IO ───────────────────────────────────────────────────────────
    def save_pretrained(self, save_directory: str | os.PathLike[str]) -> None:
        """Write ``tokenizer.json`` + ``tokenizer_config.json`` to ``save_directory``."""
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save(os.path.join(str(save_directory), "tokenizer.json"))  # type: ignore[attr-defined]

        cfg = {
            "tokenizer_class": "AksaraTokenizer",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "added_special_tokens": list(SPECIAL_TOKENS),
            "chat_template": AKSARA_CHAT_TEMPLATE,
            "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
            "model_max_length": 8192,
        }
        with open(os.path.join(str(save_directory), "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, load_directory: str | os.PathLike[str]) -> AksaraTokenizer:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(str(load_directory), "tokenizer.json"))
        return cls(tokenizer=tok)

    # ── Encoding / decoding ──────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()  # type: ignore[attr-defined]

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)  # type: ignore[attr-defined]

    def id_to_token(self, idx: int) -> str | None:
        return self.tokenizer.id_to_token(idx)  # type: ignore[attr-defined]

    @property
    def bos_token_id(self) -> int:
        return int(self.tokenizer.token_to_id("[BOS]"))  # type: ignore[attr-defined]

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.token_to_id("[EOS]"))  # type: ignore[attr-defined]

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.token_to_id("[PAD]"))  # type: ignore[attr-defined]

    @property
    def unk_token_id(self) -> int:
        return int(self.tokenizer.token_to_id("[UNK]"))  # type: ignore[attr-defined]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids  # type: ignore[attr-defined]
        if add_bos:
            ids = [self.bos_token_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)  # type: ignore[attr-defined]

    def apply_chat_template(
        self,
        messages: Sequence[dict],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> str | list[int]:
        """Render and optionally tokenize a chat. Mirrors the HF API."""
        rendered = render_chat(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return self.encode(rendered)
        return rendered
