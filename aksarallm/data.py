"""
aksaraLLM - Data Loading Pipeline (with Disk Cache)
"""
import os
import glob
import json
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset

from .config import aksaraLLMConfig


class _AksaraTokenizerAdapter:
    """Minimal HF-tokenizer-compatible shim around AksaraTokenizer.

    Lets the pretraining pipeline (this module + train.py) use the
    Indonesian-optimized tokenizer trained by aksara-tokenizer without a
    parallel code path — it only implements the handful of methods/attributes
    this pipeline actually calls (encode/decode/eos_token_id/pad_token).
    """

    def __init__(self, aksara_tokenizer):
        self._tok = aksara_tokenizer
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.eos_token_id = aksara_tokenizer.tokenizer.token_to_id("[EOS]")
        self.pad_token_id = aksara_tokenizer.tokenizer.token_to_id("[PAD]")
        self.vocab_size = aksara_tokenizer.vocab_size

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        ids = self._tok.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._tok.decode(ids)


def load_tokenizer(config: aksaraLLMConfig):
    """Load the tokenizer used for pretraining.

    Uses the Indonesian-optimized AksaraTokenizer when ``config.tokenizer_path``
    points at a directory saved by ``AksaraTokenizer.save_pretrained()`` (see
    the aksara-tokenizer repo). Falls back to GPT-2's tokenizer otherwise —
    that fallback is English-biased and meant only for architecture
    smoke-testing, not real Indonesian pretraining.
    """
    if config.tokenizer_path:
        from .tokenizer_utils import AksaraTokenizer

        aksara_tok = AksaraTokenizer.from_pretrained(config.tokenizer_path)
        print(
            f"📝 Using AksaraTokenizer from {config.tokenizer_path} "
            f"(vocab_size={aksara_tok.vocab_size})"
        )
        return _AksaraTokenizerAdapter(aksara_tok)

    print(
        "⚠️  No tokenizer_path set — falling back to GPT-2's tokenizer. "
        "It is English-biased and meant only for architecture smoke-testing; "
        "train aksara-tokenizer first for real Indonesian pretraining."
    )
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def _iter_local_jsonl_records(path: str):
    """Yield dicts from a local .jsonl file or every .jsonl file in a
    directory (same convention as aksara-tokenizer's trainer script)."""
    files = [path] if os.path.isfile(path) else sorted(
        glob.glob(os.path.join(path, "**/*.jsonl"), recursive=True)
    )
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _load_local_split(path: str, split: str, max_samples):
    """Load a local JSONL corpus (file or directory) as a train/validation
    split. There's no train/validation distinction on disk for a plain
    corpus, so this holds out a fixed tail (~2%, at least 1 example) as
    'validation' deterministically — same idea as the HF-dataset fallback
    below, just applied locally instead of hitting the network."""
    records = list(_iter_local_jsonl_records(path))
    if not records:
        raise ValueError(f"No JSONL records found under '{path}'")

    n_val = max(1, int(len(records) * 0.02))
    if split == "train":
        subset = records[: max(len(records) - n_val, 1)]
    else:
        subset = records[-n_val:]

    if max_samples:
        subset = subset[:max_samples]
    return HFDataset.from_list(subset)


def _load_hf_split(dataset_name, dataset_config, split, max_samples):
    """Load a corpus split — either a local JSONL file/directory (when
    `dataset_name` exists on disk) or a HuggingFace dataset id — tolerating
    datasets with no dedicated 'validation' split (e.g. raw Wikipedia dumps
    only have 'train', and local corpora never have one) by holding out the
    tail of 'train' instead."""
    if os.path.exists(dataset_name):
        return _load_local_split(dataset_name, split, max_samples)

    args = (dataset_name, dataset_config) if dataset_config else (dataset_name,)
    try:
        ds = load_dataset(*args, split=split, trust_remote_code=True)
    except (ValueError, KeyError):
        if split == "train":
            raise
        print(
            f"⚠️  Dataset '{dataset_name}' has no '{split}' split — "
            f"holding out the tail of 'train' instead."
        )
        full = load_dataset(*args, split="train", trust_remote_code=True)
        n = min(max_samples or 2000, len(full))
        return full.select(range(len(full) - n, len(full)))

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


class TextDataset(Dataset):
    """Tokenized text dataset for language model training."""

    def __init__(
        self,
        config: aksaraLLMConfig,
        split: str = "train",
        tokenizer=None,
        max_samples: int | None = None,
    ):
        self.config = config
        self.seq_len = config.max_seq_len

        self.tokenizer = tokenizer or load_tokenizer(config)

        # ===== SISTEM CACHE: Simpan hasil tokenisasi ke disk =====
        # Buat nama file cache unik berdasarkan dataset + split + jumlah sample
        cache_id = f"{config.dataset_name}_{config.dataset_config}_{split}_{max_samples}_{config.max_seq_len}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()[:10]
        cache_dir = os.path.join("cache_tokens")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"cached_{cache_hash}.pt")

        # Cek apakah file cache sudah ada
        if os.path.exists(cache_path):
            print(f"⚡ CACHE DITEMUKAN! Memuat data instan dari: {cache_path}")
            self.data = torch.load(cache_path, weights_only=True)
            print(f"✅ Dataset ready (dari cache): {len(self.data)} sequences")
            return  # Langsung selesai tanpa tokenisasi!

        # ===== Kalau belum ada cache, tokenisasi dari awal =====
        print(f"📥 Loading dataset: {config.dataset_name} ({split})...")

        ds = _load_hf_split(config.dataset_name, config.dataset_config, split, max_samples)
        # Try common text column names
        text_key = "text"
        if text_key not in ds.column_names:
            for key in ["content", "story", "sentence", "document"]:
                if key in ds.column_names:
                    text_key = key
                    break

        print(f"📊 Tokenizing {len(ds)} examples...")

        import array
        # Detect GPU for keepalive pings
        _keepalive_device = None
        if torch.cuda.is_available():
            _keepalive_device = torch.device("cuda")
            print("   (GPU keepalive aktif — Colab nggak akan matiin sesi)")

        # Tokenize all texts and concatenate into one big sequence
        # We use an array of 32-bit unsigned integers to save memory instead of a Python list
        all_tokens = array.array('I')
        for i, example in enumerate(ds):
            tokens = self.tokenizer.encode(
                example[text_key],
                add_special_tokens=True,
            )
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eos_token_id)

            if (i + 1) % 10000 == 0:
                print(f"   Processed {i + 1}/{len(ds)} examples "
                      f"({len(all_tokens) / 1e6:.1f}M tokens)")

            # GPU Keepalive: tiap 100k baris, sentuh GPU biar Colab gak curiga
            if _keepalive_device and (i + 1) % 100000 == 0:
                _ = torch.randn(64, 64, device=_keepalive_device) @ torch.randn(64, 64, device=_keepalive_device)
                torch.cuda.synchronize()

        # Convert to tensor and chunk into sequences
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        n_sequences = len(all_tokens) // (self.seq_len + 1)
        all_tokens = all_tokens[:n_sequences * (self.seq_len + 1)]
        self.data = all_tokens.view(n_sequences, self.seq_len + 1)

        # ===== SIMPAN CACHE KE DISK =====
        print(f"💾 Menyimpan cache ke disk: {cache_path}")
        torch.save(self.data, cache_path)

        print(f"✅ Dataset ready: {len(self.data)} sequences, "
              f"{len(all_tokens) / 1e6:.1f}M tokens total")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.data[idx]
        return {
            "input_ids": chunk[:-1],   # (seq_len,)
            "targets": chunk[1:],       # (seq_len,)
        }


def create_dataloaders(
    config: aksaraLLMConfig,
    tokenizer=None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = 5000,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    tokenizer = tokenizer or load_tokenizer(config)

    train_dataset = TextDataset(
        config, split="train", tokenizer=tokenizer,
        max_samples=max_train_samples,
    )
    val_dataset = TextDataset(
        config, split="validation", tokenizer=tokenizer,
        max_samples=max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader
