"""
aksaraLLM - Data Loading Pipeline (with Disk Cache)
"""
import os
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from .config import aksaraLLMConfig


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
        
        # Load tokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ===== SISTEM CACHE: Simpan hasil tokenisasi ke disk =====
        # Buat nama file cache unik berdasarkan dataset + split + jumlah sample
        cache_id = f"{config.dataset_name}_{split}_{max_samples}_{config.max_seq_len}"
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
        
        # Load dataset
        if config.dataset_name == "roneneldan/TinyStories":
            ds = load_dataset(config.dataset_name, split=split, trust_remote_code=True)
            text_key = "text"
        else:
            ds = load_dataset(config.dataset_name, split=split, trust_remote_code=True)
            # Try common text column names
            text_key = "text"
            if text_key not in ds.column_names:
                for key in ["content", "story", "sentence", "document"]:
                    if key in ds.column_names:
                        text_key = key
                        break
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
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
