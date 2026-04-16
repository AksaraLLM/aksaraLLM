"""
AksaraLLM — Upload Model ke HuggingFace Hub
============================================
Support pre-trained checkpoint maupun SFT checkpoint.

Cara pakai:
  # Upload model pre-trained
  python upload_to_hf.py \\
      --checkpoint checkpoints/best_model.pt \\
      --repo AksaraLLM/Kiel-Mini-59M

  # Upload model hasil SFT
  python upload_to_hf.py \\
      --checkpoint checkpoints/sft/sft_best.pt \\
      --repo AksaraLLM/Kiel-Mini-59M-SFT \\
      --sft
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

import sys
sys.path.insert(0, str(Path(__file__).parent))
from aksarallm.config import aksaraLLMConfig
from aksarallm.model import aksaraLLMModel


# ════════════════════════════════════════════════════════════════════════════
# Model Card Generator
# ════════════════════════════════════════════════════════════════════════════

def make_model_card(repo_id: str, cfg: dict, meta: dict, is_sft: bool) -> str:
    model_name = repo_id.split("/")[-1]
    n_params = meta.get("n_params", "~59M")
    step      = meta.get("step", "?")
    val_loss  = meta.get("val_loss", "?")
    epoch     = meta.get("epoch", "?")

    if is_sft:
        training_desc = f"Fine-tuned for **{epoch} epoch(s)** on the Indonesian Alpaca dataset (52,000 instruction-answer pairs)."
        use_case = "**Instruction following** in Indonesian and English.\nAsk it to write articles, translate text, answer questions, summarize, etc."
        sample_prompt = """### Instruksi:
Jelaskan apa itu kecerdasan buatan dalam bahasa sederhana.

### Jawaban:"""
        tag_extra = "\n  - instruction-tuning\n  - sft\n  - indonesian"
        dataset_desc = "[Indonesian Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) — 52K instruksi yang diterjemahkan ke Bahasa Indonesia."
    else:
        training_desc = f"Pre-trained for **{step} steps** on the TinyStories dataset."
        use_case = "**Text completion** and general language modeling."
        sample_prompt = "Pada suatu hari di sebuah desa kecil,"
        tag_extra = "\n  - pre-trained"
        dataset_desc = "[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — dataset cerita pendek untuk pre-training."

    return f"""---
language:
  - id
  - en
license: apache-2.0
tags:
  - AksaraLLM
  - language-model
  - transformer
  - open-source
  - from-scratch{tag_extra}
library_name: pytorch
model_type: causal-lm
---

<p align="center">
  <img src="https://raw.githubusercontent.com/AksaraLLM/.github/main/profile/banner.png" width="700">
</p>

# {model_name}

> Part of the **[AksaraLLM](https://github.com/AksaraLLM)** open-source LLM initiative.
> *Building the world's most transparent Large Language Model.*

## Model Overview

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only Transformer |
| **Parameters** | ~{n_params} |
| **Layers** | {cfg.get("n_layers", "?")} |
| **Attention Heads** | {cfg.get("n_heads", "?")} |
| **Hidden Size** | {cfg.get("n_embd", "?")} |
| **Context Length** | {cfg.get("max_seq_len", "?")} tokens |
| **Tokenizer** | GPT-2 (50,257 vocab) |
| **Val Loss** | {val_loss} |

## Architecture Highlights
- 🔄 **RMSNorm** (pre-normalization)
- 🌀 **RoPE** (Rotary Position Embeddings)
- ⚡ **SwiGLU** activation function
- 🔗 Weight-tied input/output embeddings

## Training
{training_desc}

**Dataset:** {dataset_desc}

## Use Case
{use_case}

## Quick Start

```python
import torch
from transformers import AutoTokenizer

# Download dari HuggingFace
checkpoint = torch.load(
    hf_hub_download("{repo_id}", "checkpoint.pt"),
    map_location="cpu",
    weights_only=False,
)

# Setup model
from aksarallm.config import aksaraLLMConfig
from aksarallm.model import aksaraLLMModel

config = aksaraLLMConfig(**checkpoint["config"])
model  = aksaraLLMModel(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate teks
prompt = "{sample_prompt}"
ids    = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(ids, max_new_tokens=150, temperature=0.8)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Reproducing Training

```bash
# Clone repo
git clone https://github.com/AksaraLLM/aksaraLLM.git
cd aksaraLLM

# Install dependencies
pip install -r requirements.txt

# Pre-training
python train.py --size mini

# Fine-tuning (SFT)
python sft.py \\
  --checkpoint checkpoints/best_model.pt \\
  --data translated_alpaca_id.jsonl \\
  --epochs 3
```

## About AksaraLLM

**AksaraLLM** is a community-driven open-source initiative to build LLMs focused on
regional and national languages, especially Southeast Asian languages.

We open-source *everything* — architecture, training code, datasets, and methodology.

- 🌐 GitHub: [github.com/AksaraLLM](https://github.com/AksaraLLM)
- 💬 Discord: [discord.gg/aksarallm](https://discord.gg/aksarallm)
- 🤗 HuggingFace: [huggingface.co/AksaraLLM](https://huggingface.co/AksaraLLM)

## License

Apache License 2.0 — free for commercial and research use.

## Citation

```bibtex
@software{{aksarallm2026,
  title   = {{AksaraLLM: Open-Source LLM for Regional Languages}},
  author  = {{AksaraLLM Community}},
  year    = {{2026}},
  url     = {{https://github.com/AksaraLLM}},
}}
```
"""


# ════════════════════════════════════════════════════════════════════════════
# Upload Logic
# ════════════════════════════════════════════════════════════════════════════

def upload(args):
    print(f"\n📦 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    cfg      = ckpt.get("config", {})
    step     = ckpt.get("global_step", ckpt.get("step", "?"))
    val_loss = ckpt.get("loss", ckpt.get("val_loss", "?"))
    epoch    = ckpt.get("epoch", "?")

    # Hitung jumlah parameter
    n_params = "?"
    try:
        config  = aksaraLLMConfig(**{k: v for k, v in cfg.items()
                                     if k in aksaraLLMConfig.__dataclass_fields__})
        model   = aksaraLLMModel(config)
        n_params = f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
        del model
    except Exception as e:
        print(f"  ⚠️ Couldn't count params: {e}")

    meta = {"n_params": n_params, "step": step, "val_loss": val_loss, "epoch": epoch}
    print(f"   Parameters : {n_params}")
    print(f"   Step/Epoch : {step} / {epoch}")
    print(f"   Val Loss   : {val_loss}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simpan file-file yang akan diupload
        torch.save(ckpt, os.path.join(tmpdir, "checkpoint.pt"))
        torch.save(ckpt.get("model_state_dict", {}), os.path.join(tmpdir, "model.pt"))

        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        card = make_model_card(args.repo, cfg, meta, is_sft=args.sft)
        with open(os.path.join(tmpdir, "README.md"), "w", encoding="utf-8") as f:
            f.write(card)

        print(f"\n🚀 Uploading ke HuggingFace: {args.repo}")
        api = HfApi()

        try:
            create_repo(args.repo, exist_ok=True, repo_type="model", private=False)
            print(f"   Repo: https://huggingface.co/{args.repo}")
        except Exception as e:
            print(f"   ⚠️ Repo: {e}")

        api.upload_folder(
            folder_path=tmpdir,
            repo_id=args.repo,
            commit_message=args.commit,
        )

    print(f"\n{'='*50}")
    print(f"🎉 BERHASIL DIUPLOAD!")
    print(f"🔗 https://huggingface.co/{args.repo}")
    print(f"{'='*50}")


# ════════════════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AksaraLLM — Upload model ke HuggingFace Hub"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path ke checkpoint (.pt)")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace repo ID, contoh: AksaraLLM/Kiel-Mini-59M")
    parser.add_argument("--commit", type=str, default="Upload AksaraLLM model",
                        help="Pesan commit")
    parser.add_argument("--sft", action="store_true",
                        help="Tandai sebagai model SFT (beda model card)")
    args = parser.parse_args()
    upload(args)


if __name__ == "__main__":
    main()
