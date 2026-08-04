"""
AksaraLLM — Upload Model ke HuggingFace Hub
============================================
Mengonversi checkpoint from-scratch (pre-trained maupun SFT) ke format
`transformers` standar (lewat `aksarallm.hf_export`) dan mengunggahnya —
hasilnya langsung bisa dipakai dengan `AutoModelForCausalLM.from_pretrained()`,
tanpa kode model kustom.

Cara pakai:
  # Upload model pre-trained
  python upload_to_hf.py \\
      --checkpoint checkpoints/best_model.pt \\
      --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b \\
      --repo AksaraLLM/aksarallm-mini

  # Upload model hasil SFT
  python upload_to_hf.py \\
      --checkpoint checkpoints/sft/sft_best.pt \\
      --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b \\
      --repo AksaraLLM/aksarallm-mini-sft \\
      --sft
"""
import argparse
import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo

import sys
sys.path.insert(0, str(Path(__file__).parent))
from aksarallm.config import aksaraLLMConfig
from aksarallm.model import aksaraLLMModel
from aksarallm.tokenizer_utils import AksaraTokenizer
from aksarallm.hf_export import export_to_hf


# ════════════════════════════════════════════════════════════════════════════
# Model Card Generator
# ════════════════════════════════════════════════════════════════════════════

def make_model_card(repo_id: str, cfg: dict, meta: dict, is_sft: bool) -> str:
    model_name = repo_id.split("/")[-1]
    n_params = meta.get("n_params", "?")
    step      = meta.get("step", "?")
    val_loss  = meta.get("val_loss", "?")
    epoch     = meta.get("epoch", "?")

    if is_sft:
        training_desc = f"Fine-tuned for **{epoch} epoch(s)** on an Indonesian instruction dataset (Alpaca-style)."
        use_case = "**Instruction following** in Indonesian.\nAsk it to write articles, translate text, answer questions, summarize, etc."
        sample_prompt = "Jelaskan apa itu kecerdasan buatan dalam bahasa sederhana."
        tag_extra = "\n  - instruction-tuning\n  - sft\n  - indonesian"
        dataset_desc = "Indonesian instruction data curated in [aksara-data](https://github.com/AksaraLLM/aksara-data)."
    else:
        training_desc = f"Pre-trained from scratch for **{step} steps** on Indonesian text."
        use_case = "**Text completion** and general Indonesian language modeling (base model, not instruction-tuned)."
        sample_prompt = "Indonesia adalah negara"
        tag_extra = "\n  - pre-trained"
        dataset_desc = "Indonesian Wikipedia and/or other corpora — see [aksara-data](https://github.com/AksaraLLM/aksara-data) for the pipeline."

    return f"""---
language:
  - id
license: apache-2.0
tags:
  - AksaraLLM
  - language-model
  - transformer
  - open-source
  - from-scratch{tag_extra}
library_name: transformers
model_type: causal-lm
---

# {model_name}

> Part of the **[AksaraLLM](https://github.com/AksaraLLM)** open-source LLM initiative.
> Trained from scratch — not a fine-tune of another base model.

## Model Overview

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only Transformer (RoPE, GQA, RMSNorm, SwiGLU) |
| **Parameters** | ~{n_params} |
| **Layers** | {cfg.get("n_layers", "?")} |
| **Attention Heads** | {cfg.get("n_heads", "?")} ({cfg.get("n_kv_heads", "?")} KV heads) |
| **Hidden Size** | {cfg.get("n_embd", "?")} |
| **Context Length** | {cfg.get("max_seq_len", "?")} tokens |
| **Tokenizer** | [AksaraTokenizer](https://github.com/AksaraLLM/aksara-tokenizer) ({cfg.get("vocab_size", "?")} vocab, Indonesian-optimized BPE) |
| **Val Loss** | {val_loss} |

Exported to standard `transformers` format via `aksarallm.hf_export` (see the
[aksaraLLM repo](https://github.com/AksaraLLM/aksaraLLM)) — this repo works
directly with `AutoModelForCausalLM`, no `trust_remote_code` needed.

## Training
{training_desc}

**Dataset:** {dataset_desc}

## Use Case
{use_case}

## Quick Start

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="auto", device_map="auto")

prompt = "{sample_prompt}"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Reproducing Training

```bash
git clone https://github.com/AksaraLLM/aksaraLLM.git
cd aksaraLLM
pip install -r requirements.txt

# Pre-training (needs a tokenizer trained via aksara-tokenizer first)
python train.py --size mini --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b

# Fine-tuning (SFT)
python sft.py --checkpoint checkpoints/best_model.pt --data data/sft_id.jsonl \\
    --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b
```

## About AksaraLLM

**AksaraLLM** is a community-driven open-source initiative to build an
Indonesian LLM **from scratch** — architecture, tokenizer, training code,
data pipeline, and evaluation are all open, not just the final weights.

- 🌐 GitHub: [github.com/AksaraLLM](https://github.com/AksaraLLM)
- 💬 Discord: [discord.gg/aksarallm](https://discord.gg/aksarallm)
- 🤗 HuggingFace: [huggingface.co/AksaraLLM](https://huggingface.co/AksaraLLM)

## License

Apache License 2.0 — free for commercial and research use.

## Citation

```bibtex
@software{{aksarallm2026,
  title   = {{AksaraLLM: An Open-Source Indonesian LLM, Trained From Scratch}},
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

    cfg_dict = ckpt.get("config", {})
    step     = ckpt.get("global_step", ckpt.get("step", "?"))
    val_loss = ckpt.get("loss", ckpt.get("val_loss", "?"))
    epoch    = ckpt.get("epoch", "?")

    config = aksaraLLMConfig(**{
        k: v for k, v in cfg_dict.items() if k in aksaraLLMConfig.__dataclass_fields__
    })

    model = aksaraLLMModel(config)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    n_params = f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    meta = {"n_params": n_params, "step": step, "val_loss": val_loss, "epoch": epoch}
    print(f"   Parameters : {n_params}")
    print(f"   Step/Epoch : {step} / {epoch}")
    print(f"   Val Loss   : {val_loss}")

    print(f"\n📝 Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AksaraTokenizer.from_pretrained(args.tokenizer_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n🔄 Exporting to standard HF format (config.json + safetensors)...")
        export_to_hf(model, config, tmpdir, tokenizer=tokenizer)

        cfg_for_card = dict(config.__dict__)
        cfg_for_card["vocab_size"] = tokenizer.vocab_size
        card = make_model_card(args.repo, cfg_for_card, meta, is_sft=args.sft)
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
        description="AksaraLLM — Upload model ke HuggingFace Hub (standard transformers format)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path ke checkpoint (.pt)")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path ke AksaraTokenizer yang dipakai untuk melatih checkpoint ini")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace repo ID, contoh: AksaraLLM/aksarallm-mini")
    parser.add_argument("--commit", type=str, default="Upload AksaraLLM model",
                        help="Pesan commit")
    parser.add_argument("--sft", action="store_true",
                        help="Tandai sebagai model SFT (beda model card)")
    args = parser.parse_args()
    upload(args)


if __name__ == "__main__":
    main()
