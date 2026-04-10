"""
aksaraLLM — Upload trained model to HuggingFace Hub

Usage:
    python upload_to_hf.py \
        --checkpoint checkpoints/aksarallm-nano/best_model.pt \
        --repo aksaraLLM/aksaraLLM-nano \
        --commit "Initial release: aksaraLLM-nano (8M params)"
"""
import argparse
import json
import os
import shutil
import tempfile

import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

from aksarallm.config import aksaraLLMConfig
from aksarallm.model import aksaraLLMModel


def upload(checkpoint_path: str, repo_id: str, commit_message: str):
    """Upload model to HuggingFace Hub."""
    
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    cfg = checkpoint["config"]
    step = checkpoint.get("step", "unknown")
    val_loss = checkpoint.get("val_loss", "unknown")
    
    # Create temp dir for upload
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model weights
        torch.save(checkpoint["model_state_dict"], os.path.join(tmpdir, "model.pt"))
        
        # Save full checkpoint
        torch.save(checkpoint, os.path.join(tmpdir, "checkpoint.pt"))
        
        # Save config
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        
        # Create model card
        n_params = "unknown"
        try:
            config = aksaraLLMConfig(**{k: v for k, v in cfg.items() if k != "dropout" and k != "bias"})
            n_params = f"{config.n_params / 1e6:.1f}M"
        except:
            pass
        
        model_card = f"""---
language:
  - en
  - id
license: apache-2.0
tags:
  - aksaraLLM
  - language-model
  - transformer
  - open-source
  - from-scratch
library_name: pytorch
---

# {repo_id.split("/")[-1]}

A small language model trained from scratch as part of the [aksaraLLM](https://github.com/aksaraLLM) project.

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only Transformer |
| **Parameters** | ~{n_params} |
| **Layers** | {cfg.get("n_layers", "?")} |
| **Heads** | {cfg.get("n_heads", "?")} |
| **Hidden Size** | {cfg.get("n_embd", "?")} |
| **Context Length** | {cfg.get("max_seq_len", "?")} |
| **Tokenizer** | GPT-2 (50,257 vocab) |
| **Training Steps** | {step} |
| **Val Loss** | {val_loss} |

## Features
- RMSNorm (pre-norm)
- Rotary Position Embeddings (RoPE)
- SwiGLU activation
- Weight-tied embeddings

## Usage

```python
import torch
from aksarallm.config import aksaraLLMConfig
from aksarallm.model import aksaraLLMModel
from transformers import AutoTokenizer

# Load
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
config = aksaraLLMConfig(**checkpoint["config"])
model = aksaraLLMModel(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

## Training

Trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

```bash
git clone https://github.com/aksaraLLM/aksaraLLM
cd aksaraLLM
pip install -r requirements.txt
python train.py --size nano
```

## License

Apache License 2.0

## About aksaraLLM

aksaraLLM is a community-driven initiative to build a truly open-source LLM.
Everything is open: code, data, training methodology, and models.

- GitHub: [github.com/aksaraLLM](https://github.com/aksaraLLM)
- Discord: [discord.gg/aksarallm](https://discord.gg/aksarallm)
"""
        
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write(model_card)
        
        # Create repo and upload
        print(f"\n🚀 Uploading to HuggingFace: {repo_id}")
        api = HfApi()
        
        try:
            create_repo(repo_id, exist_ok=True, repo_type="model")
        except Exception as e:
            print(f"⚠️  Repo creation: {e}")
        
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        
        print(f"\n✅ Model uploaded!")
        print(f"🔗 https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload aksaraLLM to HuggingFace")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True, help="HF repo id (e.g., aksaraLLM/aksaraLLM-nano)")
    parser.add_argument("--commit", type=str, default="Upload aksaraLLM model")
    
    args = parser.parse_args()
    upload(args.checkpoint, args.repo, args.commit)


if __name__ == "__main__":
    main()
