<p align="center">
  <h1 align="center">🧠 aksaraLLM</h1>
  <p align="center">A truly open-source Large Language Model — architecture, training, and inference.</p>
</p>

<p align="center">
  <a href="https://github.com/aksaraLLM/aksaraLLM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://discord.gg/aksarallm"><img src="https://img.shields.io/badge/Discord-Join-7289da?logo=discord" alt="Discord"></a>
  <a href="https://huggingface.co/aksaraLLM"><img src="https://img.shields.io/badge/🤗-Models-yellow" alt="HuggingFace"></a>
</p>

---

## 🚀 Quick Start — Train Your Own LLM in 15 Minutes

```bash
# 1. Clone the repo
git clone https://github.com/aksaraLLM/aksaraLLM.git
cd aksaraLLM

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train a nano model (~8M params, ~15 min on Mac)
python train.py --size nano

# 4. Try your model!
python demo.py --checkpoint checkpoints/aksarallm-nano/best_model.pt
```

That's it! You just trained an LLM from scratch. 🎉

## Model Sizes

| Size | Params | Layers | Heads | Hidden | Train Time (Mac M1) |
|------|--------|--------|-------|--------|---------------------|
| `nano` | ~8M | 4 | 4 | 256 | ~15 min |
| `micro` | ~15M | 6 | 6 | 384 | ~30 min |
| `mini` | ~40M | 8 | 8 | 512 | ~2 hours |
| `small` | ~85M | 12 | 12 | 768 | ~6 hours |

## Architecture

aksaraLLM uses a modern decoder-only Transformer with:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• RMSNorm (pre-normalization)
• Rotary Position Embeddings (RoPE)
• SwiGLU activation function
• Weight-tied embeddings
• No bias terms
• Cosine LR schedule with warmup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Training

```bash
# Train with different sizes
python train.py --size nano      # Quick experiment
python train.py --size micro     # Better quality
python train.py --size mini      # Even better (needs patience)

# Custom training
python train.py --size micro --max-steps 10000 --batch-size 16

# Custom dataset
python train.py --size nano --dataset "wikitext/wikitext-2-raw-v1"
```

### What happens during training:
1. 📥 Downloads TinyStories dataset from HuggingFace
2. 📝 Tokenizes with GPT-2 tokenizer
3. 🏗️ Builds transformer model from scratch
4. 🚀 Trains with AdamW optimizer + cosine LR
5. 📏 Evaluates periodically + generates sample text
6. 💾 Saves best checkpoint

### Hardware Support
- ✅ **Apple Silicon (M1/M2/M3)** — uses MPS acceleration
- ✅ **NVIDIA GPU** — uses CUDA
- ✅ **CPU** — works but slower

## Demo

```bash
# Interactive text completion
python demo.py --checkpoint checkpoints/aksarallm-nano/best_model.pt

# Chat mode (experimental)
python demo.py --checkpoint checkpoints/aksarallm-nano/best_model.pt --mode chat

# Single prompt
python demo.py --checkpoint checkpoints/aksarallm-nano/best_model.pt \
  --prompt "Once upon a time" --max-tokens 200
```

## Upload to HuggingFace

```bash
# Login to HuggingFace
huggingface-cli login

# Upload your trained model
python upload_to_hf.py \
  --checkpoint checkpoints/aksarallm-nano/best_model.pt \
  --repo aksaraLLM/aksaraLLM-nano
```

## Project Structure

```
aksaraLLM/
├── aksarallm/
│   ├── __init__.py
│   ├── config.py        # Model configurations (nano/micro/mini/small)
│   ├── model.py         # Transformer architecture (RoPE, RMSNorm, SwiGLU)
│   └── data.py          # Data loading & tokenization pipeline
├── train.py             # Training script
├── demo.py              # Interactive demo
├── upload_to_hf.py      # Upload to HuggingFace Hub
├── requirements.txt
├── LICENSE
└── README.md
```

## Related Repositories

- [`aksara-data`](https://github.com/aksaraLLM/aksara-data) — Data curation pipeline
- [`aksara-train`](https://github.com/aksaraLLM/aksara-train) — Distributed training (for scaling up)
- [`aksara-eval`](https://github.com/aksaraLLM/aksara-eval) — Evaluation suite
- [`aksara-tokenizer`](https://github.com/aksaraLLM/aksara-tokenizer) — Custom tokenizer
- [`community`](https://github.com/aksaraLLM/community) — Governance & RFCs

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/aksaraLLM/community/blob/main/CONTRIBUTING.md).

## Citation

```bibtex
@software{aksarallm2026,
  title={aksaraLLM: A Truly Open-Source Large Language Model},
  author={aksaraLLM Community},
  year={2026},
  url={https://github.com/aksaraLLM/aksaraLLM}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
