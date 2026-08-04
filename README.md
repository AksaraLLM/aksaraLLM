# 🇮🇩 aksaraLLM

**Large Language Model Bahasa Indonesia — dilatih dari nol, bukan fine-tune model tertutup.**

Repo ini berisi arsitektur model, tokenizer utilities, dan pipeline pre-training +
alignment (SFT/DPO) dari nol untuk AksaraLLM. Tidak ada script di sini yang
mem-fine-tune model pihak ketiga (Qwen, LLaMA, dll) — kalau kamu mencari itu,
bukan di sini tempatnya. Lihat [community/RFC-001](https://github.com/AksaraLLM/community/blob/main/rfcs/RFC-001-model-architecture.md)
untuk detail keputusan arsitektur.

## Arsitektur

Decoder-only Transformer bergaya modern:
- **RoPE** (Rotary Position Embeddings)
- **RMSNorm** (pre-normalization)
- **SwiGLU** activation
- **GQA** (Grouped-Query Attention) untuk skala ≥200M
- Weight-tied input/output embeddings
- Gradient checkpointing untuk hemat VRAM

Lihat [`aksarallm/model.py`](aksarallm/model.py) dan [`aksarallm/config.py`](aksarallm/config.py)
untuk implementasi dan preset skala (nano/micro/mini/small/medium/large/xlarge,
~10M sampai ~1B parameter).

## Struktur Repo

| Path | Fungsi |
|---|---|
| `aksarallm/model.py` | Arsitektur transformer |
| `aksarallm/config.py` | Preset konfigurasi skala model |
| `aksarallm/tokenizer_utils.py` | Wrapper `AksaraTokenizer` (BPE byte-level) — dilatih via [aksara-tokenizer](https://github.com/AksaraLLM/aksara-tokenizer) |
| `aksarallm/data.py` | Data loading + tokenisasi untuk pre-training |
| `aksarallm/trainer.py` | Training loop pre-training (CPU/CUDA/MPS/TPU) |
| `aksarallm/hf_export.py` | Konversi checkpoint ke format `transformers` standar (`LlamaForCausalLM`) |
| `train.py` | CLI pre-training dari nol |
| `sft.py` | Supervised fine-tuning (instruction-tuning) di atas checkpoint pre-training sendiri |
| `dpo.py` | Direct Preference Optimization (alignment) — [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) |
| `demo.py` / `demo/gradio_chat.py` | Demo CLI / web untuk chat dengan checkpoint yang sudah dilatih |
| `aksara_cli.py` | CLI interaktif dengan tools (baca file, jalankan perintah, dst) |

## Quick Start

```bash
pip install -r requirements.txt

# 1. Latih tokenizer dulu (lihat aksara-tokenizer repo), atau pakai yang sudah ada
# 2. Pre-training dari nol — default corpus: Indonesian Wikipedia
python train.py --size mini --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b

# 3. (Opsional) Instruction-tuning
python sft.py --checkpoint checkpoints/aksarallm-mini/best_model.pt \
    --data data/sft_id.jsonl --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b

# 4. (Opsional) Alignment dengan preference data
python dpo.py --sft-checkpoint checkpoints/sft/sft_best.pt \
    --data data/dpo_id.jsonl --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b

# 5. Coba hasilnya
python demo.py --checkpoint checkpoints/aksarallm-mini/best_model.pt
```

Untuk training skala besar / TPU, lihat repo [aksara-train](https://github.com/AksaraLLM/aksara-train).

## Kompatibilitas Ekosistem

Arsitektur AksaraLLM (RoPE + GQA + RMSNorm + SwiGLU + tied embeddings) secara
fungsional setara dengan LLaMA. `aksarallm.hf_export` mengonversi checkpoint
terlatih ke format `transformers.LlamaForCausalLM` standar, sehingga bisa
langsung dipakai dengan `AutoModelForCausalLM`, [aksara-eval](https://github.com/AksaraLLM/aksara-eval),
vLLM, dan konversi GGUF (`scripts/convert_gguf.sh`) — tanpa kode model kustom.

```bash
python upload_to_hf.py --checkpoint checkpoints/aksarallm-mini/best_model.pt \
    --repo AksaraLLM/aksarallm-mini --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b
```

## License

Apache License 2.0 — Bebas digunakan untuk keperluan komersial maupun riset.

## Team

**AksaraLLM** — Proyek AI Open-Source Indonesia

## Acknowledgments

- [Google TRC Program](https://sites.research.google/trc/) — TPU compute
- [Hugging Face](https://huggingface.co) — Model hosting & datasets
- Komunitas AI Indonesia 🇮🇩

---

<p align="center">
  <b>AksaraLLM — Revolusi AI Indonesia dimulai dari sini. 🇮🇩🚀</b>
</p>
