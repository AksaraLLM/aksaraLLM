---
language:
- id
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- indonesian
- aksarallm
- bahasa-indonesia
- qwen2
- sft
- dpo
- chat
base_model: Qwen/Qwen2.5-1.5B-Instruct
datasets:
- AksaraLLM/aksara-mega-sft-v5
- AksaraLLM/aksara-dpo-id-v4
model-index:
- name: aksarallm-1.5b-v2
  results: []
---

# 🇮🇩 AksaraLLM 1.5B v2

<p align="center">
  <b>Model Bahasa AI Open-Source Terbaik untuk Bahasa Indonesia</b><br>
  <i>The Best Open-Source Indonesian Language Model</i>
</p>

---

## ✨ Highlights

- 🇮🇩 **Fokus Bahasa Indonesia** — Dilatih khusus dengan 500K+ data instruksi Indonesia
- ⚡ **Ringan & Cepat** — 1.5B parameter, bisa jalan di laptop bahkan HP Android (via GGUF)
- 🛡️ **Aligned & Safe** — DPO training dengan 200K preference pairs
- 🧠 **Identitas Kuat** — Tahu siapa dirinya, menolak konten berbahaya
- 📖 **100% Open Source** — Model, data, dan kode tersedia bebas

## 📊 Model Details

| Attribute | Value |
|---|---|
| **Base Model** | Qwen2.5-1.5B-Instruct |
| **Parameters** | 1.78B |
| **Context Window** | 32K tokens |
| **Training Data** | 500K SFT + 200K DPO |
| **Training Hardware** | Google Cloud TPU v6e-4 |
| **Training Method** | Full fine-tuning (SFT → DPO) |
| **Precision** | BFloat16 |
| **License** | Apache 2.0 |

## 🚀 Quick Start

### Transformers (Python)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "AksaraLLM/aksarallm-1.5b-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas dan membantu."},
    {"role": "user", "content": "Jelaskan apa itu Pancasila!"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Ollama (Lokal)
```bash
# Segera tersedia
ollama run aksarallm
```

### llama.cpp (GGUF)
```bash
# Download GGUF dari tab "Files"
./llama-cli -m aksarallm-1.5b-v2-Q4_K_M.gguf -p "Siapa kamu?" -n 256
```

## 📚 Training Data

### SFT Data (500K+ samples)
| Sumber | Jumlah | Kategori |
|---|---|---|
| Bactrian-X Indonesian | ~80K | Instruksi umum |
| Alpaca GPT-4 Indonesian | ~52K | Instruksi umum |
| XLSum Indonesian | ~30K | Summarization |
| WikiLingua ID | ~30K | Cross-lingual |
| Paraphrase Augmentation | ~200K | Augmented |
| Math & Coding | ~3.5K | STEM |
| Identity & Safety | ~500 | Alignment |
| Other Sources | ~100K | Mixed |

### DPO Data (200K pairs)
8 strategi rejected response:
- Jawaban terlalu pendek
- Bahasa salah (English instead of Indonesian)
- Identitas salah (mengaku ChatGPT)
- Nada kasar/tidak sopan
- Jawaban minimal/malas
- Jawaban repetitif
- Menolak menjawab tanpa alasan
- Jawaban asal-asalan

## 🧪 Evaluation

| Benchmark | Score | Notes |
|---|---|---|
| Identity Test | *pending* | Apakah model tahu dirinya AksaraLLM |
| Safety Test | *pending* | Menolak konten berbahaya |
| Indonesian Knowledge | *pending* | Pancasila, sejarah, geografi |
| General QA | *pending* | Pengetahuan umum |
| Math | *pending* | Aritmatika & soal cerita |
| Coding | *pending* | Python, JavaScript |
| Fluency | *pending* | Panjang & kualitas teks |

> Hasil evaluasi akan diupdate setelah benchmark selesai. Lihat [aksarallm-eval-results](https://huggingface.co/AksaraLLM/aksarallm-eval-results) untuk detail.

## ⚠️ Limitations

- **Bukan pengganti profesional**: Jangan gunakan untuk keputusan medis, hukum, atau keuangan
- **Bisa berhalusinasi**: Model mungkin menghasilkan informasi yang salah tapi terlihat meyakinkan
- **Knowledge cutoff**: Pengetahuan terbatas pada data training
- **Bahasa daerah**: Belum mendukung bahasa Jawa, Sunda, dll.
- **Multi-turn**: Performa menurun pada percakapan yang sangat panjang

## 🏗️ Architecture

```
Qwen2.5-1.5B-Instruct (Base)
├── Transformer Decoder-only
├── 28 layers
├── Hidden size: 1536
├── Attention heads: 12 (GQA with 2 KV heads)
├── Intermediate size: 8960 (SwiGLU)
├── Vocabulary: 151,936 tokens
├── Position encoding: RoPE
├── Normalization: RMSNorm
└── Context: 32,768 tokens
```

## 📜 License

Apache License 2.0 — Bebas digunakan untuk keperluan komersial maupun riset.

## 👥 Team

**AksaraLLM** — Proyek AI Open-Source Indonesia

## 🙏 Acknowledgments

- [Qwen Team (Alibaba)](https://huggingface.co/Qwen) — Base model
- [Google TRC Program](https://sites.research.google/trc/) — TPU compute
- [Hugging Face](https://huggingface.co) — Model hosting & datasets
- Komunitas AI Indonesia 🇮🇩

---

<p align="center">
  <b>AksaraLLM — Revolusi AI Indonesia dimulai dari sini. 🇮🇩🚀</b>
</p>
