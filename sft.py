"""
AksaraLLM — Supervised Fine-Tuning (SFT) Script
=================================================
Ngajarin Kiel-Mini mengikuti instruksi menggunakan dataset Alpaca Indonesia.

Cara pakai di Google Colab:
  python sft.py --checkpoint checkpoints/best_model.pt --data translated_alpaca_id.jsonl

Format data yang didukung (JSONL):
  {"instruction": "...", "input": "...", "output": "..."}

Template prompt yang dipakai (Alpaca format):
  ### Instruksi:
  {instruction}

  ### Konteks:
  {input}

  ### Jawaban:
  {output}
"""

import os
import json
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# ── Import model kita sendiri ────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from aksarallm.model import aksaraLLMModel
from aksarallm.config import aksaraLLMConfig, CONFIGS
from aksarallm.data import load_tokenizer
from aksarallm.trainer import build_optimizer


# ════════════════════════════════════════════════════════════════════════════
# 1. DATASET
# ════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """### Instruksi:
{instruction}

### Konteks:
{input}

### Jawaban:
{output}"""

PROMPT_TEMPLATE_NO_INPUT = """### Instruksi:
{instruction}

### Jawaban:
{output}"""


class AlpacaDataset(Dataset):
    """Dataset JSONL berisi instruksi-jawaban dalam Bahasa Indonesia.

    Loss is masked over the prompt (instruction/context) tokens — only the
    response tokens contribute to the training signal. Training on the
    prompt too is a common SFT bug: the model burns capacity "learning" to
    predict text it was only ever given, not asked to generate, which dilutes
    instruction-following quality. Every serious SFT recipe (Stanford Alpaca,
    HF's SFTTrainer, InstructGPT) masks the prompt for this reason.
    """

    def __init__(self, filepath: str, tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        print(f"📦 Memuat dataset dari {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Support format dengan atau tanpa field 'input'
                    instruction = item.get("instruction", "").strip()
                    inp         = item.get("input", "").strip()
                    output      = item.get("output", "").strip()

                    if not instruction or not output:
                        continue

                    self.samples.append((instruction, inp, output))
                except json.JSONDecodeError:
                    continue

        print(f"✅ Dataset dimuat: {len(self.samples):,} sampel")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        instruction, inp, output = self.samples[idx]
        if inp:
            full_text = PROMPT_TEMPLATE.format(instruction=instruction, input=inp, output=output)
            prompt_text = PROMPT_TEMPLATE.format(instruction=instruction, input=inp, output="")
        else:
            full_text = PROMPT_TEMPLATE_NO_INPUT.format(instruction=instruction, output=output)
            prompt_text = PROMPT_TEMPLATE_NO_INPUT.format(instruction=instruction, output="")

        ids = self.tokenizer.encode(full_text, max_length=self.max_seq_len, truncation=True)
        # Tokenized separately from full_text — BPE merges can shift a token or
        # two right at the boundary, but this is the standard, widely-used
        # approximation (e.g. Stanford Alpaca's own train.py does the same).
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = min(len(prompt_ids), len(ids))

        ids = torch.tensor(ids, dtype=torch.long)
        inputs = ids[:-1]
        targets = ids[1:].clone()
        # targets[j] predicts ids[j+1]; mask every target position whose
        # predicted token still falls inside the prompt.
        mask_len = max(prompt_len - 1, 0)
        targets[:mask_len] = -1
        return inputs, targets


def collate_fn(batch):
    """Padding batch ke panjang yang sama."""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    padded_inputs  = torch.full((len(inputs), max_len), 0, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), -1, dtype=torch.long)  # -1 = ignore

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = inp.size(0)
        padded_inputs[i, :L]  = inp
        padded_targets[i, :L] = tgt

    return padded_inputs, padded_targets


# ════════════════════════════════════════════════════════════════════════════
# 2. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(args):
    # ── Setup device ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   VRAM: {props.total_memory / 1e9:.1f} GB")
        use_fp16 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon MPS aktif")
        use_fp16 = False
    else:
        device = torch.device("cpu")
        print("💻 CPU mode (akan lambat, sabar ya)")
        use_fp16 = False

    # ── Load checkpoint pre-trained ───────────────────────────────────────
    print(f"\n📥 Memuat checkpoint dari: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Rekonstruksi config dari checkpoint
    if "config" in ckpt:
        cfg_dict = ckpt["config"]
        config = aksaraLLMConfig(**cfg_dict)
    else:
        # Fallback: pakai config 'mini'
        print("⚠️  Config tidak ada di checkpoint, pakai config 'mini'")
        config = CONFIGS["mini"]

    # Override max_seq_len untuk SFT (bisa lebih panjang)
    config.max_seq_len = min(args.max_seq_len, config.max_seq_len)

    # ── Buat model & muat weight ──────────────────────────────────────────
    model = aksaraLLMModel(config).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    # Bersihkan prefix 'module.' kalau ada (dari DistributedDataParallel)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️  Keys hilang: {len(missing)} (wajar kalau checkpoint beda versi)")
    print(f"✅ Model dimuat! {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameter")

    # ── Tokenizer (harus sama dengan yang dipakai saat pre-training) ───────
    config.tokenizer_path = args.tokenizer_path or config.tokenizer_path
    tokenizer = load_tokenizer(config)

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    dataset = AlpacaDataset(args.data, tokenizer, config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    # SFT pakai learning rate lebih kecil dari pre-training; decoupled weight
    # decay (none on norm/bias params) — see aksarallm.trainer.build_optimizer.
    config.weight_decay = 0.01
    config.learning_rate = args.lr
    optimizer = build_optimizer(model, config)

    total_steps = args.epochs * len(loader)
    warmup_steps = min(100, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    scaler    = GradScaler("cuda", enabled=use_fp16)

    # ── Output dir ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"🚀 MULAI SFT — {args.epochs} epoch | {total_steps} total step")
    print(f"   Dataset  : {len(dataset):,} sampel")
    print(f"   Batch    : {args.batch_size}")
    print(f"   LR       : {args.lr}")
    print(f"   Max Seq  : {config.max_seq_len}")
    print(f"{'='*55}\n")

    global_step = 0
    best_loss   = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (input_ids, targets) in enumerate(loader):
            input_ids = input_ids.to(device)
            targets   = targets.to(device)

            # Pastikan sequence tidak melebihi max_seq_len
            if input_ids.size(1) > config.max_seq_len:
                input_ids = input_ids[:, :config.max_seq_len]
                targets   = targets[:, :config.max_seq_len]

            with autocast("cuda", enabled=use_fp16):
                logits, _ = model(input_ids)
                # ignore_index=-1 skips both padding AND masked prompt tokens
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1,
                )

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            # Log progress
            if global_step % args.log_every == 0:
                elapsed  = time.time() - t0
                avg_loss = epoch_loss / (step + 1)
                lr_now   = scheduler.get_last_lr()[0] * args.lr
                samples_per_sec = (step + 1) * args.batch_size / elapsed
                print(
                    f"Epoch {epoch+1}/{args.epochs} | Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | "
                    f"{samples_per_sec:.1f} samples/s"
                )

        # ── Simpan checkpoint tiap epoch ──────────────────────────────────
        avg_loss = epoch_loss / len(loader)
        print(f"\n✨ Epoch {epoch+1} selesai | Avg Loss: {avg_loss:.4f}")

        ckpt_data = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": config.__dict__,
        }

        # Simpan checkpoint epoch ini
        epoch_path = out_dir / f"sft_epoch_{epoch+1}.pt"
        torch.save(ckpt_data, epoch_path)
        print(f"💾 Checkpoint disimpan: {epoch_path}")

        # Simpan sebagai best jika loss terbaik
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = out_dir / "sft_best.pt"
            torch.save(ckpt_data, best_path)
            print(f"🏆 Best model diperbarui! Loss: {best_loss:.4f} → {best_path}")

        print()

    print("=" * 55)
    print("🎉 SFT SELESAI SEMPURNA!")
    print(f"   Best Loss : {best_loss:.4f}")
    print(f"   Model     : {out_dir / 'sft_best.pt'}")
    print("=" * 55)
    print("\nUntuk test hasilnya:")
    print(f"  python demo.py --checkpoint {out_dir / 'sft_best.pt'} --mode chat")


# ════════════════════════════════════════════════════════════════════════════
# 3. ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AksaraLLM SFT — Fine-tuning dengan data instruksi Indonesia"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path ke checkpoint pre-trained (best_model.pt)"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path ke file JSONL dataset instruksi (translated_alpaca_id.jsonl)"
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default=None,
        help="Path ke AksaraTokenizer (harus sama dengan yang dipakai saat pre-training). "
             "Kalau kosong, pakai tokenizer_path dari checkpoint jika ada, atau fallback GPT-2."
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/sft",
        help="Folder untuk menyimpan checkpoint SFT"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Berapa epoch fine-tuning (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (turunkan jika GPU OOM)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5, lebih kecil dari pre-training)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Panjang sequence maksimum"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--log-every", type=int, default=50,
        help="Log setiap berapa step"
    )

    args = parser.parse_args()
    train(args)
