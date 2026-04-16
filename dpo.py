"""
AksaraLLM — Direct Preference Optimization (DPO)
=================================================
Ngajarin Kiel untuk memilih jawaban yang BAGUS dan menghindari yang JELEK,
menggunakan data hh-rlhf (chosen vs rejected pairs).

Urutan training yang benar:
  1. Pre-training  → train.py  (sudah selesai ✅)
  2. SFT           → sft.py    (Alpaca 52K ✅)
  3. DPO           → dpo.py    (hh-rlhf 80K, langkah ini!)

Cara pakai di Colab:
  python dpo.py \\
    --sft-checkpoint /content/drive/MyDrive/.../sft_best.pt \\
    --data /content/drive/MyDrive/.../translated_hh_rlhf_shard_2.jsonl \\
    --output-dir /content/drive/MyDrive/.../dpo

Referensi paper DPO:
  "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
  Rafailov et al., 2023 — https://arxiv.org/abs/2305.18290
"""

import os
import json
import math
import time
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from aksarallm.model import aksaraLLMModel
from aksarallm.config import aksaraLLMConfig, CONFIGS


# ════════════════════════════════════════════════════════════════════════════
# 1. DATASET
# ════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """### Instruksi:
{instruction}

### Jawaban:
"""


class HHRLHFDataset(Dataset):
    """
    Dataset untuk DPO dari format hh-rlhf.
    Format JSONL: {"chosen": "...", "rejected": "...", "original_idx": N}
    """

    def __init__(self, filepath: str, tokenizer, max_seq_len: int = 256):
        self.tokenizer  = tokenizer
        self.max_seq_len = max_seq_len
        self.samples    = []

        print(f"📦 Memuat dataset DPO dari {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item    = json.loads(line)
                    chosen  = item.get("chosen", "").strip()
                    rejected = item.get("rejected", "").strip()

                    # Skip kalau salah satu kosong
                    if not chosen or not rejected:
                        continue

                    # Skip kalau dua-duanya sama persis
                    if chosen == rejected:
                        continue

                    self.samples.append({
                        "chosen":   chosen,
                        "rejected": rejected,
                    })
                except json.JSONDecodeError:
                    continue

        print(f"✅ DPO dataset dimuat: {len(self.samples):,} pairs (chosen/rejected)")

    def __len__(self):
        return len(self.samples)

    def _encode(self, text: str):
        """Tokenize & truncate."""
        ids = self.tokenizer.encode(
            text,
            max_length=self.max_seq_len,
            truncation=True,
        )
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.samples[idx]
        chosen_ids  = self._encode(item["chosen"])
        rejected_ids = self._encode(item["rejected"])
        return chosen_ids, rejected_ids


def dpo_collate_fn(batch):
    """Padding batch chosen & rejected masing-masing."""
    chosen_list, rejected_list = zip(*batch)

    def pad_batch(seqs):
        max_len = max(s.size(0) for s in seqs)
        padded  = torch.full((len(seqs), max_len), 0, dtype=torch.long)
        mask    = torch.zeros(len(seqs), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            L = s.size(0)
            padded[i, :L] = s
            mask[i, :L]   = True
        return padded, mask

    chosen_ids, chosen_mask     = pad_batch(chosen_list)
    rejected_ids, rejected_mask = pad_batch(rejected_list)
    return chosen_ids, chosen_mask, rejected_ids, rejected_mask


# ════════════════════════════════════════════════════════════════════════════
# 2. DPO LOSS
# ════════════════════════════════════════════════════════════════════════════

def get_log_probs(model, input_ids, attention_mask):
    """
    Hitung log-probability per sequence dari model.
    Returns: (batch_size,) tensor of summed log-probs.
    """
    B, T = input_ids.shape

    # Truncate ke max_seq_len model
    max_len = model.config.max_seq_len
    if T > max_len:
        input_ids     = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        T = max_len

    with torch.no_grad() if not model.training else torch.enable_grad():
        logits, _ = model(input_ids)   # (B, T, vocab_size)

    # Shift: prediksi token t+1 dari token t
    shift_logits = logits[:, :-1, :]          # (B, T-1, V)
    shift_labels = input_ids[:, 1:]           # (B, T-1)
    shift_mask   = attention_mask[:, 1:].float()  # (B, T-1)

    # Log probability untuk setiap token
    log_probs_all = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
    token_log_probs = log_probs_all.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)

    # Mask padding, sum per sequence
    seq_log_probs = (token_log_probs * shift_mask).sum(dim=1)  # (B,)
    return seq_log_probs


def dpo_loss(
    policy_model,
    ref_model,
    chosen_ids, chosen_mask,
    rejected_ids, rejected_mask,
    beta: float = 0.1,
):
    """
    Hitung DPO loss.

    DPO Loss = -log σ(β * (log π(chosen) - log π_ref(chosen))
                       - β * (log π(rejected) - log π_ref(rejected)))

    Intuisi: dorong model untuk lebih prefer chosen dan less prefer rejected
    dibandingkan reference model (SFT model yang frozen).
    """
    # Log probs dari policy model (yang sedang ditraining)
    policy_chosen_logps  = get_log_probs(policy_model, chosen_ids, chosen_mask)
    policy_rejected_logps = get_log_probs(policy_model, rejected_ids, rejected_mask)

    # Log probs dari reference model (frozen SFT)
    with torch.no_grad():
        ref_chosen_logps    = get_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_logps  = get_log_probs(ref_model, rejected_ids, rejected_mask)

    # Hitung advantages
    chosen_advantage  = policy_chosen_logps - ref_chosen_logps
    rejected_advantage = policy_rejected_logps - ref_rejected_logps

    # DPO loss
    logits = beta * (chosen_advantage - rejected_advantage)
    loss   = -F.logsigmoid(logits).mean()

    # Metrics tambahan untuk monitoring
    with torch.no_grad():
        chosen_rewards  = beta * chosen_advantage
        rejected_rewards = beta * rejected_advantage
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        margin   = (chosen_rewards - rejected_rewards).mean()

    return loss, {
        "accuracy": accuracy.item(),
        "margin":   margin.item(),
        "chosen_reward":   chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(args):
    # ── Setup device ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # ⚡ auto-tune convolutions
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        use_fp16 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon MPS")
        use_fp16 = False
    else:
        device = torch.device("cpu")
        print("💻 CPU mode")
        use_fp16 = False

    # ── Load SFT checkpoint ───────────────────────────────────────────────
    print(f"\n📥 Memuat SFT checkpoint: {args.sft_checkpoint}")
    ckpt = torch.load(args.sft_checkpoint, map_location=device, weights_only=False)

    cfg_dict = ckpt.get("config", {})
    config   = aksaraLLMConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in aksaraLLMConfig.__dataclass_fields__
    })
    config.max_seq_len = min(args.max_seq_len, config.max_seq_len)

    # Policy model (yang akan ditraining)
    policy = aksaraLLMModel(config).to(device)
    state  = ckpt.get("model_state_dict", ckpt)
    state  = {k.replace("module.", ""): v for k, v in state.items()}
    policy.load_state_dict(state, strict=False)
    print(f"✅ Policy model: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params")

    # Reference model (frozen copy dari SFT — TIDAK ditraining)
    ref_model = copy.deepcopy(policy).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print("🔒 Reference model: frozen (tidak ikut training)")

    # ⚡ Compile models (PyTorch 2.x — 20-40% speedup)
    if args.compile and hasattr(torch, "compile"):
        print("⚡ torch.compile() aktif — warm-up pertama mungkin lambat...")
        policy    = torch.compile(policy,    mode="reduce-overhead")
        ref_model = torch.compile(ref_model, mode="reduce-overhead")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    dataset = HHRLHFDataset(args.data, tokenizer, config.max_seq_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dpo_collate_fn,
        num_workers=4,                          # ⚡ parallel data loading
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,                      # ⚡ prefetch next batch
        persistent_workers=True,                # ⚡ keep workers alive
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    total_steps  = args.epochs * len(loader)
    warmup_steps = min(50, total_steps // 20)

    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── Output dir ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"🚀 MULAI DPO — {args.epochs} epoch | β={args.beta}")
    print(f"   Dataset : {len(dataset):,} pairs")
    print(f"   Batch   : {args.batch_size}")
    print(f"   LR      : {args.lr}")
    print(f"{'='*55}\n")

    global_step = 0
    best_acc    = 0.0

    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_acc  = 0.0
        t0 = time.time()

        for step, (chosen_ids, chosen_mask, rejected_ids, rejected_mask) in enumerate(loader):
            chosen_ids    = chosen_ids.to(device)
            chosen_mask   = chosen_mask.to(device)
            rejected_ids  = rejected_ids.to(device)
            rejected_mask = rejected_mask.to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                loss, metrics = dpo_loss(
                    policy, ref_model,
                    chosen_ids, chosen_mask,
                    rejected_ids, rejected_mask,
                    beta=args.beta,
                )

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()
            epoch_acc  += metrics["accuracy"]
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / (step + 1)
                avg_acc  = epoch_acc / (step + 1)
                lr_now   = scheduler.get_last_lr()[0] * args.lr
                elapsed  = time.time() - t0
                print(
                    f"Epoch {epoch+1} | Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%} | "
                    f"Margin: {metrics['margin']:.4f} | LR: {lr_now:.2e} | "
                    f"{(step+1)*args.batch_size/elapsed:.1f} pairs/s"
                )

        avg_loss = epoch_loss / len(loader)
        avg_acc  = epoch_acc / len(loader)
        print(f"\n✨ Epoch {epoch+1} selesai | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%}")
        print(f"   Accuracy artinya: model memilih 'chosen' dengan benar sebanyak {avg_acc:.0%} dari waktu")

        ckpt_data = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "accuracy": avg_acc,
            "beta": args.beta,
            "config": config.__dict__,
        }

        epoch_path = out_dir / f"dpo_epoch_{epoch+1}.pt"
        torch.save(ckpt_data, epoch_path)
        print(f"💾 Disimpan: {epoch_path}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(ckpt_data, out_dir / "dpo_best.pt")
            print(f"🏆 Best model! Accuracy: {best_acc:.2%}")

        print()

    print("=" * 55)
    print("🎉 DPO SELESAI SEMPURNA!")
    print(f"   Best Accuracy : {best_acc:.2%}")
    print(f"   Model         : {out_dir / 'dpo_best.pt'}")
    print("=" * 55)
    print("\nUpload ke HuggingFace:")
    print(f"  python upload_to_hf.py --checkpoint {out_dir / 'dpo_best.pt'} \\")
    print(f"    --repo AksaraLLM/Kiel-Mini-59M-DPO --sft")


# ════════════════════════════════════════════════════════════════════════════
# 4. ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AksaraLLM DPO — Alignment dengan data preferensi Indonesia"
    )
    parser.add_argument("--sft-checkpoint", type=str, required=True,
                        help="Path ke SFT checkpoint (sft_best.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path ke file JSONL hh-rlhf (translated_hh_rlhf_shard_2.jsonl)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="LR sangat kecil untuk DPO (default: 5e-7)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Temperatur DPO — lebih besar = lebih konservatif")
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Panjang sequence max — 128 sudah cukup untuk teks terjemahan pendek")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--compile", action="store_true",
                        help="Aktifkan torch.compile() untuk speedup 20-40%")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Batasi jumlah pairs (0 = semua). Contoh: 80000 untuk training lebih cepat")
    args = parser.parse_args()
    train(args)
