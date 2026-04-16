"""
aksaraLLM — Training Script v2 (Resilient Edition)

Fitur Utama:
  - Resume training dari checkpoint (anti-mati Colab)
  - Continue pre-training dengan dataset baru
  - Mixed precision (fp16) untuk kecepatan 2x di GPU
  - Auto-save ke Google Drive
  - GPU keepalive terintegrasi

Usage:
    # Training dari nol
    python train.py --size mini --batch-size 16

    # Resume training yang terputus
    python train.py --size mini --batch-size 16 --resume checkpoints/aksarallm-mini/latest.pt

    # Continue pre-training dengan dataset baru
    python train.py --size mini --batch-size 16 --resume best_model.pt --dataset wikipedia --max-steps 20000

    # Simpan ke Google Drive (Colab)
    python train.py --size mini --batch-size 16 --output-dir /content/drive/MyDrive/aksaraLLM
"""
import os
import sys
import time
import json
import argparse
import math

import torch
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer

from aksarallm.config import aksaraLLMConfig, CONFIGS
from aksarallm.model import aksaraLLMModel
from aksarallm.data import create_dataloaders


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔥 Using CUDA: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU (training will be slower)")
    return device


def get_lr(step: int, config: aksaraLLMConfig) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    if step > config.max_steps:
        return config.learning_rate * 0.1
    
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * 0.1 + coeff * (config.learning_rate - config.learning_rate * 0.1)


@torch.no_grad()
def evaluate(model, val_loader, device, config) -> float:
    """Run evaluation and return average loss."""
    model.eval()
    losses = []
    
    for i, batch in enumerate(val_loader):
        if i >= config.eval_steps:
            break
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        
        if device.type == "cuda":
            with autocast("cuda"):
                _, loss = model(input_ids, targets)
        else:
            _, loss = model(input_ids, targets)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def save_checkpoint(model, optimizer, scaler, step, config, val_loss, path):
    """Save a training checkpoint with full state for resuming."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "step": step,
        "val_loss": val_loss,
        "config": {
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_embd": config.n_embd,
            "n_inner": config.n_inner,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "dropout": config.dropout,
            "bias": config.bias,
        },
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Load checkpoint and return the step number."""
    print(f"📂 Memuat checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("   ✅ Optimizer state dimuat (training dilanjutkan persis)")
        except Exception:
            print("   ⚠️ Optimizer state tidak cocok (memulai optimizer baru)")
    
    if scaler and checkpoint.get("scaler_state_dict"):
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception:
            pass
    
    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"   📍 Melanjutkan dari step {step} (val_loss: {val_loss:.4f})")
    return step, val_loss


def train(config: aksaraLLMConfig, resume_path: str = None):
    """Main training loop with resume support."""
    
    print("=" * 60)
    print("🧠 aksaraLLM Training v2 (Resilient Edition)")
    print("=" * 60)
    print(f"Config: {config.n_layers}L / {config.n_heads}H / {config.n_embd}D")
    print(f"Estimated params: ~{config.n_params / 1e6:.1f}M")
    print(f"Max steps: {config.max_steps}")
    print(f"Batch size: {config.batch_size} × {config.gradient_accumulation_steps} (accum)")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Dataset: {config.dataset_name}")
    if resume_path:
        print(f"📂 Resume from: {resume_path}")
    print("=" * 60)
    
    device = get_device()
    use_amp = device.type == "cuda"  # Mixed precision hanya untuk GPU NVIDIA
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data (batasi agar tokenisasi cepat)
    print("\n📊 Loading data...")
    max_train = min(config.max_steps * 50, 500000)
    train_loader, val_loader = create_dataloaders(
        config, tokenizer=tokenizer,
        max_train_samples=max_train,
        max_val_samples=2000,
    )
    
    # Create model
    print("\n🏗️ Building model...")
    model = aksaraLLMModel(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    
    # Mixed precision scaler (fp16 = 2x lebih cepat di GPU)
    scaler = GradScaler("cuda") if use_amp else None
    
    # Resume dari checkpoint kalau ada
    start_step = 0
    best_val_loss = float("inf")
    
    if resume_path and os.path.exists(resume_path):
        start_step, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scaler
        )
    
    # Training loop
    remaining = config.max_steps - start_step
    print(f"\n🚀 Memulai training dari step {start_step}... ({remaining} steps tersisa)\n")
    
    model.train()
    step = start_step
    train_losses = []
    start_time = time.time()
    data_iter = iter(train_loader)
    
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, "training_log.jsonl")
    latest_path = os.path.join(config.output_dir, "latest.pt")
    
    while step < config.max_steps:
        # Get batch (infinite dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        
        # Forward pass (dengan mixed precision kalau GPU)
        if use_amp:
            with autocast("cuda"):
                _, loss = model(input_ids, targets)
                loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            _, loss = model(input_ids, targets)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                lr = get_lr(step, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                lr = get_lr(step, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Logging
        current_loss = loss.item() * config.gradient_accumulation_steps
        train_losses.append(current_loss)
        
        if (step + 1) % config.log_interval == 0:
            avg_loss = sum(train_losses[-config.log_interval:]) / config.log_interval
            elapsed = time.time() - start_time
            steps_done = step + 1 - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            eta = (config.max_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
            
            lr = get_lr(step, config)
            
            print(
                f"  Step {step + 1:>6d}/{config.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Speed: {steps_per_sec:.1f} steps/s | "
                f"ETA: {eta / 60:.1f} min"
            )
            
            with open(log_file, "a") as f:
                json.dump({
                    "step": step + 1,
                    "train_loss": avg_loss,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                }, f)
                f.write("\n")
        
        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, config)
            print(f"\n  📏 Eval @ step {step + 1}: val_loss = {val_loss:.4f}")
            
            # Generate sample
            prompt = "Once upon a time"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            generated = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8)
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"  📝 Sample: {generated_text[:200]}...")
            print()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scaler, step + 1, config, val_loss,
                    os.path.join(config.output_dir, "best_model.pt"),
                )
                print(f"  🏆 Model terbaik baru! val_loss={val_loss:.4f}")
            
            # Selalu simpan latest (untuk resume)
            save_checkpoint(
                model, optimizer, scaler, step + 1, config, val_loss,
                latest_path,
            )
            print(f"  💾 Auto-save: {latest_path}")
            
            with open(log_file, "a") as f:
                json.dump({
                    "step": step + 1,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                }, f)
                f.write("\n")
        
        # Periodic checkpoint (tiap 1000 steps)
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scaler, step + 1, config, best_val_loss,
                os.path.join(config.output_dir, f"checkpoint_step{step + 1}.pt"),
            )
            # Selalu update latest.pt juga
            save_checkpoint(
                model, optimizer, scaler, step + 1, config, best_val_loss,
                latest_path,
            )
            print(f"  💾 Checkpoint + auto-save @ step {step + 1}")
        
        step += 1
    
    # Save final model
    save_checkpoint(
        model, optimizer, scaler, step, config, best_val_loss,
        os.path.join(config.output_dir, "final_model.pt"),
    )
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Total time: {total_time / 60:.1f} minutes")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved in: {config.output_dir}/")
    print("=" * 60)
    print(f"\n🚀 Try your model: python demo.py --checkpoint {config.output_dir}/best_model.pt")
    print(f"🔄 Resume training: python train.py --size mini --resume {latest_path}")


def main():
    parser = argparse.ArgumentParser(description="Train aksaraLLM v2")
    parser.add_argument(
        "--size", type=str, default="nano",
        choices=["nano", "micro", "mini", "small"],
        help="Model size preset (default: nano)"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--resume", type=str, default=None, help="Path ke checkpoint untuk melanjutkan training")
    
    args = parser.parse_args()
    
    config = CONFIGS[args.size]
    
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        config.output_dir = f"checkpoints/aksarallm-{args.size}"
    if args.dataset:
        config.dataset_name = args.dataset
    
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
