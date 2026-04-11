"""
aksaraLLM — Training Script

Train a small language model from scratch on your machine.

Usage:
    python train.py --size super_micro_nano # ~2M params
    python train.py --size super_nano       # ~4M params
    python train.py --size nano     # ~8M params, ~15 min on Mac
    python train.py --size micro    # ~15M params, ~30 min on Mac
    python train.py --size mini     # ~40M params, ~2 hours on Mac
"""
import os
import sys
import time
import json
import argparse
import math

import torch
from transformers import AutoTokenizer

from aksarallm.config import aksaraLLMConfig, CONFIGS
from aksarallm.model import aksaraLLMModel
from aksarallm.data import create_dataloaders


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using CUDA: {torch.cuda.get_device_name()}")
        return device
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, 'get_device_name') else 'Intel GPU'
        print(f"🚀 Using Intel XPU (iGPU/dGPU): {name}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon (MPS)")
        return device
    else:
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                print(f"🚀 Using DirectML ({torch_directml.device_name(0)})")
                return device
        except ImportError:
            pass

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
        _, loss = model(input_ids, targets)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def save_checkpoint(model, optimizer, step, config, val_loss, path):
    """Save a training checkpoint safely to avoid pickler memory explosion."""
    import gc
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # We serialize entirely through CPU variables to prevent DML driver leaks during Pickling
    cpu_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Optional optimizer save. Skip optimizer states for mini/small models to save memory!
    # Because saving 400MB+ nested dict crashes on some systems.
    
    torch.save({
        "model_state_dict": cpu_model_state,
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
    
    del cpu_model_state
    gc.collect()
    print(f"💾 Checkpoint saved: {path}")


def train(config: aksaraLLMConfig):
    """Main training loop."""
    
    print("=" * 60)
    print("🧠 aksaraLLM Training")
    print("=" * 60)
    print(f"Config: {config.n_layers}L / {config.n_heads}H / {config.n_embd}D")
    print(f"Estimated params: ~{config.n_params / 1e6:.1f}M")
    print(f"Max steps: {config.max_steps}")
    print(f"Batch size: {config.batch_size} × {config.gradient_accumulation_steps} (accum)")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Dataset: {config.dataset_name}")
    print("=" * 60)
    
    device = get_device()
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("\n📊 Loading data...")
    # For nano/micro, limit training data for speed
    max_train = 50000 if config.max_steps <= 5000 else None
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
    
    # Training loop
    print("\n🚀 Starting training...\n")
    
    model.train()
    step = 0
    best_val_loss = float("inf")
    train_losses = []
    start_time = time.time()
    data_iter = iter(train_loader)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Training log
    log_file = os.path.join(config.output_dir, "training_log.jsonl")
    
    while step < config.max_steps:
        # Get batch (with infinite dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        
        # Forward pass
        _, loss = model(input_ids, targets)
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Update learning rate
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
            steps_per_sec = (step + 1) / elapsed
            eta = (config.max_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
            
            lr = get_lr(step, config)
            
            print(
                f"  Step {step + 1:>6d}/{config.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Speed: {steps_per_sec:.1f} steps/s | "
                f"ETA: {eta / 60:.1f} min"
            )
            
            # Log to file
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
                    model, optimizer, step + 1, config, val_loss,
                    os.path.join(config.output_dir, "best_model.pt"),
                )
            
            # Log eval
            with open(log_file, "a") as f:
                json.dump({
                    "step": step + 1,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                }, f)
                f.write("\n")
        
        # Periodic checkpoint
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, step + 1, config, best_val_loss,
                os.path.join(config.output_dir, f"checkpoint_step{step + 1}.pt"),
            )
        
        step += 1
    
    # Save final model
    save_checkpoint(
        model, optimizer, step, config, best_val_loss,
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


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    parser = argparse.ArgumentParser(description="Train aksaraLLM")
    parser.add_argument(
        "--size", type=str, default="nano",
        choices=["super_micro_nano", "super_nano", "nano", "micro", "mini", "small"],
        help="Model size preset (default: nano)"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    
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
    
    train(config)


if __name__ == "__main__":
    main()
