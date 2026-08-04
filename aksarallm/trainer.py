"""
aksaraLLM — Pretraining loop (shared by aksaraLLM/train.py and aksara-train/pretrain.py)

From-scratch pretraining only — no fine-tuning of another base model lives
here. Supports CUDA, Apple Silicon (MPS), CPU, and TPU (via torch_xla, when
available) with resume-from-checkpoint.

The TPU branch mirrors the device/step pattern the project has previously
run successfully on TPU v4-8/v6e-4 (MpDeviceLoader, xm.mark_step() per step,
xm.reduce_gradients() before clipping, CPU round-trip for checkpoint saves) —
it isn't executable in this environment (no TPU hardware, torch_xla isn't
installable here) but wiring is intentionally identical to that proven
pattern. Only the CUDA/CPU path has been run in this session.
"""
import os
import time
import json
import math

import torch
from torch.amp import autocast, GradScaler

from .config import aksaraLLMConfig
from .model import aksaraLLMModel
from .data import create_dataloaders, load_tokenizer

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as xla_pl
    _HAS_XLA = True
except ImportError:
    xm = None
    xla_pl = None
    _HAS_XLA = False


def get_device() -> torch.device:
    """Get the best available device: TPU > CUDA > MPS > CPU."""
    if _HAS_XLA:
        try:
            device = xm.xla_device()
            print(f"⚡ Using TPU: {device}")
            return device
        except Exception as e:
            print(f"⚠️  torch_xla present but no TPU device found ({e}); falling back")
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


def resolve_precision(config: aksaraLLMConfig, device: torch.device) -> str:
    """Pick a compute precision. bf16 is preferred whenever the accelerator
    supports it — unlike fp16 it has fp32's full exponent range, so it needs
    no loss-scaling and won't overflow/underflow mid-training. This is what
    modern LLM pretraining runs use by default; fp16+GradScaler is a fallback
    for older GPUs without bf16 support."""
    mode = config.precision
    if mode != "auto":
        return mode
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    if device.type == "cuda":
        return "fp16"
    return "fp32"


def build_optimizer(model: aksaraLLMModel, config: aksaraLLMConfig) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay applied only to matrix-shaped
    parameters (embeddings, linear weights) — not to RMSNorm weights or
    biases. Regularizing normalization/bias params has no theoretical
    justification and empirically hurts; this split is standard practice
    across GPT-3/nanoGPT/GPT-NeoX-style pretraining codebases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    print(
        f"   Optimizer param groups: {len(decay)} decayed tensors, "
        f"{len(no_decay)} no-decay tensors (norms/biases)"
    )

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )


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

    precision = resolve_precision(config, device)
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    use_autocast = device.type == "cuda" and autocast_dtype is not None

    for i, batch in enumerate(val_loader):
        if i >= config.eval_steps:
            break
        input_ids = batch["input_ids"].to(device) if device.type != "xla" else batch["input_ids"]
        targets = batch["targets"].to(device) if device.type != "xla" else batch["targets"]

        if use_autocast:
            with autocast("cuda", dtype=autocast_dtype):
                _, loss = model(input_ids, targets)
        else:
            _, loss = model(input_ids, targets)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def save_checkpoint(model, optimizer, scaler, step, config, val_loss, path, device=None):
    """Save a training checkpoint with full state for resuming."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # XLA tensors must be moved to CPU before torch.save().
    if device is not None and device.type == "xla":
        model.to("cpu")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "step": step,
        "val_loss": val_loss,
        "config": {
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "n_embd": config.n_embd,
            "n_inner": config.n_inner,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "dropout": config.dropout,
            "bias": config.bias,
        },
    }, path)

    if device is not None and device.type == "xla":
        model.to(device)


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


def pretrain(config: aksaraLLMConfig, resume_path: str = None, on_checkpoint=None):
    """Main from-scratch pretraining loop with resume support.

    `on_checkpoint(path, step)`, if given, is called every time a checkpoint
    is written — e.g. so a caller can upload it to HuggingFace.
    """
    print("=" * 60)
    print("🧠 aksaraLLM Pretraining (from scratch)")
    print("=" * 60)
    print(f"Config: {config.n_layers}L / {config.n_heads}H / {config.n_embd}D")
    print(f"Estimated params: ~{config.n_params / 1e6:.1f}M")
    print(f"Max steps: {config.max_steps}")
    print(f"Batch size: {config.batch_size} × {config.gradient_accumulation_steps} (accum)")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Dataset: {config.dataset_name} ({config.dataset_config or 'default'})")
    if resume_path:
        print(f"📂 Resume from: {resume_path}")
    print("=" * 60)

    device = get_device()
    is_xla = device.type == "xla"

    precision = resolve_precision(config, device)
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    use_autocast = device.type == "cuda" and autocast_dtype is not None
    use_scaler = precision == "fp16"
    print(f"⚙️  Precision: {precision}" + (" (loss-scaled)" if use_scaler else ""))

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = load_tokenizer(config)
    if config.vocab_size != tokenizer.vocab_size:
        print(f"   Adjusting config.vocab_size {config.vocab_size} -> {tokenizer.vocab_size} to match tokenizer")
        config.vocab_size = tokenizer.vocab_size

    # Load data (batasi agar tokenisasi cepat)
    print("\n📊 Loading data...")
    max_train = min(config.max_steps * 50, 500000)
    train_loader, val_loader = create_dataloaders(
        config, tokenizer=tokenizer,
        max_train_samples=max_train,
        max_val_samples=2000,
    )

    if is_xla:
        train_loader = xla_pl.MpDeviceLoader(train_loader, device)
        val_loader = xla_pl.MpDeviceLoader(val_loader, device)

    # Create model
    print("\n🏗️ Building model...")
    model = aksaraLLMModel(config).to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Optimizer (decoupled weight decay — see build_optimizer docstring)
    optimizer = build_optimizer(model, config)

    # Loss-scaler only needed for fp16; bf16/fp32 don't need it.
    scaler = GradScaler("cuda") if use_scaler else None

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

    def _checkpoint(path, val):
        save_checkpoint(model, optimizer, scaler, step + 1, config, val, path, device=device)
        if on_checkpoint:
            on_checkpoint(path, step + 1)

    while step < config.max_steps:
        # Get batch (infinite dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"] if is_xla else batch["input_ids"].to(device)
        targets = batch["targets"] if is_xla else batch["targets"].to(device)

        # Forward pass (mixed precision on CUDA — bf16 preferred, fp16 falls back to loss-scaling)
        if use_autocast:
            with autocast("cuda", dtype=autocast_dtype):
                _, loss = model(input_ids, targets)
                loss = loss / config.gradient_accumulation_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            _, loss = model(input_ids, targets)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            if is_xla:
                xm.reduce_gradients(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                lr = get_lr(step, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
            elif use_scaler:
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

        if is_xla:
            xm.mark_step()

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
            prompt = "Indonesia adalah"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
            if not is_xla:
                prompt_ids = prompt_ids.to(device)
            generated = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8)
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"  📝 Sample: {generated_text[:200]}...")
            print()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _checkpoint(os.path.join(config.output_dir, "best_model.pt"), val_loss)
                print(f"  🏆 Model terbaik baru! val_loss={val_loss:.4f}")

            # Selalu simpan latest (untuk resume)
            _checkpoint(latest_path, val_loss)
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
            _checkpoint(os.path.join(config.output_dir, f"checkpoint_step{step + 1}.pt"), best_val_loss)
            _checkpoint(latest_path, best_val_loss)
            print(f"  💾 Checkpoint + auto-save @ step {step + 1}")

        step += 1

    # Save final model
    final_path = os.path.join(config.output_dir, "final_model.pt")
    _checkpoint(final_path, best_val_loss)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Total time: {total_time / 60:.1f} minutes")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved in: {config.output_dir}/")
    print("=" * 60)
    print(f"\n🚀 Try your model: python demo.py --checkpoint {config.output_dir}/best_model.pt")
    print(f"🔄 Resume training: python train.py --size mini --resume {latest_path}")

    return final_path
