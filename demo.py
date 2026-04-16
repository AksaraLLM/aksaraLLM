"""
aksaraLLM — Interactive Demo

Chat with your trained aksaraLLM model!

Usage:
    python demo.py --checkpoint checkpoints/aksarallm-nano/best_model.pt
    python demo.py --checkpoint checkpoints/aksarallm-nano/best_model.pt --mode chat
    python demo.py --checkpoint checkpoints/aksarallm-nano/best_model.pt --mode complete
"""
import argparse
import sys

import torch
from transformers import AutoTokenizer

from aksarallm.config import aksaraLLMConfig
from aksarallm.model import aksaraLLMModel


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct config
    cfg = checkpoint["config"]
    config = aksaraLLMConfig(
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_embd=cfg["n_embd"],
        n_inner=cfg["n_inner"],
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        dropout=0.0,  # No dropout during inference
        bias=cfg["bias"],
    )
    
    model = aksaraLLMModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    step = checkpoint.get("step", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"✅ Model loaded (step {step}, val_loss: {val_loss})")
    
    return model, config


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def interactive_mode(model, tokenizer, device, config, mode="complete"):
    """Run interactive text generation."""
    
    print("\n" + "=" * 60)
    print("🧠 aksaraLLM Interactive Demo")
    print("=" * 60)
    print(f"Model: {config.n_layers}L / {config.n_heads}H / {config.n_embd}D")
    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print("-" * 60)
    
    if mode == "complete":
        print("📝 Text Completion Mode")
        print("   Type a prompt and the model will complete it.")
        print("   Type 'quit' or 'exit' to stop.")
        print("-" * 60)
        
        while True:
            try:
                prompt = input("\n✏️  Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Bye!")
                break
            
            if prompt.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            
            if not prompt:
                continue
            
            # Generate
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            print("\n🤖 aksaraLLM:", end=" ", flush=True)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                )
            
            output = tokenizer.decode(generated[0], skip_special_tokens=True)
            # Only print the generated part (after prompt)
            print(output)
    
    elif mode == "chat":
        print("💬 Chat Mode (experimental)")
        print("   This is a base model, not instruction-tuned.")
        print("   It works best as a story/text completer.")
        print("   Type 'quit' or 'exit' to stop.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Bye!")
                break
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            
            if not user_input:
                continue
            
            # For a base model, we format as a story continuation
            prompt = f"{user_input}"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            print("\n🤖 aksaraLLM:", end=" ", flush=True)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                )
            
            output = tokenizer.decode(generated[0], skip_special_tokens=True)
            response = output[len(prompt):].strip()
            print(response)


def main():
    parser = argparse.ArgumentParser(description="aksaraLLM Demo")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--mode", type=str, default="complete",
        choices=["complete", "chat"],
        help="Demo mode (default: complete)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt (non-interactive mode)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    device = get_device()
    model, config = load_model(args.checkpoint, device)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.prompt:
        # Non-interactive: single generation
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(output)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, device, config, args.mode)


if __name__ == "__main__":
    main()
