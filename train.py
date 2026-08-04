"""
aksaraLLM — Pretraining CLI (from scratch)

Fitur Utama:
  - Resume training dari checkpoint (anti-mati Colab)
  - Continue pre-training dengan dataset baru
  - Mixed precision (bf16 by default when supported, fp16 fallback) untuk kecepatan 2x di GPU
  - GPU keepalive terintegrasi

Usage:
    # Training dari nol (Indonesian Wikipedia + AksaraTokenizer)
    python train.py --size mini --batch-size 16 --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b

    # Resume training yang terputus
    python train.py --size mini --batch-size 16 --resume checkpoints/aksarallm-mini/latest.pt

    # Continue pre-training dengan dataset baru
    python train.py --size mini --batch-size 16 --resume best_model.pt --dataset wikimedia/wikipedia --dataset-config 20231101.id --max-steps 20000

For distributed/TPU training, see the aksara-train repo.
"""
import argparse

from aksarallm.config import CONFIGS
from aksarallm.trainer import pretrain


def main():
    parser = argparse.ArgumentParser(description="Pretrain aksaraLLM from scratch")
    parser.add_argument(
        "--size", type=str, default="nano",
        choices=list(CONFIGS.keys()),
        help="Model size preset (default: nano)"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--eval-interval", type=int, default=None,
                         help="Override eval/checkpoint interval — lower this for quick smoke tests "
                              "(best_model.pt/latest.pt are only written on eval steps)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dataset", type=str, default=None,
                         help="HuggingFace dataset name, OR a local .jsonl file/directory "
                              "(records need a \"text\" field)")
    parser.add_argument("--dataset-config", type=str, default=None, help="HuggingFace dataset config/subset name (e.g. 20231101.id)")
    parser.add_argument(
        "--tokenizer-path", type=str, default=None,
        help="Path to a directory saved by AksaraTokenizer.save_pretrained() (see aksara-tokenizer). "
             "Without this, training falls back to GPT-2's English tokenizer for smoke-testing only."
    )
    parser.add_argument("--resume", type=str, default=None, help="Path ke checkpoint untuk melanjutkan training")
    parser.add_argument(
        "--precision", type=str, default=None, choices=["auto", "bf16", "fp16", "fp32"],
        help="Compute precision (default: auto — bf16 if the GPU supports it, else fp16, else fp32)"
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Trade ~30%% compute for much lower activation memory — use this to fit bigger batches/models"
    )

    args = parser.parse_args()

    config = CONFIGS[args.size]

    if args.max_steps:
        config.max_steps = args.max_steps
    if args.eval_interval:
        config.eval_interval = args.eval_interval
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        config.output_dir = f"checkpoints/aksarallm-{args.size}"
    if args.dataset:
        config.dataset_name = args.dataset
    if args.dataset_config is not None:
        config.dataset_config = args.dataset_config
    if args.tokenizer_path:
        config.tokenizer_path = args.tokenizer_path
    if args.precision:
        config.precision = args.precision
    if args.gradient_checkpointing:
        config.gradient_checkpointing = True

    pretrain(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
