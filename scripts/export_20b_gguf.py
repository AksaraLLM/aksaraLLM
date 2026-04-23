#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — GGUF EXPORTER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Exports a trained :class:`aksarallm.model.aksaraLLMModel` checkpoint to
GGUF F16 so it can be consumed by llama.cpp / Ollama / LM Studio.

We write the file directly with :mod:`gguf.GGUFWriter` instead of relying
on ``convert_hf_to_gguf.py`` because the AksaraLLM architecture is
**custom** (not one of the HF model families llama.cpp auto-detects).

After this script produces the F16 file, pipe it through
``llama-quantize`` to get Q4_K_M / Q8_0 / Q2_K variants. See the companion
shell wrapper :file:`scripts/convert_20b_gguf.sh` for the full workflow.

Run:
    python3 scripts/export_20b_gguf.py \\
        --model-dir ./aksarallm-20b-sft \\
        --tokenizer-dir ./aksara-tokenizer-20b \\
        --out ./aksarallm-20b-F16.gguf

Dry-run (tiny config, writes a tiny GGUF to /tmp, ~2s CPU):
    python3 scripts/export_20b_gguf.py --dry-run --out /tmp/tiny.gguf
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
#  Tensor name mapping: AksaraLLM → GGUF (llama.cpp convention)
# ══════════════════════════════════════════════════════════════════
def _map_name(aksara_name: str) -> str | None:
    """Map a PyTorch parameter name to a GGUF tensor name.

    Returns ``None`` if the parameter should be skipped (e.g. tied
    ``lm_head.weight`` which GGUF reconstructs from ``token_embd``).
    """
    n = aksara_name

    if n == "token_emb.weight":
        return "token_embd.weight"
    if n == "norm.weight":
        return "output_norm.weight"
    if n == "lm_head.weight":
        # Tied to token_embd; llama.cpp will reuse token_embd when absent.
        return None

    if n.startswith("layers."):
        _, idx, *rest = n.split(".", 2)
        rest_s = rest[0]
        prefix = f"blk.{idx}"
        if rest_s == "attn_norm.weight":
            return f"{prefix}.attn_norm.weight"
        if rest_s == "mlp_norm.weight":
            return f"{prefix}.ffn_norm.weight"
        if rest_s == "attn.q_proj.weight":
            return f"{prefix}.attn_q.weight"
        if rest_s == "attn.k_proj.weight":
            return f"{prefix}.attn_k.weight"
        if rest_s == "attn.v_proj.weight":
            return f"{prefix}.attn_v.weight"
        if rest_s == "attn.out_proj.weight":
            return f"{prefix}.attn_output.weight"
        if rest_s == "mlp.gate_proj.weight":
            return f"{prefix}.ffn_gate.weight"
        if rest_s == "mlp.up_proj.weight":
            return f"{prefix}.ffn_up.weight"
        if rest_s == "mlp.down_proj.weight":
            return f"{prefix}.ffn_down.weight"
    return None


# ══════════════════════════════════════════════════════════════════
#  Writer
# ══════════════════════════════════════════════════════════════════
def export(model, tokenizer, out_path: str) -> None:
    import gguf
    import numpy as np
    import torch

    cfg = model.config
    arch = "llama"  # AksaraLLM is llama-3 style, so this is the right profile.
    writer = gguf.GGUFWriter(out_path, arch)

    writer.add_name(f"AksaraLLM-{cfg.n_embd}d-{cfg.n_layers}L")
    writer.add_description("AksaraLLM — Indonesian from-scratch LLM.")
    writer.add_context_length(cfg.max_seq_len)
    writer.add_embedding_length(cfg.n_embd)
    writer.add_block_count(cfg.n_layers)
    writer.add_feed_forward_length(cfg.n_inner)
    writer.add_head_count(cfg.n_heads)
    writer.add_head_count_kv(cfg.n_kv_heads)
    writer.add_rope_freq_base(float(getattr(cfg, "rope_theta", 10000.0)))
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # Tokenizer vocabulary.
    vocab = tokenizer.tokenizer.get_vocab()  # type: ignore[attr-defined]
    inv = [""] * (max(vocab.values()) + 1)
    for token, tid in vocab.items():
        inv[tid] = token
    writer.add_tokenizer_model("llama")
    writer.add_token_list(inv)
    writer.add_token_types([gguf.TokenType.NORMAL] * len(inv))
    writer.add_bos_token_id(tokenizer.bos_token_id)
    writer.add_eos_token_id(tokenizer.eos_token_id)
    writer.add_pad_token_id(tokenizer.pad_token_id)
    writer.add_unk_token_id(tokenizer.unk_token_id)

    # Tensors.
    state = model.state_dict()
    n_written = 0
    for name, tensor in state.items():
        gguf_name = _map_name(name)
        if gguf_name is None:
            continue
        arr = tensor.detach().to(torch.float16).cpu().numpy()
        writer.add_tensor(gguf_name, arr)
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log(f"wrote {n_written} tensors → {out_path} "
        f"({os.path.getsize(out_path) / 1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════════
#  Dry-run
# ══════════════════════════════════════════════════════════════════
def _dry_run(out_path: str) -> int:
    from aksarallm.config import get_config
    from aksarallm.model import aksaraLLMModel
    from aksarallm.tokenizer_utils import AksaraTokenizer

    cfg = get_config("tiny")
    model = aksaraLLMModel(cfg)
    corpus = ["halo saya aksarallm", "indonesia merdeka"] * 30
    tok = AksaraTokenizer.train_bpe_from_iterator(
        iter(corpus), vocab_size=cfg.vocab_size, min_frequency=1
    )
    try:
        export(model, tok, out_path)
    except ImportError as e:
        log(f"gguf library missing: {e}", level="WARN")
        log("[dry-run] SKIPPED (install with: pip install gguf)")
        return 0
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
    log("[dry-run] OK")
    return 0


# ══════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export AksaraLLM checkpoint to GGUF F16")
    ap.add_argument("--model-dir")
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--out", default="./aksarallm-20b-F16.gguf")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        return _dry_run(args.out)
    if not args.model_dir or not args.tokenizer_dir:
        ap.error("--model-dir and --tokenizer-dir required unless --dry-run.")

    from aksarallm.model import aksaraLLMModel
    from aksarallm.tokenizer_utils import AksaraTokenizer

    log(f"loading model {args.model_dir}")
    model = aksaraLLMModel.from_pretrained(args.model_dir, map_location="cpu")
    log(f"loading tokenizer {args.tokenizer_dir}")
    tok = AksaraTokenizer.from_pretrained(args.tokenizer_dir)
    export(model, tok, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
