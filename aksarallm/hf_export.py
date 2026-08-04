"""
aksaraLLM — HuggingFace Transformers export

Converts a trained ``aksaraLLMModel`` checkpoint into a standard
``transformers.LlamaForCausalLM``. aksaraLLM's architecture (RoPE + GQA +
RMSNorm + SwiGLU + tied embeddings, see RFC-001) is functionally the same as
LLaMA's — it's implemented from scratch here rather than importing the
library class, but that means trained checkpoints otherwise can't be loaded
by ``AutoModelForCausalLM``, ``aksara-eval``, ``demo/gradio_chat.py``, GGUF
conversion (llama.cpp), or vLLM. Exporting to the standard Llama layout makes
all of that work with zero custom/``trust_remote_code`` model code.

Caveat this module exists to handle: aksaraLLM applies RoPE to *interleaved*
adjacent pairs (x0,x1), (x2,x3), ... (like Meta's original LLaMA reference
implementation — see ``model.apply_rotary_emb``), while HF's
``LlamaAttention`` rotates *half-split* pairs (x_i, x_{i+head_dim/2}) instead
via ``rotate_half``. These are different rotations of the same weights, so
converting requires permuting the rows of the Q/K projections — the same
permutation HF's own ``convert_llama_weights_to_hf.py`` applies when
importing Meta's original checkpoints.
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from .config import aksaraLLMConfig
from .model import aksaraLLMModel


def to_llama_config(config: aksaraLLMConfig) -> LlamaConfig:
    """Map an aksaraLLMConfig onto an equivalent transformers LlamaConfig."""
    return LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.n_embd,
        intermediate_size=config.n_inner,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads or config.n_heads,
        max_position_embeddings=config.max_seq_len,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        attention_bias=config.bias,
        mlp_bias=config.bias,
        hidden_act="silu",
    )


def _permute_rope_weight(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder Q/K projection rows from aksaraLLM's interleaved RoPE pairing
    to HF Llama's half-split pairing (see module docstring)."""
    dim1, dim2 = w.shape
    return (
        w.view(n_heads, head_dim // 2, 2, dim2)
        .transpose(1, 2)
        .contiguous()
        .reshape(dim1, dim2)
    )


def convert_state_dict(state_dict: dict, config: aksaraLLMConfig) -> dict:
    """Map an aksaraLLMModel state_dict onto transformers' LlamaForCausalLM key layout."""
    head_dim = config.n_embd // config.n_heads
    n_kv_heads = config.n_kv_heads or config.n_heads

    out = {
        "model.embed_tokens.weight": state_dict["token_emb.weight"],
        "lm_head.weight": state_dict["lm_head.weight"],
        "model.norm.weight": state_dict["norm.weight"],
    }

    for i in range(config.n_layers):
        src = f"layers.{i}."
        dst = f"model.layers.{i}."

        out[dst + "input_layernorm.weight"] = state_dict[src + "attn_norm.weight"]
        out[dst + "post_attention_layernorm.weight"] = state_dict[src + "mlp_norm.weight"]

        out[dst + "self_attn.q_proj.weight"] = _permute_rope_weight(
            state_dict[src + "attn.q_proj.weight"], config.n_heads, head_dim
        )
        out[dst + "self_attn.k_proj.weight"] = _permute_rope_weight(
            state_dict[src + "attn.k_proj.weight"], n_kv_heads, head_dim
        )
        out[dst + "self_attn.v_proj.weight"] = state_dict[src + "attn.v_proj.weight"]
        out[dst + "self_attn.o_proj.weight"] = state_dict[src + "attn.out_proj.weight"]

        out[dst + "mlp.gate_proj.weight"] = state_dict[src + "mlp.gate_proj.weight"]
        out[dst + "mlp.up_proj.weight"] = state_dict[src + "mlp.up_proj.weight"]
        out[dst + "mlp.down_proj.weight"] = state_dict[src + "mlp.down_proj.weight"]

    return out


def build_hf_model(model: aksaraLLMModel, config: aksaraLLMConfig) -> LlamaForCausalLM:
    """Convert a trained aksaraLLMModel into an equivalent (weight-identical)
    transformers LlamaForCausalLM, in memory."""
    llama_config = to_llama_config(config)
    hf_model = LlamaForCausalLM(llama_config)

    converted = convert_state_dict(model.state_dict(), config)
    missing, unexpected = hf_model.load_state_dict(converted, strict=False)
    # Under tie_word_embeddings=True, transformers may report one side of the
    # tied embed_tokens/lm_head pair as "missing" even though both are present
    # in `converted` and tying makes them share storage anyway — harmless.
    real_missing = [k for k in missing if "embed_tokens" not in k and "lm_head" not in k]
    if real_missing or unexpected:
        raise RuntimeError(
            f"aksaraLLM -> HF Llama conversion mismatch — "
            f"missing={real_missing}, unexpected={unexpected}"
        )
    return hf_model


def export_to_hf(model: aksaraLLMModel, config: aksaraLLMConfig, out_dir: str, tokenizer=None) -> LlamaForCausalLM:
    """Convert a trained aksaraLLMModel + config and save it in standard HF
    format (config.json + model.safetensors) to `out_dir`, ready for
    AutoModelForCausalLM.from_pretrained(). Also exports `tokenizer` (an
    AksaraTokenizer) as a standard HF tokenizer when provided."""
    hf_model = build_hf_model(model, config)
    hf_model.save_pretrained(out_dir, safe_serialization=True)

    if tokenizer is not None:
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer.tokenizer,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]",
        )
        hf_tokenizer.save_pretrained(out_dir)

    return hf_model
