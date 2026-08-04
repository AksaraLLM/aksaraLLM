"""
aksaraLLM — Test-time reasoning ("thinking") generation

Two-phase generation using the [THINK]/[/THINK] special tokens (see
aksarallm.tokenizer_utils.SPECIAL_TOKENS): first generate a reasoning trace
— stopping at [/THINK] or a token budget — then generate the final answer.
This is the same idea behind o1/R1/extended-thinking-style models: spend
more inference-time compute deliberating in a scratch space before
committing to an answer, instead of emitting the first plausible
continuation.

This module only implements the *inference-side* mechanics. A freshly
initialized/pretrained model has no reason to use [THINK] meaningfully —
it needs SFT data with explicit reasoning traces in that format first (see
aksara-data/generators/multiturn_cot.py, which emits both a plain
chain-of-thought answer and a [THINK]-tagged variant for this).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import aksaraLLMModel, KVCache


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Single-step top-k/top-p/temperature sampling — the same logic
    aksaraLLMModel.generate() uses, factored out here so this module can
    drive generation token-by-token (across the think/answer boundary)
    without duplicating it."""
    logits = logits.clone() / max(temperature, 1e-5)

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_with_thinking(
    model: aksaraLLMModel,
    tokenizer,
    prompt: str,
    max_think_tokens: int = 512,
    max_answer_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> dict:
    """Generate a {"reasoning": ..., "answer": ...} pair for `prompt`.

    `tokenizer` is an AksaraTokenizer (needs [THINK]/[/THINK]/[EOS] in its
    vocab — retrain via aksara-tokenizer if it doesn't). Both phases share
    one KV-cache, so the prompt and reasoning trace are each only encoded
    once regardless of how long they are.
    """
    model.eval()
    device = device or next(model.parameters()).device

    think_id = tokenizer.tokenizer.token_to_id("[THINK]")
    think_end_id = tokenizer.tokenizer.token_to_id("[/THINK]")
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    if think_id is None or think_end_id is None:
        raise ValueError(
            "Tokenizer is missing [THINK]/[/THINK] — retrain it with the current "
            "aksarallm.tokenizer_utils.SPECIAL_TOKENS."
        )

    prompt_ids = tokenizer.encode(prompt) + [think_id]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    kv_cache = KVCache(model.config.n_layers)
    logits, _ = model(input_ids, kv_cache=kv_cache, start_pos=0)
    cur_pos = input_ids.shape[1]
    next_logits = logits[:, -1, :]

    # ── Phase 1: reasoning trace, until [/THINK] or the token budget ──
    think_ids: list[int] = []
    hit_think_end = False
    for _ in range(max_think_tokens):
        next_token = _sample_next(next_logits, temperature, top_k, top_p)
        tok_id = next_token.item()
        hit_think_end = tok_id == think_end_id
        if not hit_think_end:
            think_ids.append(tok_id)

        if cur_pos >= model.config.max_seq_len:
            break
        # Feed this token through even when it's [/THINK] — the model must
        # actually see the closing tag in its own context before we start
        # sampling the answer, otherwise the phase boundary is invisible to
        # it and "thinking" bleeds into "answering" with no signal.
        logits, _ = model(next_token, kv_cache=kv_cache, start_pos=cur_pos)
        cur_pos += 1
        next_logits = logits[:, -1, :]
        if hit_think_end:
            break

    if not hit_think_end and cur_pos < model.config.max_seq_len:
        # Model rambled past the think budget without closing the tag —
        # force it closed so the answer phase still gets a clean signal.
        think_end_tensor = torch.tensor([[think_end_id]], dtype=torch.long, device=device)
        logits, _ = model(think_end_tensor, kv_cache=kv_cache, start_pos=cur_pos)
        cur_pos += 1
        next_logits = logits[:, -1, :]

    # ── Phase 2: final answer, until [EOS] or the token budget ──
    answer_ids: list[int] = []
    for _ in range(max_answer_tokens):
        next_token = _sample_next(next_logits, temperature, top_k, top_p)
        tok_id = next_token.item()
        if tok_id == eos_id:
            break
        answer_ids.append(tok_id)

        if cur_pos >= model.config.max_seq_len:
            break
        logits, _ = model(next_token, kv_cache=kv_cache, start_pos=cur_pos)
        cur_pos += 1
        next_logits = logits[:, -1, :]

    return {
        "reasoning": tokenizer.decode(think_ids),
        "answer": tokenizer.decode(answer_ids),
    }
