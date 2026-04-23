"""
aksaraLLM - Inference pipeline

Provides:

- :class:`AksaraChatSession` — a multi-turn chat session with KV-cache-backed
  generation, wrapping :class:`aksaraLLMModel` and :class:`AksaraTokenizer`.
- A minimal CLI (``python -m aksarallm.inference``) for interactive chat.
- A Gradio chat UI (``python -m aksarallm.inference --gradio``).

The CLI supports ``--dry-run`` for CPU-only smoke testing without needing any
trained weights. This is what the Phase 5 smoke test in REPORT.md exercises.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import torch

from .config import DEFAULT_SYSTEM_PROMPT, get_config
from .model import aksaraLLMModel
from .tokenizer_utils import AksaraTokenizer, render_chat


@dataclass
class AksaraChatSession:
    """Stateful chat session over a single :class:`aksaraLLMModel`.

    ``history`` accumulates ``{"role": ..., "content": ...}`` messages.
    :meth:`reply` renders them via the canonical chat template, feeds them
    to the model's KV-cache-aware ``generate``, and appends the assistant
    response back onto ``history``.
    """

    model: aksaraLLMModel
    tokenizer: AksaraTokenizer
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history: list[dict] = field(default_factory=list)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def reset(self) -> None:
        self.history = []

    def _ensure_system(self) -> None:
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, {"role": "system", "content": self.system_prompt})

    def reply(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Generate an assistant response for ``user_message``.

        Side-effect: appends both the user and assistant turns to
        :attr:`history`.
        """
        self._ensure_system()
        self.history.append({"role": "user", "content": user_message})

        prompt = render_chat(self.history, add_generation_prompt=True)
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device
        )

        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        new_ids = out[0, input_ids.shape[1]:].tolist()
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        self.history.append({"role": "assistant", "content": response})
        return response


# ──────────────────────────────────────────────────────────────────────
#  Dry-run helpers (used by the smoke tests)
# ──────────────────────────────────────────────────────────────────────
def _build_dry_run_session() -> AksaraChatSession:
    """Construct a random-weights ``tiny`` session + byte-BPE tokenizer.

    No HF download, no TPU, no GPU — runs in well under a second on CPU.
    """
    cfg = get_config("tiny")
    model = aksaraLLMModel(cfg)
    model.eval()

    # Train a 200-vocab BPE on a handful of Indonesian sentences so the
    # tokenizer round-trip is meaningful.
    corpus = [
        "halo, saya aksarallm.",
        "indonesia adalah negara kepulauan.",
        "pancasila adalah dasar negara indonesia.",
        "apa kabar hari ini?",
        "selamat pagi dari jakarta.",
    ] * 20
    tok = AksaraTokenizer.train_bpe_from_iterator(
        iter(corpus), vocab_size=cfg.vocab_size, min_frequency=1
    )
    return AksaraChatSession(model=model, tokenizer=tok)


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def _cli_loop(session: AksaraChatSession, *, max_new_tokens: int, temperature: float) -> None:
    print("[aksarallm] chat session started. Type 'quit' to exit, 'reset' to clear history.")
    while True:
        try:
            user = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        user = user.strip()
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            return
        if user.lower() == "reset":
            session.reset()
            print("[aksarallm] history cleared.")
            continue

        reply = session.reply(
            user, max_new_tokens=max_new_tokens, temperature=temperature
        )
        print(f"aksarallm> {reply}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aksarallm.inference")
    ap.add_argument("--model", default=None,
                    help="Path to a directory saved by aksaraLLMModel.save_pretrained")
    ap.add_argument("--tokenizer", default=None,
                    help="Path to a directory saved by AksaraTokenizer.save_pretrained")
    ap.add_argument("--size", default="20b",
                    help="Preset name; only used when --model is not given.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gradio", action="store_true", help="Launch a Gradio UI.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Initialize a random 'tiny' model and tokenizer on CPU, "
                         "answer a single hard-coded prompt, and exit 0. No HF needed.")
    args = ap.parse_args(argv)

    # ── Dry-run: exercise the full code path on CPU ─────────────────
    if args.dry_run:
        session = _build_dry_run_session()
        reply = session.reply("Siapa kamu?", max_new_tokens=8, temperature=0.0)
        print(f"[dry-run] prompt='Siapa kamu?' reply_len={len(reply)}")
        print("[dry-run] OK")
        return 0

    # ── Real model load ──────────────────────────────────────────────
    if args.model is None or args.tokenizer is None:
        print(
            "error: --model and --tokenizer are required unless --dry-run is used.",
            file=sys.stderr,
        )
        return 2

    model = aksaraLLMModel.from_pretrained(args.model)
    tok = AksaraTokenizer.from_pretrained(args.tokenizer)
    session = AksaraChatSession(model=model, tokenizer=tok)

    if args.gradio:
        _launch_gradio(session, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    else:
        _cli_loop(session, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    return 0


def _launch_gradio(session: AksaraChatSession, *, max_new_tokens: int, temperature: float) -> None:
    try:
        import gradio as gr
    except ImportError:
        print("gradio not installed. `pip install gradio` first.", file=sys.stderr)
        sys.exit(1)

    def respond(message: str, history: list[tuple[str, str]]) -> str:
        del history  # We use our own session history.
        return session.reply(
            message, max_new_tokens=max_new_tokens, temperature=temperature
        )

    gr.ChatInterface(
        fn=respond,
        title="AksaraLLM Chat",
        description="Asisten AI Bahasa Indonesia (aksarallm-20b)",
    ).launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    sys.exit(main())
