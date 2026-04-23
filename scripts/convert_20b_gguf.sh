#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AksaraLLM 20B — GGUF BUILD PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. export_20b_gguf.py:  custom aksaraLLMModel → GGUF F16
#  2. llama-quantize:      F16 → Q4_K_M, Q8_0, Q2_K
#  3. Build a 20B Modelfile (uses [SYS]/[INST] template, not ChatML)
#
#  Usage:
#    export HF_TOKEN=...         # only needed if --upload is set
#    bash scripts/convert_20b_gguf.sh \
#        ./aksarallm-20b-sft ./aksara-tokenizer-20b ./gguf_out
#
#  Positional args:
#    $1  MODEL_DIR       path to a trained aksaraLLMModel checkpoint dir
#    $2  TOKENIZER_DIR   path to a matching AksaraTokenizer dir
#    $3  OUT_DIR         destination directory (created if missing)
#
#  Optional env:
#    QUANTS="Q4_K_M Q8_0 Q2_K"   # space-separated list; default = all three
#    UPLOAD_REPO=Ezekiel999/AksaraLLM-20B-GGUF    # skipped if empty
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

MODEL_DIR="${1:?usage: $0 MODEL_DIR TOKENIZER_DIR OUT_DIR}"
TOKENIZER_DIR="${2:?tokenizer dir required}"
OUT_DIR="${3:?output dir required}"
QUANTS="${QUANTS:-Q4_K_M Q8_0 Q2_K}"
UPLOAD_REPO="${UPLOAD_REPO:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$OUT_DIR"
F16="$OUT_DIR/AksaraLLM-20B-F16.gguf"

# ── Step 1: ensure llama.cpp is built ─────────────────────────────
if [[ ! -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
    echo "[+] Building llama.cpp at $LLAMA_CPP_DIR"
    if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
        git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    fi
    ( cd "$LLAMA_CPP_DIR" && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
        && cmake --build build --config Release -j )
fi

# ── Step 2: aksaraLLMModel → GGUF F16 ─────────────────────────────
echo "[+] Exporting checkpoint → GGUF F16"
python3 "$SCRIPT_DIR/export_20b_gguf.py" \
    --model-dir "$MODEL_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --out "$F16"

# ── Step 3: quantize ──────────────────────────────────────────────
for q in $QUANTS; do
    OUT="$OUT_DIR/AksaraLLM-20B-$q.gguf"
    echo "[+] Quantizing → $q"
    "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$F16" "$OUT" "$q"
done

# ── Step 4: write Modelfile (Ollama / LM Studio) ──────────────────
cat > "$OUT_DIR/Modelfile" <<'EOF'
FROM ./AksaraLLM-20B-Q4_K_M.gguf

# AksaraLLM template — [SYS]…[/SYS][INST]…[/INST] (NOT ChatML).
TEMPLATE """[SYS]{{ .System }}[/SYS][INST]{{ .Prompt }}[/INST]"""

SYSTEM "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, sopan, dan membantu. Jawab dengan jelas, jujur, dan ringkas."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.15
PARAMETER stop "[EOS]"
PARAMETER stop "[/INST]"
PARAMETER stop "[INST]"
EOF

echo "[+] Wrote Modelfile → $OUT_DIR/Modelfile"

# ── Step 5: optional HF upload ────────────────────────────────────
if [[ -n "$UPLOAD_REPO" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[!] HF_TOKEN is not set — skipping upload"
        exit 0
    fi
    echo "[+] Uploading to https://huggingface.co/$UPLOAD_REPO"
    python3 - <<PYEOF
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("$UPLOAD_REPO", repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path="$OUT_DIR", repo_id="$UPLOAD_REPO", repo_type="model")
print("  upload OK")
PYEOF
fi

echo ""
echo "[+] Done. Artifacts in $OUT_DIR:"
ls -la "$OUT_DIR"
