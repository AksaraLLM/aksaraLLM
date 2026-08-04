#!/bin/bash
# ═══════════════════════════════════════
# AksaraLLM CLI — Quick Setup
# ═══════════════════════════════════════

echo "🇮🇩 Setting up AksaraLLM CLI..."

# Install dependencies
pip install rich huggingface_hub tokenizers 2>/dev/null || pip3 install rich huggingface_hub tokenizers

echo ""
echo "✅ Setup complete!"
echo ""
echo "Usage:"
echo "  python aksara_cli.py                # Default: local AksaraLLM checkpoint (from scratch)"
echo "  python aksara_cli.py --model ollama # Ollama backend (GGUF export of AksaraLLM)"
echo "  python aksara_cli.py --model none   # Tools only"
echo ""
echo "🚀 Jalankan: python aksara_cli.py"
