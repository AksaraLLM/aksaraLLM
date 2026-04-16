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
echo "  python aksara_cli.py              # Default (Qwen API)"
echo "  python aksara_cli.py --model local # Local 26M model"  
echo "  python aksara_cli.py --model none  # Tools only"
echo ""
echo "🚀 Jalankan: python aksara_cli.py"
