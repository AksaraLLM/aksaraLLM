#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  📤 Upload Model Card ke HuggingFace
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cara pakai: bash ~/aksarallm_upload_card.sh

set -e

MODEL_REPO="AksaraLLM/aksarallm-1.5b-v2"
HF_TOKEN="$HF_TOKEN"

echo "📤 Uploading model card to $MODEL_REPO..."

python3 -c "
from huggingface_hub import HfApi
api = HfApi()

# Upload model card as README.md
api.upload_file(
    path_or_fileobj='$HOME/aksarallm_model_card.md',
    path_in_repo='README.md',
    repo_id='$MODEL_REPO',
    token='$HF_TOKEN'
)
print('✅ Model card uploaded to $MODEL_REPO')
"

echo "🎉 Done! Check: https://huggingface.co/$MODEL_REPO"
