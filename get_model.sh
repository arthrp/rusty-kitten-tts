#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/kitten_tts_micro_v0_8.onnx"
URL="https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/kitten_tts_micro_v0_8.onnx"

# Optional: HF_TOKEN for private/gated repos (not needed for this public model).
CURL_AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

exec curl -fL "${CURL_AUTH[@]}" -C - -o "${OUT}" "${URL}"
