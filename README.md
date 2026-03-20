# kittentts-rs

Rust CLI for **KittenTTS**: same pipeline as the Python [package](https://github.com/KittenML/KittenTTS) — text preprocessing → **eSpeak NG** (IPA) → ONNX inference → WAV.

## System dependency: eSpeak NG

Phonemization uses the **native** C library `libespeak-ng` (not the pure-Rust port).

- **macOS:** `brew install espeak-ng`  
  If linking still fails, set `export HOMEBREW_PREFIX=/opt/homebrew` (Apple Silicon) or `/usr/local` (Intel).
- **Linux:** install `libespeak-ng-dev` / `espeak-ng-devel` for your distro (package name varies).

## Usage

From the `kittentts-rs` directory (defaults assume model and voices in the repo root):

```bash
cargo run --release -- \
  --text "Hello from KittenTTS." \
  --voice Bruno \
  --output /tmp/out.wav
```

- `--model` default: `../kitten_tts_micro_v0_8.onnx`
- `--voices` default: `../voices.npz`

## ONNX Runtime

The `ort` crate may download a matching ONNX Runtime binary at **build** time unless you configure `load-dynamic` and a system install yourself.
