# Session Log

## Turn 1
User asked: is it trivial to build an Android text editor app with TTS, where
TTS = primary TTS model + small LLM for corrections. Wants validation in this
environment (Python) first using sample audio/text, then an architecture doc
with testing/direction guidance.

Actions:
- Inspected repo: empty, branch `claude/android-text-editor-tts-1jqfu6`, no commits.
- Checked installed stack: Python 3.13, numpy only. No torch/transformers/TTS/espeak.
- Installed `espeak-ng` via apt (offline, works, no network needed at runtime).
- Installed `piper-tts` via pip (works), but voice model download failed:
  proxy status shows huggingface.co and github.com are policy-denied
  (403 on CONNECT) in this sandbox's network policy. So no real neural TTS
  voice models or LLM weights can be fetched here.
- Decision: validate the *pipeline shape* (text -> correction stage -> TTS
  stage -> audio) using espeak-ng as a stand-in primary TTS engine and a
  deterministic text-normalization module as a stand-in for the small
  correction LLM, since real model weights are unreachable from this
  container. Document the real production stack (Piper neural voices +
  on-device small LLM via llama.cpp/MediaPipe) separately for Android,
  to be validated on-device or in an environment with model access.
