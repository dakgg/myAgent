# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file CLI runner for Google's **Gemma 4 E2B** multimodal model
([gemma4.py](gemma4.py)), following the model card's "Getting Started" section.
Not a library, not a package — one script, run directly.

## Commands

```powershell
# 대화 모드 — 프롬프트 인자를 생략하면 진입. 모델을 1회만 로드한다.
.\.venv\Scripts\python.exe gemma4.py

# 일회성 (venv python directly; no activation needed)
.\.venv\Scripts\python.exe gemma4.py "프롬프트"
.\.venv\Scripts\python.exe gemma4.py --think "프롬프트"          # reasoning 모드
.\.venv\Scripts\python.exe gemma4.py --image a.png "프롬프트"    # --audio / --video 도 동일
.\.venv\Scripts\python.exe gemma4.py --max-new-tokens 2048 --system "..." "프롬프트"

# Checkpoint 교체
$env:MODEL_ID="google/gemma-4-12B-it"; .\.venv\Scripts\python.exe gemma4.py "프롬프트"

# 의존성 재설치 (torch/torchvision 은 반드시 CPU 인덱스에서)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

There is no test suite and no build step. Verification is done by running the
script and inspecting output.

## Environment constraints (the non-obvious part)

This machine drives most of the design decisions in the script:

| | |
|---|---|
| GPU | AMD RX 6700 XT — **no CUDA**, Windows PyTorch has no ROCm build. Everything runs on CPU. |
| RAM | 31 GB total, often only ~11 GB free. Loading E2B needs ~10 GB — it is genuinely tight. |
| C: | Small free space. **HF cache is redirected to `D:\hf-cache`** via `HF_HOME`, set at the top of the script before importing transformers. |

Consequences already encoded in the script, do not "fix" them back:

- **`device_map="auto"` is conditional on `torch.cuda.is_available()`.** The model
  card uses it unconditionally, but on a CPU-only box accelerate reacts to low RAM
  by offloading weights to *disk*, which is much slower. Plain load is correct here.
- **No explicit `temperature`/`top_p`/`top_k`** in `generate()`. The recommended
  values (1.0 / 0.95 / 64) already ship in the checkpoint's `generation_config.json`.
- **`torchvision` is a hard requirement** even for text-only use —
  `Gemma4ImageProcessor` imports it at module load. The model card's
  `pip install` line omits it; that is a doc bug, not something to trim.

If a run dies silently with **exit code 5 and no traceback**, that is memory
pressure during weight load, not a code bug. Freeing a few GB (e.g. `wsl --shutdown`)
resolves it.

## Model facts that affect the code

- `google/gemma-4-E2B-it`: 2.3B *effective* / **5.1B total** params. The gap is
  Per-Layer Embeddings — a 262144 × 256 lookup table per each of 35 layers. All
  5.1B must be resident; "E2B" describes compute, not memory.
- 128K context, text + image + audio + video. (The larger `26B-A4B` variant has
  `audio_config: null` — no audio — and needs ~51 GB in bf16, so it **cannot run
  on this machine**.)
- Multimodal content ordering matters: **images/video before the text block,
  audio after it**. This is why `content` is assembled in that specific order.
- `processor.parse_response()` returns a dict — `{'role', 'content'}`, plus a
  separate `'thinking'` key when `enable_thinking=True`.

## Chat mode

Omitting the positional prompt enters a REPL that keeps one loaded model and a
growing `messages` list. Slash commands (`/think`, `/image`, `/reset`, `/exit`)
are handled before the line ever reaches the model. Streaming is on except in
thinking mode, where the full response must be decoded before `parse_response()`
can split reasoning from answer.

Two encoding guards exist because piped input on Windows breaks otherwise —
`sys.stdin.reconfigure(encoding="utf-8")` when stdin is not a tty (cp949 would
mangle Korean), and a BOM strip on each input line (a leading `﻿` makes the
first `/command` miss its `startswith("/")` check and get sent to the model).
Both are load-bearing for `Get-Content file | python gemma4.py`.

## Gotchas

- Each *process start* reloads ~10 GB from disk; expect **15–20 s**. Chat mode
  pays this once. Benchmark timings are dominated by it, not by generation.
- `HF_TOKEN` is not set — downloads work but are rate-limited.
- Prefer writing throwaway Python to a scratchpad file over `python -c` with a
  here-string; PowerShell mangles the embedded quotes.
