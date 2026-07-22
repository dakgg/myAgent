r"""Gemma 4 E2B 러너 — 모델 카드 Getting Started 기준.

대화 모드 (프롬프트 없이 실행하면 진입, 모델은 한 번만 로드):
  python gemma4.py

일회성:
  python gemma4.py "9.11 과 9.9 중 뭐가 큰가"
  python gemma4.py --think "어려운 질문"
  python gemma4.py --image cat.png "이 사진 설명해줘"

MODEL_ID 로 체크포인트 교체 가능. 가중치는 D:\hf-cache 에 받는다 (C: 여유 부족).
"""
import argparse, os, sys

os.environ.setdefault("HF_HOME", r"D:\hf-cache")

# 파이프로 넣으면 Windows 기본 인코딩(cp949)이라 한글이 깨진다. 콘솔은 건드리지 않는다.
if not sys.stdin.isatty():
    sys.stdin.reconfigure(encoding="utf-8")

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, TextStreamer

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it")

p = argparse.ArgumentParser()
p.add_argument("prompt", nargs="?", help="생략하면 대화 모드")
p.add_argument("--image", help="이미지 경로 또는 URL")
p.add_argument("--audio", help="오디오 경로 또는 URL (최대 30초)")
p.add_argument("--video", help="비디오 경로 또는 URL (최대 60초)")
p.add_argument("--think", action="store_true", help="reasoning 모드")
p.add_argument("--system", default="너는 친근한 도우미야. 항상 편한 반말로, 친구처럼 짧고 다정하게 대답해.")
p.add_argument("--max-new-tokens", type=int, default=1024)
args = p.parse_args()


def build_content(text, image=None, audio=None, video=None):
    """모델 카드 규칙: 이미지/비디오는 텍스트 앞, 오디오는 텍스트 뒤."""
    content = [{"type": k, "url": v} for k, v in (("image", image), ("video", video)) if v]
    content.append({"type": "text", "text": text})
    if audio:
        content.append({"type": "audio", "url": audio})
    return content


def ask(messages, think, max_new_tokens, stream=False):
    """messages 에 assistant 응답을 덧붙이고 파싱된 dict 를 반환."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=think,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # 권장 샘플링(temp 1.0 / top_p 0.95 / top_k 64)은 generation_config.json 에 이미 있다
    # skip_special_tokens 없으면 <turn|> 같은 토큰이 화면에 샌다
    streamer = TextStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    ) if stream else None
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    parsed = processor.parse_response(response, prefix=inputs["input_ids"])
    messages.append({"role": "assistant", "content": parsed.get("content", "")})
    return parsed


processor = AutoProcessor.from_pretrained(MODEL_ID)
# device_map="auto" 는 GPU 샤딩용. CPU 뿐이면 오히려 디스크 오프로딩을 유발해서 느려진다.
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto" if torch.cuda.is_available() else None
)

messages = [{"role": "system", "content": args.system}]

if args.prompt:
    messages.append({"role": "user", "content": build_content(
        args.prompt, args.image, args.audio, args.video)})
    print(ask(messages, args.think, args.max_new_tokens))
    raise SystemExit

# ── 대화 모드 ────────────────────────────────────────────────
HELP = """명령:
  /think          reasoning 모드 토글
  /image PATH     다음 질문에 이미지 첨부 (/audio, /video 도 동일)
  /reset          대화 기록 비우기
  /exit           종료 (Ctrl+C 도 동일)"""

print(f"{MODEL_ID} 준비 완료. /help 로 명령 목록.\n")
think, attach = args.think, {}

while True:
    try:
        line = input("\n> ").strip().lstrip("﻿")  # 파일로 넣을 때 첫 줄 BOM
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        continue

    if line.startswith("/"):
        cmd, _, rest = line.partition(" ")
        if cmd == "/exit":
            break
        elif cmd == "/help":
            print(HELP)
        elif cmd == "/think":
            think = not think
            print(f"thinking: {'on' if think else 'off'}")
        elif cmd == "/reset":
            del messages[1:]
            attach.clear()
            print("기록을 비웠다.")
        elif cmd in ("/image", "/audio", "/video"):
            attach[cmd[1:]] = rest.strip() or None
            print(f"{cmd[1:]}: {attach[cmd[1:]] or '해제'}")
        else:
            print(f"모르는 명령: {cmd}\n{HELP}")
        continue

    # ponytail: 매 턴 전체 기록을 다시 인코딩한다. KV 캐시 재사용은 대화가
    # 길어져서 체감될 때 붙이면 된다.
    messages.append({"role": "user", "content": build_content(line, **attach)})
    attach.clear()

    parsed = ask(messages, think, args.max_new_tokens, stream=not think)
    if think:
        print(f"\n[thinking]\n{parsed.get('thinking', '')}\n\n[답변]\n{parsed.get('content', '')}")
