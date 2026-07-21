# gemma4-runner

Google **Gemma 4 E2B** 로컬 실행 CLI. 파일 하나 ([gemma4.py](gemma4.py)) 가 전부다.

텍스트 · 이미지 · 오디오 · 비디오 입력과 reasoning 모드를 지원한다.

## 요구사항

- Windows + Python 3.14
- RAM 16GB 이상 (로딩에 ~10GB 상주)
- 여유 디스크 10GB — 가중치는 `D:\hf-cache` 에 받는다
- GPU 불필요. CPU로 동작한다

## 설치

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`torch`/`torchvision` 을 CPU 인덱스에서 받는 이유는 NVIDIA GPU가 없어서다.
기본 PyPI 휠은 CUDA 런타임이 묶여 있어 2.5GB를 헛되이 먹는다. CPU 휠은 124MB.

첫 실행 시 가중치 9.6GB를 자동으로 내려받는다.

## 사용

### 대화 모드 (권장)

프롬프트 없이 실행하면 모델을 **한 번만 로드**하고 계속 질문을 받는다.
매번 15~20초씩 기다릴 필요가 없고 대화 기록도 이어진다.

```powershell
.\.venv\Scripts\python.exe gemma4.py
```

```
> 내 이름은 해승이야
안녕하세요, 해승님! 만나서 반갑습니다.

> 내 이름이 뭐라고 했지?
해승
```

대화 중 명령:

| 명령 | 설명 |
|---|---|
| `/think` | reasoning 모드 토글 |
| `/image PATH` | 다음 질문에 이미지 첨부 (`/audio`, `/video` 도 동일) |
| `/reset` | 대화 기록 비우기 |
| `/exit` | 종료 (Ctrl+C 도 동일) |
| `/help` | 명령 목록 |

답변은 생성되는 대로 스트리밍된다 (`--think` 모드 제외).

### 일회성 실행

```powershell
.\.venv\Scripts\python.exe gemma4.py "안녕, 자기소개 해봐"
```

| 옵션 | 설명 |
|---|---|
| `--think` | reasoning 모드. 출력에 `thinking` 키가 추가된다 |
| `--image PATH\|URL` | 이미지 입력 |
| `--audio PATH\|URL` | 오디오 입력 (최대 30초) |
| `--video PATH\|URL` | 비디오 입력 (최대 60초) |
| `--system TEXT` | 시스템 프롬프트 |
| `--max-new-tokens N` | 생성 길이 (기본 1024) |

체크포인트 교체는 환경변수로:

```powershell
$env:MODEL_ID="google/gemma-4-12B-it"; .\.venv\Scripts\python.exe gemma4.py "안녕"
```

## 알아둘 것

- 가중치 로드에 **15~20초** 걸린다. 여러 번 물을 거면 대화 모드를 써라.
- 일회성 실행의 출력은 dict 형태다: `{'role': 'assistant', 'content': '...'}`
- 대화가 길어질수록 매 턴 전체 기록을 다시 인코딩해서 느려진다. `/reset` 으로 끊으면 된다.
- 무출력 종료(exit 5)는 코드 버그가 아니라 로딩 중 메모리 부족이다. 몇 GB 확보하면 된다.
- `HF_TOKEN` 미설정 시 다운로드에 rate limit 이 걸린다.

## E2B 가 뭔가

"E" 는 *effective* — 유효 파라미터 2.3B, **총 5.1B**. 차이는 Per-Layer Embeddings로,
35개 레이어 각각이 토큰별 작은 임베딩 룩업 테이블을 갖는 구조다.
연산량은 2.3B급이지만 메모리에는 **5.1B 전부**가 올라간다.

상위 모델인 `gemma-4-26B-A4B-it` 는 MoE로 활성 3.8B / 총 25.2B, bf16 기준 ~51GB가
필요해 일반 데스크톱에서는 돌지 않는다. 같은 이유다 — MoE도 PLE도 연산을 줄이지
메모리를 줄이지 않는다.

## 파인튜닝(LoRA)은 왜 안 되나

이 머신에서는 못 한다. 클라우드 GPU를 빌려야 한다.

LoRA는 베이스 가중치를 얼리므로 흔히 지목되는 옵티마이저 상태는 문제가 아니다
(어댑터 몫이라 수십 MB급). 막히는 건 두 가지다.

**메모리** — 추론만으로 이미 ~10GB가 상주하는데 여유 RAM은 ~11GB다. 학습은 여기에
역전파용 activation이 35개 레이어 전부에 얹힌다. gradient checkpointing으로
짜내면 들어갈 수는 있다. 문제는 그게 시간을 더 쓰는 방식이다.

**속도** — 이쪽이 진짜 벽이다. GPU가 없어 forward부터 CPU로 도는데, backward는
대략 그 2~3배가 든다. 스텝당 수십 초 규모이고 체크포인팅을 켜면 forward를 한 번
더 돌리니 더 늘어난다. 수천 스텝짜리 파인튜닝은 며칠~주 단위가 된다.
메모리는 트릭으로 줄일 수 있어도 CPU 연산량은 줄일 방법이 없다.

그래서 순서는 이렇다:

1. **시스템 프롬프트 / few-shot** — 대부분의 "말투를 바꾸고 싶다"는 여기서 끝난다
2. **RAG** — 지식을 넣고 싶은 경우. 학습이 아니라 검색 문제다
3. **LoRA** — 위 둘로 안 될 때만. 런팟·Colab 등에서 `peft` 로 어댑터만 학습해
   내려받아 로컬 추론에 얹는 식. 추론은 이 머신에서 계속 된다