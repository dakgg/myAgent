r"""폴더의 .md/.txt 를 검색해 Gemma4 에 근거로 넘기는 RAG (rag.py).

  python rag.py "환불 언제까지 돼?"            # 검색 결과 + 생성 프롬프트만 출력
  python rag.py --ask "환불 언제까지 돼?"       # gemma4 로 답변까지
  python rag.py --docs ./내문서 -k 6 "질문"
  python rag.py --demo                          # 모델 없이 청킹·랭킹 자체검증

임베딩 모델은 D:\hf-cache 에 받는다. EMBED_MODEL 로 교체 가능.
faiss/벡터DB 없이 numpy 브루트포스 — 수천 chunk 까진 충분하다.
"""
import argparse, glob, os, subprocess, sys

os.environ.setdefault("HF_HOME", r"D:\hf-cache")
sys.stdout.reconfigure(encoding="utf-8")  # cp949 콘솔에 한글·em-dash 출력하다 죽는 것 방지
import numpy as np

# MiniLM: 한글 되고 가볍고 prefix 규칙 없음.
# ponytail: 검색 품질 아쉬우면 multilingual-e5-small 로. 단 "query:"/"passage:" prefix 필요.
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def load_chunks(folder):
    """문단(빈 줄) 단위로 쪼갠다. (파일명, 본문) 리스트."""
    chunks = []
    for path in glob.glob(os.path.join(folder, "**", "*.*"), recursive=True):
        if not path.lower().endswith((".md", ".txt")):
            continue
        for para in open(path, encoding="utf-8").read().split("\n\n"):
            para = para.strip()
            if len(para) >= 20:  # 제목·구분선 같은 파편은 버린다
                chunks.append((os.path.basename(path), para))
    return chunks


_model = None
def embed(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(texts, normalize_embeddings=True)  # 정규화 → cosine = 내적


def rank(q_vec, doc_vecs, k):
    """상위 k 개 chunk 인덱스. 모델과 분리돼 있어 테스트 가능."""
    return np.argsort(-(doc_vecs @ q_vec))[:k]


def retrieve(query, chunks, k=4):
    doc_vecs = embed([c[1] for c in chunks])
    q_vec = embed([query])[0]
    idx = rank(q_vec, doc_vecs, k)
    return [(*chunks[i], float(doc_vecs[i] @ q_vec)) for i in idx]


def build_prompt(query, hits):
    ctx = "\n\n".join(f"[{src}] {txt}" for src, txt, _ in hits)
    return (f"아래 자료만 참고해서 질문에 답해. 자료에 없으면 모른다고 해.\n\n"
            f"=== 자료 ===\n{ctx}\n\n=== 질문 ===\n{query}")


def demo():
    """모델 없이 청킹·랭킹이 안 깨지는지만 본다."""
    here = os.path.dirname(os.path.abspath(__file__))
    chunks = load_chunks(os.path.join(here, "docs"))
    assert chunks, "docs/ 에서 chunk 를 못 읽었다"
    # 3번 문서가 질의와 정확히 일치하도록 가짜 벡터를 심고, rank 가 그걸 1등으로 뽑는지
    fake = np.eye(len(chunks), dtype=np.float32)
    q = fake[2]
    assert rank(q, fake, 1)[0] == 2, "rank 가 최상위 chunk 를 못 골랐다"
    print(f"OK — chunk {len(chunks)}개, 랭킹 정상")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--docs", default="docs")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--ask", action="store_true", help="검색만 하지 말고 gemma4 로 답변까지")
    p.add_argument("--demo", action="store_true", help="모델 없이 자체검증")
    args = p.parse_args()

    if args.demo:
        demo()
        raise SystemExit
    if not args.query:
        sys.exit("질문을 줘. 예) python rag.py \"환불 언제까지 돼?\"")

    chunks = load_chunks(args.docs)
    if not chunks:
        sys.exit(f"{args.docs} 에 .md/.txt 문서가 없다.")

    hits = retrieve(args.query, chunks, args.k)
    prompt = build_prompt(args.query, hits)

    if args.ask:
        subprocess.run([sys.executable, "gemma4.py", prompt])
    else:
        for src, txt, score in hits:
            print(f"\n[{score:.3f}] {src}\n{txt}")
        print("\n--- 생성 프롬프트 ---\n" + prompt)
