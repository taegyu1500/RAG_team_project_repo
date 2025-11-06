#!/usr/bin/env python3
# app_rag_recipes_ollama_history_pdf_fallback.py
# ---------------------------------------------------------
# 🍳 레시피 Q&A RAG (Ollama + FAISS) — History + Robust PDF
# - CSV / PDF / TXT / JSON / URL 인덱싱
# - 슬라이딩 청크 + 완전일치 dedup
# - Ollama 임베딩(/api/embed) + IP(L2정규화=코사인)
# - 검색: k=3~8 + 시간/난이도 필터
# - 답변: 스트리밍 + 대체재 제안
# - ↪️ 추가질문 버튼: 하단 고정 영역에서 안정 동작
# - 🧠 대화 이력: 최근 히스토리를 프롬프트에 주입 (맥락 유지)
# - 📄 PDF: TTF 강제 + sanitize + 하드랩 + fpdf2 실패 시 ReportLab(Platypus CJK) 폴백
# ---------------------------------------------------------

import os
import re
import json
import glob
import unicodedata
import hashlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

import faiss
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from fpdf import FPDF  # 1차 PDF 백엔드

# ==============================
# 환경설정 (ENV로 덮어쓰기 가능)
# ==============================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL     = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL      = os.environ.get("CHAT_MODEL", "llama3.1:8b")

# 디폴트 경로
DEFAULT_CSV_PATH   = "./data/251105.csv"
DEFAULT_PDF_DIR    = "./docs_pdfs"
DEFAULT_TXT_DIR    = "./notes_txt"
DEFAULT_JSON_PATH  = "./notes.json"
DEFAULT_URLS_TXT   = "./urls.txt"
INDEX_DIR          = "./index_out"
FAISS_PATH         = os.path.join(INDEX_DIR, "faiss.index")
META_PATH          = os.path.join(INDEX_DIR, "meta.json")

# ==============================
# CSV 컬럼 매핑 (한글 스키마)
# ==============================
CSV_NAME_COL  = "요리명"
CSV_ING_COL   = "요리재료내용"
CSV_DESC_COL  = None     # 분류 합성
CSV_SERVE_COL = "요리인분별명"
CSV_LEVEL_COL = "요리난이도"
CSV_TIME_COL  = "요리소요시간"
CSV_IMG_COL   = None
CLASS_COLS = ["요리방법별명", "요리상황별명", "요리재료별명", "요리종류별명"]

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="🍳 레시피 RAG (Ollama)", layout="wide")
st.title("🍳 레시피 Q&A RAG (Ollama + FAISS)")

# 세션 상태
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_answer_text" not in st.session_state:
    st.session_state.last_answer_text = ""
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "last_followups" not in st.session_state:
    st.session_state.last_followups = []

# ------------------------------
# 사이드바: 인덱스 옵션
# ------------------------------
with st.sidebar:
    st.subheader("인덱스 옵션")
    csv_path  = st.text_input("CSV 경로", value=DEFAULT_CSV_PATH)
    pdf_dir   = st.text_input("PDF 디렉토리", value=DEFAULT_PDF_DIR)
    txt_dir   = st.text_input("TXT 디렉토리", value=DEFAULT_TXT_DIR)
    json_path = st.text_input("JSON 경로(선택)", value=DEFAULT_JSON_PATH)
    urls_txt  = st.text_input("URL 리스트 파일(선택)", value=DEFAULT_URLS_TXT)

    chunk_size    = st.slider("청크 크기", 300, 2000, 1200, step=50)
    chunk_overlap = st.slider("청크 오버랩", 0, 400, 40, step=10)
    dedup_mode = st.selectbox("중복 제거", ["없음(-1)", "완전일치(0)"], index=0)

    if st.button("📦 인덱스 생성/갱신"):
        with st.spinner("인덱스 생성 중... (임베딩에 시간이 걸릴 수 있어요)"):
            dmode = -1 if "없음" in dedup_mode else 0
            try:
                docs, texts = build_corpus(
                    csv_path, pdf_dir, txt_dir, json_path, urls_txt,
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                    dedup_hamming=dmode,
                )
            except Exception as e:
                st.error(f"코퍼스 생성 실패: {e}")
                st.stop()

            if not texts:
                st.error("인덱싱할 텍스트가 없습니다. 경로/파일을 확인하세요.")
            else:
                st.write(f"원시 청크 수: {len(texts)}")
                try:
                    index = build_faiss_and_meta(docs, texts)
                    st.success(f"인덱스 생성 완료! · 차원: {index.d} · 벡터 수: {index.ntotal}")
                except Exception as e:
                    st.error(f"인덱스 생성 실패: {e}")

# ------------------------------
# 사이드바: 검색/필터
# ------------------------------
with st.sidebar:
    st.subheader("검색/필터")
    query_mode   = st.radio("검색 기준", ["키워드", "재료"], index=0, horizontal=True)
    time_filter  = st.selectbox("조리 시간", ["(전체)","10분이내","15분이내","20분이내","30분이내","60분이내","90분이내"])
    level_filter = st.selectbox("난이도", ["(전체)","아무나","초급","중급","고급"])

# ==============================
# 유틸 / 전처리
# ==============================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def simhash_64(text: str) -> int:
    h = hashlib.md5((text or "").strip().encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 40) -> List[str]:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return []
    chunks, start = [], 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(s):
        end = min(len(s), start + chunk_size)
        chunks.append(s[start:end])
        start += step
    return chunks

# ===== Chat history → prompt 텍스트로 직조 =====
def build_history_text(messages, max_chars: int = 4000):
    """
    messages: st.session_state.messages (list of {"role","content"})
    - 최신 턴부터 거꾸로 붙여 max_chars를 넘지 않도록 잘라서 반환
    """
    parts = []
    used = 0
    for m in reversed(messages):
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        line = f"[{role.upper()}] {content}\n"
        if used + len(line) > max_chars:
            break
        parts.append(line)
        used += len(line)
    return "".join(reversed(parts))  # 시간순 재정렬

# ==============================
# Ollama Embedding (배치)
# ==============================
def _embed_inputs(texts: List[str]) -> List[str]:
    return [f"Represent this sentence for retrieval: {t}" for t in texts]

def ollama_embed_batch(texts: List[str], model: str, base_url: str, timeout=120) -> np.ndarray:
    url = f"{base_url.rstrip('/')}/api/embed"
    payload = {"model": model, "input": _embed_inputs(texts)}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data:
        return np.array(data["embeddings"], dtype="float32")
    if "embedding" in data:
        return np.array([data["embedding"]], dtype="float32")
    raise RuntimeError(f"Unexpected embed response: {data}")

def embed_corpus(texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        v = ollama_embed_batch(batch, EMBED_MODEL, OLLAMA_BASE_URL)
        vecs.append(v)
        st.sidebar.write(f"임베딩 {i+len(batch)}/{len(texts)} 완료")
    return np.vstack(vecs) if vecs else np.zeros((0, 0), dtype="float32")

# ==============================
# 소스 로더 (CSV/PDF/TXT/JSON/URL)
# ==============================
def _read_csv(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="cp949", on_bad_lines="skip")

def load_csv_rows(csv_path: str) -> List[Tuple[str, Dict]]:
    rows = []
    if not os.path.exists(csv_path):
        return rows
    df = _read_csv(csv_path)
    def getv(rec, col):
        return str(rec.get(col, "")).strip() if (col and col in df.columns and pd.notna(rec.get(col))) else ""
    for i in range(len(df)):
        rec = df.iloc[i]
        name = getv(rec, CSV_NAME_COL)
        ing  = getv(rec, CSV_ING_COL)
        if CSV_DESC_COL and CSV_DESC_COL in df.columns:
            desc = getv(rec, CSV_DESC_COL)
        else:
            parts = []
            labels = ["조리법","상황","재료분류","종류"]
            for col, lab in zip(CLASS_COLS, labels):
                v = getv(rec, col)
                if v:
                    parts.append(f"{lab}:{v}")
            desc = " · ".join(parts)
        text = " | ".join([t for t in [name, ing, desc] if t]).strip(" |")
        meta = {
            "source": "csv",
            "row": i,
            "file": csv_path,
            "cols": [c for c in [CSV_NAME_COL, CSV_ING_COL] if c] + ([CSV_DESC_COL] if CSV_DESC_COL else []),
            CSV_SERVE_COL: getv(rec, CSV_SERVE_COL) if CSV_SERVE_COL else "",
            CSV_LEVEL_COL: getv(rec, CSV_LEVEL_COL) if CSV_LEVEL_COL else "",
            CSV_TIME_COL:  getv(rec, CSV_TIME_COL)  if CSV_TIME_COL  else "",
        }
        if CSV_IMG_COL and CSV_IMG_COL in df.columns:
            meta[CSV_IMG_COL] = getv(rec, CSV_IMG_COL)
        rows.append((text, meta))
    return rows

def load_txt_dir(txt_dir: str) -> List[Tuple[str, Dict]]:
    items = []
    if not os.path.isdir(txt_dir):
        return items
    for p in glob.glob(os.path.join(txt_dir, "**/*.txt"), recursive=True):
        try:
            t = open(p, "r", encoding="utf-8", errors="ignore").read()
            items.append((t, {"source": "txt", "file": p}))
        except Exception:
            pass
    return items

def load_json(json_path: str) -> List[Tuple[str, Dict]]:
    items = []
    if not os.path.exists(json_path):
        return items
    try:
        data = json.load(open(json_path, "r", encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                text = " ".join([str(obj.get(k, "")) for k in obj.keys()])
                items.append((text, {"source": "json", "file": json_path}))
        else:
            text = json.dumps(data, ensure_ascii=False)
            items.append((text, {"source": "json", "file": json_path}))
    except Exception:
        pass
    return items

def load_pdf_texts(pdf_dir: str) -> List[Tuple[str, Dict]]:
    items = []
    if not os.path.isdir(pdf_dir):
        return items
    try:
        from pdfminer_high_level import extract_text  # optional
    except Exception:
        try:
            from pdfminer.high_level import extract_text
        except Exception:
            st.warning("pdfminer.six 미설치로 PDF는 건너뜁니다. (pip install pdfminer.six)")
            return items
    for p in glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True):
        try:
            t = extract_text(p) or ""
            items.append((t, {"source": "pdf", "file": p}))
        except Exception:
            pass
    return items

def load_urls(urls_txt: str) -> List[Tuple[str, Dict]]:
    items = []
    if not os.path.exists(urls_txt):
        return items
    for line in open(urls_txt, "r", encoding="utf-8", errors="ignore"):
        url = line.strip()
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")
            for s in soup(["script","style","noscript"]):
                s.extract()
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            items.append((text, {"source": "url", "url": url}))
        except Exception:
            pass
    return items

# ==============================
# 인덱스 구축/로드
# ==============================
def build_corpus(
    csv_path=DEFAULT_CSV_PATH,
    pdf_dir=DEFAULT_PDF_DIR,
    txt_dir=DEFAULT_TXT_DIR,
    json_path=DEFAULT_JSON_PATH,
    urls_txt=DEFAULT_URLS_TXT,
    chunk_size=1200,
    chunk_overlap=40,
    dedup_hamming=-1
) -> Tuple[List[Document], List[str]]:
    docs = []
    raw = []
    raw += load_csv_rows(csv_path)
    raw += load_pdf_texts(pdf_dir)
    raw += load_txt_dir(txt_dir)
    raw += load_json(json_path)
    raw += load_urls(urls_txt)

    seen = set()
    for text, meta in raw:
        for ch in chunk_text(text, chunk_size, chunk_overlap):
            if not ch:
                continue
            if dedup_hamming >= 0:
                sig = simhash_64(ch)
                if sig in seen:
                    continue
                seen.add(sig)
            docs.append(Document(page_content=ch, metadata=meta))
    return docs, [d.page_content for d in docs]

def build_faiss_and_meta(docs: List[Document], texts: List[str]):
    ensure_dir(INDEX_DIR)
    if not texts:
        raise RuntimeError("임베딩할 텍스트가 없습니다.")
    probe = ollama_embed_batch([texts[0]], EMBED_MODEL, OLLAMA_BASE_URL)
    dim = probe.shape[1]
    vecs = embed_corpus(texts, batch_size=64)
    if vecs.shape[1] != dim:
        raise RuntimeError(f"임베딩 차원 불일치: {vecs.shape[1]} != {dim}")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, FAISS_PATH)
    meta_json = {"docs": [{"text": d.page_content, "meta": d.metadata} for d in docs]}
    json.dump(meta_json, open(META_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    return index

@st.cache_resource(show_spinner=False)
def load_index_and_docs():
    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError("인덱스가 없습니다. 좌측에서 '인덱스 생성'을 먼저 수행하세요.")
    index = faiss.read_index(FAISS_PATH)
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    documents = [Document(page_content=d["text"], metadata=d["meta"]) for d in meta["docs"]]
    return index, documents

# ==============================
# 검색 / 필터 / 보조 도구
# ==============================
def _pass_filter(doc: Document, time_filter: str, level_filter: str) -> bool:
    meta = doc.metadata or {}
    t = meta.get(CSV_TIME_COL, "") or ""
    lv = meta.get(CSV_LEVEL_COL, "") or ""
    ok_time = (time_filter == "(전체)") or (time_filter in t)
    ok_level = (level_filter == "(전체)") or (level_filter in lv)
    return ok_time and ok_level

def _query_text(user_q: str, mode: str) -> str:
    if mode == "재료":
        return f"[INGREDIENT SEARCH] {user_q}"
    return user_q

def faiss_search(query: str, index: faiss.Index, documents: List[Document], k: int = 5):
    qv = ollama_embed_batch([query], EMBED_MODEL, OLLAMA_BASE_URL)
    qv = qv.reshape(1, -1).astype("float32")
    faiss.normalize_L2(qv)
    assert qv.shape[1] == index.d, f"Dim mismatch: query {qv.shape[1]} vs index {index.d}"
    D, I = index.search(qv, k)
    I = I.flatten().tolist()
    return [documents[i] for i in I if 0 <= i < len(documents)]

def extract_row_from_csv(meta: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    csv_path = meta.get("file")
    row_idx = meta.get("row")
    if not csv_path or row_idx is None or not os.path.exists(csv_path):
        return None, None, None, None, None, None, None
    df = _read_csv(csv_path)

    idx = int(row_idx)
    if idx >= len(df) and (idx - 1) >= 0:
        idx -= 1
    if idx < 0 or idx >= len(df):
        return None, None, None, None, None, None, None
    rec = df.iloc[idx]

    def pick(c):
        return (str(rec.get(c, "")).strip() if (c and c in df.columns and pd.notna(rec.get(c, ""))) else "")

    name = pick(CSV_NAME_COL) or pick("title") or pick("name")
    ing  = pick(CSV_ING_COL)

    if CSV_DESC_COL:
        desc = pick(CSV_DESC_COL)
    else:
        parts = []
        for col, label in zip(CLASS_COLS, ["조리법","상황","재료분류","종류"]):
            v = pick(col)
            if v:
                parts.append(f"{label}:{v}")
        desc = " · ".join(parts)

    serve = pick(CSV_SERVE_COL)
    level = pick(CSV_LEVEL_COL)
    ctime = pick(CSV_TIME_COL)
    img   = pick(CSV_IMG_COL) if CSV_IMG_COL else ""

    if not name:
        fallback = (ing or desc or "이름 없음").split("|")[0].strip()
        name = fallback if fallback else "이름 없음"

    return (name or None, ing or None, desc or None,
            serve or None, level or None, ctime or None, (img or None))

def format_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        nm, ing, desc, serve, level, ctime, img = extract_row_from_csv(meta)
        header = nm or f"row {meta.get('row','')}"
        extra = " · ".join([x for x in [serve, level, ctime] if x])
        body  = " / ".join([x for x in [ing, desc] if x]) or (d.page_content[:400] if d.page_content else "")
        if extra:
            header = f"{header} ({extra})"
        blocks.append(f"- {header}\n  {body}")
    return "\n".join(blocks)

# ------------------------------
# 대체 재료 (간단 룰)
# ------------------------------
SUB_DICT = {
    "돼지고기": ["닭가슴살","두부","표고버섯"],
    "고추기름": ["식용유+고춧가루 약불 추출"],
    "간장": ["국간장","진간장"],
    "설탕": ["올리고당","꿀"],
    "전분": ["밀가루 소량","감자전분"],
}
def suggest_substitutions(ing_text: str) -> Dict[str, List[str]]:
    ing_text = ing_text or ""
    out = {}
    for k, v in SUB_DICT.items():
        if k in ing_text:
            out[k] = v
    return out

def tool_fetch_web(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup(["script","style","noscript"]):
            s.extract()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return text[:4000]
    except Exception as e:
        return f"(웹페이지 로드 실패: {e})"

def tool_calorie_heuristic(ingredients_text: str) -> str:
    ing = (ingredients_text or "").lower()
    score = 0
    pos = ["닭가슴살","두부","채소","샐러드","버섯","달걀흰자","현미","귀리","그릭요거트"]
    neg = ["버터","크림","치즈","설탕","튀김","마요네즈","제과","라드","삼겹살","베이컨"]
    for w in pos:
        if w in ing:
            score += 1
    for w in neg:
        if w in ing:
            score -= 1
    guide = "저칼로리 경향" if score >= 1 else ("중간" if score == 0 else "고칼로리 경향")
    return f"간이 판정: {guide} (점수 {score})"

# ==============================
# RAG 답변 (스트리밍) — history 포함
# ==============================
def rag_answer_stream(user_q: str, top_docs: List[Document], history_text: str):
    context = format_context(top_docs)
    ings = []
    for d in top_docs:
        _, ing, *_ = extract_row_from_csv(d.metadata or {})
        if ing: ings.append(ing)
    ing_concat = " ".join(ings)
    subs = suggest_substitutions(ing_concat)
    sub_lines = "\n".join([f"- {k} → {', '.join(v)}" for k,v in subs.items()]) or "- (데이터 기반 대체재 미검출)"

    prompt = f"""You are a helpful cooking assistant. Answer in Korean.

[대화 이력]
{history_text}

[질문]
{user_q}

[검색 요약 컨텍스트]
{context}

[요구]
1) 반드시 '대화 이력'을 고려해 일관된 톤/맥락을 유지
2) 문서 기반으로 답하되 부족하면 일반 상식으로 보완
3) '대체 재료'를 가능하면 제안
4) 구조: 요약 → 추천 1~2개 → 변형 팁 → 주의사항

[대체 재료(룰 기반 후보)]
{sub_lines}
"""
    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
    for ch in llm.stream(prompt):
        yield ch.content if hasattr(ch, "content") else str(ch)

def suggest_followups(user_q: str, docs: List[Document]) -> List[str]:
    base = [
        "비슷한 맛의 다른 요리 추천해줘",
        "남은 재료로 만들 수 있는 메뉴 있어?",
        "조리시간 더 줄이는 방법은?",
        "매운맛 낮추는 변형 알려줘",
    ]
    if "두부" in user_q or any("두부" in d.page_content for d in docs):
        base.insert(0, "두부로 만들 수 있는 다른 단백질 메뉴는?")
    return base[:4]

# ==============================
# 입력/히스토리 렌더
# ==============================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if st.session_state.pending_question:
    user_q = st.session_state.pop("pending_question")
else:
    user_q = st.chat_input("레시피/식단/재료로 질문해보세요 (예: 저칼로리 닭가슴살 요리 추천)")

# ==============================
# 메인 처리
# ==============================
top_docs = []
if user_q:
    # 현재 질문 넣기 '전'에 이전까지 히스토리 추출
    history_text = build_history_text(st.session_state.messages, max_chars=4000)

    # 이제 현재 질문을 기록
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    extra_context = ""
    urls_in_q = re.findall(r"(https?://\S+)", user_q)
    if urls_in_q:
        fetched = []
        for u in urls_in_q[:2]:
            fetched.append(tool_fetch_web(u))
        extra_context = "\n\n[추가 웹본문]\n" + "\n".join(f"- {t[:600]}" for t in fetched if t)

    try:
        index, documents = load_index_and_docs()
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"인덱스 로드 실패: {e}")
        st.stop()

    top_k = 8
    q_for_embed = _query_text(user_q, query_mode)
    top_docs_all = faiss_search(q_for_embed + extra_context, index, documents, k=top_k*2)
    top_docs = [d for d in top_docs_all if _pass_filter(d, time_filter, level_filter)][:top_k]

    with st.expander("🔎 검색된 문서(컨텍스트) 보기"):
        if not top_docs:
            st.info("필터/질문 조건에 맞는 문서가 없습니다. 필터를 완화해보세요.")
        for i, d in enumerate(top_docs, 1):
            meta = d.metadata or {}
            nm, ing, desc, serve, level, ctime, img = extract_row_from_csv(meta)
            title = nm or f"row {meta.get('row','')}"
            sub = " · ".join([x for x in [serve, level, ctime] if x])
            st.markdown(f"**{i}. {title}**")
            if sub: st.caption(sub)
            if img: st.image(img, use_column_width=True)
            txt = " / ".join([x for x in [ing, desc] if x]) or d.page_content[:400]
            st.write(txt)

    ing_all = []
    for d in top_docs:
        nm, ing, *_ = extract_row_from_csv(d.metadata or {})
        if ing: ing_all.append(ing)
    cal_tool_note = ""
    if ing_all:
        cal_tool_note = "\n\n[도구] 칼로리 추정\n" + tool_calorie_heuristic(" ".join(ing_all))

    with st.chat_message("assistant"):
        with st.spinner("답변 작성 중..."):
            try:
                final_chunks = []
                def _capture_stream():
                    # history를 포함해서 RAG 호출
                    for token in rag_answer_stream(user_q + cal_tool_note, top_docs, history_text):
                        final_chunks.append(token)
                        yield token
                st.write_stream(_capture_stream())
                final_text = ''.join(final_chunks)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                st.session_state.last_answer_text = final_text
            except Exception as e:
                err = f"(답변 생성 중 오류: {e})"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.session_state.last_answer_text = err

    st.session_state.last_followups = suggest_followups(user_q, top_docs) if top_docs else []

# ==============================
# 📄 PDF 저장 (robust: fpdf2 → reportlab 폴백)
# ==============================
def _pick_korean_font_path() -> Optional[str]:
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "NanumGothic.ttf"),
        "./NanumGothic.ttf",
        "./fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p) and p.lower().endswith(".ttf"):
            return p
    return None

_EMOJI_REGEX = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002700-\U000027BF" "\U00002600-\U000026FF" "\U0001F1E6-\U0001F1FF" "\U0000203C\U00002049" "]+",
    flags=re.UNICODE
)
def sanitize_for_pdf(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = _EMOJI_REGEX.sub("", s)
    s = s.replace("\t", " ")
    s = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]", "", s)
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\uFEFF", "")
    s = re.sub(r"[^\S\r\n]+", " ", s)
    # 공백 없는 41+ 연속은 강제 개행
    def breaker(m):
        t = m.group(0); return "\n".join(t[i:i+40] for i in range(0, len(t), 40))
    s = re.sub(r"[^\s]{41,}", breaker, s)
    return s

def _make_pdf_fpdf2(text: str, out_path: str, font_path: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=15, top=15, right=15)
    pdf.add_page()
    pdf.add_font("KR", "", font_path)   # 최신 fpdf2: uni 파라미터 불필요
    pdf.set_font("KR", size=12)

    epw = pdf.w - pdf.l_margin - pdf.r_margin
    for raw in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw
        size = 12
        while True:
            try:
                pdf.set_font("KR", size=size)
                if pdf.get_string_width(line) > epw:
                    # 글자수 기반 수동 wrap (긴 토큰 방어)
                    wrapped = []
                    cur = line
                    width_chars = max(20, int((epw / max(1, pdf.get_string_width('가')))))
                    while cur:
                        wrapped.append(cur[:width_chars])
                        cur = cur[width_chars:]
                    for w in wrapped:
                        pdf.multi_cell(w=0, h=6, text=w)
                else:
                    pdf.multi_cell(w=0, h=6, text=line)
                break
            except Exception as e:
                size -= 1
                if size < 7:
                    raise e
    pdf.output(out_path)

def _make_pdf_reportlab(text: str, out_path: str, font_path: str):
    """
    ReportLab 폴백 — Platypus Paragraph + CJK 래핑으로
    한글/공백없는 초장문까지 안전하게 줄바꿈 처리.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except Exception as e:
        raise RuntimeError(f"ReportLab 미설치 또는 로드 실패: {e}")

    # 폰트 등록
    pdfmetrics.registerFont(TTFont("KR", font_path))

    # 문서/여백
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm,
    )

    # 스타일: CJK 래핑 + 긴 토큰 분할 허용
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        "Korean",
        parent=styles["Normal"],
        fontName="KR",
        fontSize=12,
        leading=16,
        wordWrap="CJK",
        splitLongWords=True,
        allowWidows=1,
        allowOrphans=1,
    )

    story = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(line, style))
        story.append(Spacer(1, 4))

    doc.build(story)

def make_pdf_korean(text: str, out_path: str):
    s = sanitize_for_pdf(text)
    font_path = _pick_korean_font_path()
    if not font_path:
        raise RuntimeError("한글 TTF 폰트를 찾지 못했습니다. 이 파일과 같은 폴더에 'NanumGothic.ttf'를 두세요.")
    st.info(f"PDF 생성: 사용 폰트 = {font_path}")
    try:
        _make_pdf_fpdf2(s, out_path, font_path)
        return
    except Exception as e:
        st.warning(f"fpdf2 렌더 실패 → reportlab 폴백 시도: {e}")
    _make_pdf_reportlab(s, out_path, font_path)

with st.expander("📄 PDF로 저장"):
    st.write("현재 대화의 마지막 답변을 PDF로 저장합니다.")
    if st.button("현재 답변을 PDF로 저장", key="btn_pdf_save"):
        try:
            out_pdf = "rag_answer.pdf"
            text_dump = st.session_state.get("last_answer_text", "").strip()
            if not text_dump:
                st.warning("이전에 생성된 답변이 없습니다.")
            else:
                make_pdf_korean(text_dump, out_pdf)
                # ✅ 생성 성공 안내 + 파일 크기 + 다운로드 버튼
                if os.path.exists(out_pdf):
                    st.success(f"PDF 생성 완료: {out_pdf} ({os.path.getsize(out_pdf)} bytes)")
                    with open(out_pdf, "rb") as f:
                        st.download_button("PDF 다운로드", f, file_name=out_pdf, mime="application/pdf", key="btn_pdf_dl")
                else:
                    st.error("PDF 파일이 생성되지 않았습니다.")
        except Exception as e:
            st.error(f"PDF 생성 실패: {e}\n(ReportLab 설치 여부: pip install reportlab)")

# ==============================
# ↪️ 추가질문 (하단 고정 영역)
# ==============================
if st.session_state.last_followups:
    st.markdown("### ↪️ 이런 질문은 어때요?")
    cols = st.columns(min(4, len(st.session_state.last_followups)))
    for i, q in enumerate(st.session_state.last_followups):
        with cols[i % len(cols)]:
            if st.button(q, key=f"follow_fixed_{q}"):
                st.session_state.pending_question = q
                st.rerun()
