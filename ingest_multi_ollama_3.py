#!/usr/bin/env python3
# ingest_multi_ollama_2.py  (parallel + cache + optional batch)
# ------------------------------------------------------------------
# Multi-source ingestion -> chunk -> dedup -> Ollama /api/embed
# -> Parallel/Batch embed + optional on-disk cache -> FAISS index
# ------------------------------------------------------------------
# New features:
# - --max-workers N : parallel HTTP embeds (default: 6)
# - --batch-size N  : send N texts in one /api/embed call (if server supports)
# - --cache-file    : JSON cache of text->embedding (default: disabled)
# - robust requests.Session with retry/backoff
# - order-stable parallelization (keeps original chunk order)
# ------------------------------------------------------------------

import os, re, io, json, argparse, hashlib, pathlib, time
from typing import List, Dict, Any, Iterable, Tuple
import requests
import numpy as np
import faiss
from tqdm import tqdm
from bs4 import BeautifulSoup
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Config defaults
# ----------------------------
DEFAULT_OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL  = os.environ.get("EMBED_MODEL", "mxbai-embed-large")

# ----------------------------
# Text utils
# ----------------------------
def read_txt_file(path: str) -> str:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return pathlib.Path(path).read_text(encoding="cp949")
    except Exception:
        return pathlib.Path(path).read_text(errors="ignore")

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s

def chunk_text(s: str, size: int = 1000, overlap: int = 100) -> List[str]:
    s = s.strip()
    if not s:
        return []
    out = []
    start = 0
    n = len(s)
    while start < n:
        end = min(n, start + size)
        out.append(s[start:end])
        start = end - overlap if end - overlap > start else end
    return out

# ----------------------------
# SimHash for dedup
# ----------------------------
def _hash64(x: bytes) -> int:
    return int(hashlib.blake2b(x, digest_size=8).hexdigest(), 16)

def simhash(text: str, shingles_k: int = 4, hashbits: int = 64) -> int:
    tokens = text.lower().split()
    shingles = [' '.join(tokens[i:i+shingles_k]) for i in range(max(1, len(tokens)-shingles_k+1))]
    v = [0]*hashbits
    for sh in shingles:
        h = _hash64(sh.encode("utf-8"))
        for i in range(hashbits):
            bit = 1 if (h >> i) & 1 else -1
            v[i] += bit
    out = 0
    for i in range(hashbits):
        if v[i] >= 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def dedup_by_simhash(chunks: List[Dict[str, Any]], threshold: int = 3) -> List[Dict[str, Any]]:
    # Keep order, drop near-duplicates
    if threshold < 0:
        return chunks
    threshold = min(threshold, 63)  # cap to 64-bit space
    signatures = []
    out = []
    for ch in chunks:
        sig = simhash(ch['text'])
        if all(hamming(sig, s) > threshold for s in signatures):
            signatures.append(sig)
            out.append(ch)
    return out

# ----------------------------
# Parsers for sources
# ----------------------------
def load_csv_rows(path: str, text_cols: List[str]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    import csv
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            parts = [str(row.get(c, "")) for c in text_cols]
            txt = clean_text(" | ".join(parts))
            yield txt, {"source":"csv", "row":i, "file":path, "cols": text_cols}

def load_json_items(path: str, text_keys: List[str]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    for i, item in enumerate(data):
        if isinstance(item, dict):
            parts = [str(item.get(k, "")) for k in text_keys]
            txt = clean_text(" | ".join(parts))
        else:
            txt = clean_text(str(item))
        yield txt, {"source":"json", "idx":i, "file":path, "keys": text_keys}

def load_pdfs(folder_or_file: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    paths = []
    p = pathlib.Path(folder_or_file)
    if p.is_file():
        paths = [p]
    else:
        paths = list(p.glob("**/*.pdf"))
    for pdf_path in paths:
        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    continue
            txt = clean_text("\n".join(pages))
            yield txt, {"source":"pdf", "file":str(pdf_path)}
        except Exception:
            continue

def load_txts(folder_or_file: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    paths = []
    p = pathlib.Path(folder_or_file)
    if p.is_file():
        paths = [p]
    else:
        paths = list(p.glob("**/*.txt"))
    for t in paths:
        try:
            txt = clean_text(read_txt_file(str(t)))
            yield txt, {"source":"txt", "file":str(t)}
        except Exception:
            continue

def load_urls(urls_file: str, timeout: int = 20) -> Iterable[Tuple[str, Dict[str, Any]]]:
    # urls_file: each line a URL
    for line in open(urls_file, "r", encoding="utf-8", errors="ignore"):
        url = line.strip()
        if not url:
            continue
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            # simple boilerplate removal
            for tag in soup(["script","style","noscript"]):
                tag.extract()
            txt = clean_text(soup.get_text(separator=" "))
            yield txt, {"source":"url", "url":url}
        except Exception:
            continue

# ----------------------------
# Ollama embed (single or batch) + cache + session + retry
# ----------------------------
class Embedder:
    def __init__(self, model: str, base_url: str, cache_path: str = "", timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, 'r', encoding='utf-8'))
            except Exception:
                self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ingest-embedder/1.1"})

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _save_cache(self):
        if self.cache_path:
            try:
                json.dump(self.cache, open(self.cache_path, 'w', encoding='utf-8'))
            except Exception:
                pass

    def embed_one(self, text: str) -> List[float]:
        k = self._key(text)
        if k in self.cache:
            return self.cache[k]
        payload = {"model": self.model, "input": f"Represent this sentence for retrieval: {text}"}
        url = f"{self.base_url}/api/embed"
        last_err = None
        for attempt in range(4):
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                vec = data.get("embedding")
                if vec is None:
                    # some servers return {embeddings: [[...]]}
                    emb_list = data.get("embeddings")
                    if isinstance(emb_list, list) and emb_list:
                        vec = emb_list[0]
                if vec is None:
                    raise RuntimeError(f"Unexpected embed response: {data}")
                self.cache[k] = vec
                self._save_cache()
                return vec
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Embed failed after retries: {last_err}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # cache hit mask
        keys = [self._key(t) for t in texts]
        out: List[List[float]] = [None] * len(texts)  # type: ignore
        missing_idx = [i for i, k in enumerate(keys) if k not in self.cache]
        if not missing_idx:
            return [self.cache[k] for k in keys]

        # build batch payload only for missing items
        inputs = [f"Represent this sentence for retrieval: {texts[i]}" for i in missing_idx]
        payload = {"model": self.model, "input": inputs}
        url = f"{self.base_url}/api/embed"

        last_err = None
        for attempt in range(4):
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                emb_list = data.get("embeddings")
                if not isinstance(emb_list, list):
                    # server may not support batch; fallback one-by-one
                    for i in missing_idx:
                        out[i] = self.embed_one(texts[i])
                    break
                # map back to indexes
                for j, i in enumerate(missing_idx):
                    vec = emb_list[j]
                    self.cache[keys[i]] = vec
                    out[i] = vec
                self._save_cache()
                break
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))
        # fill cached
        for i, k in enumerate(keys):
            if out[i] is None:
                if k in self.cache:
                    out[i] = self.cache[k]
                else:
                    # if still missing, try single
                    out[i] = self.embed_one(texts[i])
        return out  # type: ignore

# ----------------------------
# Main pipeline
# ----------------------------

def build_chunks(args) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    if args.csv:
        cols = [c.strip() for c in args.csv_text_cols.split(",")] if args.csv_text_cols else []
        for txt, md in load_csv_rows(args.csv, cols or []):
            for ch in chunk_text(txt, args.chunk_size, args.chunk_overlap):
                if ch:
                    chunks.append({"text": ch, "meta": md})

    if args.pdfs:
        for txt, md in load_pdfs(args.pdfs):
            for ch in chunk_text(txt, args.chunk_size, args.chunk_overlap):
                if ch:
                    chunks.append({"text": ch, "meta": md})

    if args.json:
        keys = [k.strip() for k in args.json_text_keys.split(",")] if args.json_text_keys else []
        for txt, md in load_json_items(args.json, keys or []):
            for ch in chunk_text(txt, args.chunk_size, args.chunk_overlap):
                if ch:
                    chunks.append({"text": ch, "meta": md})

    if args.urls:
        for txt, md in load_urls(args.urls):
            for ch in chunk_text(txt, args.chunk_size, args.chunk_overlap):
                if ch:
                    chunks.append({"text": ch, "meta": md})

    if args.txts:
        for txt, md in load_txts(args.txts):
            for ch in chunk_text(txt, args.chunk_size, args.chunk_overlap):
                if ch:
                    chunks.append({"text": ch, "meta": md})

    return chunks


def embed_all_parallel(chunks: List[Dict[str, Any]], dim: int, embedder: Embedder, max_workers: int, batch_size: int) -> np.ndarray:
    embs = np.zeros((len(chunks), dim), dtype="float32")

    if batch_size <= 1:
        # one-by-one in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(embedder.embed_one, ch["text"]): i for i, ch in enumerate(chunks)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="[embed-parallel]"):
                i = futures[fut]
                try:
                    embs[i] = np.array(fut.result(), dtype="float32")
                except Exception as e:
                    print(f"[WARN] embed failed at #{i}: {e}")
                    embs[i] = np.zeros((dim,), dtype="float32")
        return embs

    # Batch mode: group indices, submit batches to workers
    def batch_iter(n, bsz):
        i = 0
        while i < n:
            j = min(n, i + bsz)
            yield range(i, j)
            i = j

    # submit per-batch; inside worker we call embed_batch (which may fallback)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for idxs in batch_iter(len(chunks), batch_size):
            idxs = list(idxs)
            texts = [chunks[i]["text"] for i in idxs]
            futures[ex.submit(embedder.embed_batch, texts)] = idxs
        for fut in tqdm(as_completed(futures), total=len(futures), desc="[embed-batch]"):
            idxs = futures[fut]
            try:
                vecs = fut.result()
                for off, i in enumerate(idxs):
                    embs[i] = np.array(vecs[off], dtype="float32")
            except Exception as e:
                print(f"[WARN] batch embed failed for idxs {idxs[:3]}..: {e}")
                # fallback: try one by one
                for i in idxs:
                    try:
                        embs[i] = np.array(embedder.embed_one(chunks[i]["text"]), dtype="float32")
                    except Exception:
                        embs[i] = np.zeros((dim,), dtype="float32")

    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="CSV file path")
    ap.add_argument("--csv-text-cols", type=str, default="", help="CSV columns to concatenate as text, comma-separated")

    ap.add_argument("--pdfs", type=str, help="PDF file or folder")
    ap.add_argument("--json", type=str, help="JSON file path")
    ap.add_argument("--json-text-keys", type=str, default="", help="JSON dict keys to concatenate as text, comma-separated")

    ap.add_argument("--urls", type=str, help="Text file containing URLs (one per line)")
    ap.add_argument("--txts", type=str, help="TXT file or folder")

    ap.add_argument("--outdir", type=str, required=True, help="Output folder for FAISS and metadata")
    ap.add_argument("--embed-model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA)

    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=150)

    ap.add_argument("--dedup-hamming", type=int, default=3, help="SimHash dedup threshold (lower = more aggressive; negative to disable)")

    # new options
    ap.add_argument("--max-workers", type=int, default=6, help="Number of parallel embed workers")
    ap.add_argument("--cache-file", type=str, default="", help="JSON cache file path for embeddings (optional)")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for /api/embed (server may fallback to single)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Build chunks
    chunks = build_chunks(args)
    print(f"[ingest] built raw chunks: {len(chunks)}")

    # 2) Deduplicate
    if args.dedup_hamming < 0:
        print("[ingest] dedup disabled")
    else:
        chunks = dedup_by_simhash(chunks, threshold=args.dedup_hamming)
    print(f"[ingest] after dedup: {len(chunks)}")

    if len(chunks) < 50:
        print("[WARN] Fewer than 50 chunks. Consider adding more sources or lowering chunk_size.")

    # 3) Probe embedding dim
    probe_embedder = Embedder(args.embed_model, args.ollama_url, cache_path=args.cache_file)
    dim = len(probe_embedder.embed_one("dim probe"))
    print(f"[ingest] embed dim = {dim}")

    # 4) Embed all (parallel + optional batch)
    embs = embed_all_parallel(chunks, dim, probe_embedder, max_workers=max(1, args.max_workers), batch_size=max(1, args.batch_size))

    # 5) Build FAISS
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    # 6) Persist
    faiss.write_index(index, os.path.join(args.outdir, "faiss.index"))
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": args.embed_model,
        "ollama_url": args.ollama_url,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "dedup_hamming": args.dedup_hamming,
        "docs": [{"text": ch["text"], "meta": ch["meta"]} for ch in chunks],
    }
    json.dump(meta, open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[ingest] done. Saved to {args.outdir}/faiss.index and {args.outdir}/meta.json")
    print(f"[ingest] total vectors: {len(chunks)}")

if __name__ == "__main__":
    main()