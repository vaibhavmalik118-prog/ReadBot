"""RAG pipeline: PDF -> clean text -> chunks -> embeddings -> retrieval -> grounded answer."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=।)\s+")  # English + Hindi danda


# ---------- Errors shown to the user ----------

class RAGError(Exception):
    """Base error with a message that is safe to show in the UI."""


class EmptyDocumentError(RAGError):
    """The PDF has no extractable text (e.g. it is scanned or blank)."""


class MissingAPIKeyError(RAGError):
    """GROQ_API_KEY is not set."""


class LLMError(RAGError):
    """The Groq API call failed."""


# ---------- Data types ----------

@dataclass
class Page:
    number: int
    text: str


@dataclass
class Chunk:
    chunk_id: int
    page: int
    text: str


# ---------- Config ----------

def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML file that sits next to this module."""
    with open(BASE_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> dict[str, Any]:
    """Return settings from config.yaml."""
    return load_yaml("config.yaml")


def load_prompts() -> dict[str, str]:
    """Return prompt templates from prompts.yaml."""
    return load_yaml("prompts.yaml")


# ---------- 1. Extraction and cleaning ----------

def clean_text(text: str) -> str:
    """Normalise raw PDF text: fix hyphenated line breaks, join lines, collapse spaces."""
    text = text.replace("\x00", "").replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # "embed-\nding" -> "embedding"
    text = re.sub(r"\s+", " ", text)                       # line breaks / many spaces -> one space
    text = "".join(ch for ch in text if ch.isprintable())  # drop control characters
    return text.strip()


def extract_pages(pdf_bytes: bytes) -> list[Page]:
    """Read a PDF and return cleaned text for every non-empty page (1-based numbers)."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, OSError) as exc:
        raise RAGError(f"Could not read the PDF: {exc}") from exc

    pages = [Page(i + 1, clean_text(t)) for i, t in enumerate(raw_pages)]
    pages = [p for p in pages if p.text]
    if not pages:
        raise EmptyDocumentError(
            "No text found in this PDF. It may be empty or a scanned image (needs OCR)."
        )
    return pages


# ---------- 2. Sentence-aware chunking ----------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences at ., ! or ? followed by a capital letter or digit."""
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break a sentence longer than max_chars into word-boundary pieces."""
    pieces, current = [], ""
    for word in sentence.split():
        if current and len(current) + len(word) + 1 > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    """Return the last sentences whose total length fits within `overlap` characters."""
    tail, size = [], 0
    for sentence in reversed(sentences):
        if size + len(sentence) > overlap:
            break
        tail.insert(0, sentence)
        size += len(sentence) + 1
    return tail


def chunk_page(page: Page, chunk_size: int, overlap: int) -> list[str]:
    """Group a page's sentences into ~chunk_size pieces that overlap by ~overlap chars."""
    sentences = []
    for s in split_sentences(page.text):
        sentences.extend(split_long_sentence(s, chunk_size) if len(s) > chunk_size else [s])

    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current)) + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            current = overlap_tail(current, overlap)
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[Page], cfg: dict[str, Any]) -> list[Chunk]:
    """Chunk every page and keep the page number on each chunk."""
    c = cfg["chunking"]
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_page(page, c["chunk_size"], c["chunk_overlap"]):
            if len(text) >= c["min_chunk_chars"]:
                chunks.append(Chunk(len(chunks), page.number, text))
    if not chunks:
        raise EmptyDocumentError("The PDF text was too short to build any chunks.")
    return chunks


# ---------- 3. Embeddings with a disk cache ----------

def load_embedder(cfg: dict[str, Any]) -> SentenceTransformer:
    """Load the sentence-transformers model named in config.yaml."""
    return SentenceTransformer(cfg["embeddings"]["model"])


def cache_path(chunks: list[Chunk], cfg: dict[str, Any]) -> Path:
    """Build a cache file name from the model, prefix and the exact chunk texts."""
    emb = cfg["embeddings"]
    key = emb["model"] + emb.get("passage_prefix", "") + "".join(c.text for c in chunks)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return BASE_DIR / emb["cache_dir"] / f"{digest}.npy"


def embed_texts(
    texts: list[str], embedder: SentenceTransformer, cfg: dict[str, Any], prefix: str = ""
) -> np.ndarray:
    """Encode texts into unit-length vectors (so dot product = cosine similarity)."""
    return embedder.encode(
        [prefix + t for t in texts],
        batch_size=cfg["embeddings"]["batch_size"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def embed_chunks(chunks: list[Chunk], embedder: SentenceTransformer, cfg: dict[str, Any]) -> np.ndarray:
    """Return chunk embeddings, loading them from the cache when available."""
    path = cache_path(chunks, cfg)
    if path.exists():
        return np.load(path)
    prefix = cfg["embeddings"].get("passage_prefix", "")
    vectors = embed_texts([c.text for c in chunks], embedder, cfg, prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


# ---------- 4. Hybrid retrieval (meaning + keywords) ----------

TOKEN = re.compile(r"\w+", re.UNICODE)
STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "it its this that these those what which who whom whose when where why how do does did "
    "can could should would will shall may might must i you he she we they me him her us them "
    "my your our their about into over than then there here not no yes so such all any some "
    "please tell explain give show describe document pdf".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without common stopwords (used for keyword search)."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector and each row of a matrix."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    return (matrix @ query_vec) / np.maximum(norms, 1e-10)


def bm25_scores(query: str, docs: list[Counter[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25 keyword relevance of every chunk for the query (exact words, names, numbers)."""
    terms = set(tokenize(query))
    scores = np.zeros(len(docs), dtype=np.float32)
    if not terms:
        return scores
    lengths = [sum(d.values()) for d in docs]
    avg_len = max(sum(lengths) / len(docs), 1.0)
    n = len(docs)
    for term in terms:
        df = sum(1 for d in docs if term in d)
        if df == 0:
            continue
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs):
            tf = d.get(term, 0)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[i] / avg_len))
    return scores


def rank_chunks(
    queries: list[str],
    chunks: list[Chunk],
    embeddings: np.ndarray,
    embedder: SentenceTransformer,
    cfg: dict[str, Any],
) -> tuple[list[int], np.ndarray]:
    """Fuse meaning and keyword rankings for several query wordings (reciprocal rank fusion).

    Returns chunk indices from most to least relevant, and the best meaning-similarity
    of each chunk (shown in the UI).
    """
    r = cfg["retrieval"]
    query_vecs = embed_texts(queries, embedder, cfg, cfg["embeddings"].get("query_prefix", ""))
    docs = [Counter(tokenize(c.text)) for c in chunks]
    fused = np.zeros(len(chunks), dtype=np.float32)
    best_sim = np.full(len(chunks), -1.0, dtype=np.float32)
    for query, vec in zip(queries, query_vecs):
        sim = cosine_scores(vec, embeddings)
        best_sim = np.maximum(best_sim, sim)
        rankings = [(sim, 1.0), (bm25_scores(query, docs), r["keyword_weight"])]
        for scores, weight in rankings:
            if not scores.any():
                continue
            for rank, i in enumerate(np.argsort(scores)[::-1][: r["candidates"]]):
                fused[i] += weight / (r["rrf_k"] + rank + 1)
    order = [int(i) for i in np.argsort(fused)[::-1] if fused[i] > 0]
    return order, best_sim


def spread_indices(n_chunks: int, count: int) -> list[int]:
    """Evenly spaced chunk indices from start to end of the document."""
    count = min(count, n_chunks)
    return sorted({int(i) for i in np.linspace(0, n_chunks - 1, count)}) if count else []


def select_context(
    order: list[int], chunks: list[Chunk], cfg: dict[str, Any], broad: bool
) -> list[int]:
    """Pick chunks for the prompt within the character budget, in document order.

    Small documents are sent whole. Otherwise the best matches go first, each with its
    neighbouring chunks so sentences cut at a border are not lost. Broad questions also
    get chunks spread across the whole document.
    """
    r = cfg["retrieval"]
    budget = r["max_context_chars"]
    if sum(len(c.text) for c in chunks) <= budget:
        return list(range(len(chunks)))

    candidates: list[int] = []
    if broad:
        top = order[: r["top_k"] // 2]
        spread = spread_indices(len(chunks), budget // max(r["chunk_size_hint"], 1))
        candidates = [i for pair in zip(top, spread) for i in pair] + spread
    for i in order[: r["top_k"]]:
        for j in range(i - r["neighbor_chunks"], i + r["neighbor_chunks"] + 1):
            candidates.append(j)
    candidates += order  # fill any remaining space with the next best matches

    picked: list[int] = []
    used = 0
    for i in candidates:
        if not 0 <= i < len(chunks) or i in picked:
            continue
        size = len(chunks[i].text)
        if used + size > budget:
            continue
        picked.append(i)
        used += size
        if used >= budget * 0.97:
            break
    return sorted(picked)


# ---------- 5. Grounded generation with Groq ----------

def get_api_key() -> str:
    """Read GROQ_API_KEY from the environment / .env file."""
    load_dotenv(BASE_DIR / ".env", override=True)  # pick up edits without a restart
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        raise MissingAPIKeyError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return key


def retry_wait(exc: Exception, attempt: int, llm: dict[str, Any]) -> float:
    """Seconds to wait before retrying: Groq's retry-after header if given, else backoff."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        return min(float(header), 60.0) if header else llm["backoff_seconds"] * 2 ** (attempt - 1)
    except ValueError:
        return llm["backoff_seconds"] * 2 ** (attempt - 1)


def call_llm(
    messages: list[dict[str, str]],
    cfg: dict[str, Any],
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Call Groq with a timeout and retries on temporary errors and rate limits."""
    llm = cfg["llm"]
    client = Groq(api_key=get_api_key(), timeout=llm["timeout_seconds"], max_retries=0)
    retryable = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
    extra = {}
    effort = reasoning_effort or llm.get("reasoning_effort")
    if effort:
        extra["reasoning_effort"] = effort

    for attempt in range(1, llm["max_retries"] + 1):
        try:
            response = client.chat.completions.create(
                model=model or llm["model"],
                messages=messages,
                temperature=llm["temperature"],
                max_tokens=max_tokens or llm["max_tokens"],
                **extra,
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if not text:
                if choice.finish_reason == "length":
                    raise LLMError(
                        "The model ran out of space before answering. Try a more specific "
                        "question, or raise max_tokens in config.yaml."
                    )
                raise LLMError("The model returned an empty answer. Please ask again.")
            return text
        except AuthenticationError as exc:
            raise LLMError("Groq rejected the API key. Check GROQ_API_KEY in .env.") from exc
        except retryable as exc:
            if attempt == llm["max_retries"]:
                if isinstance(exc, RateLimitError):
                    raise LLMError(
                        "Groq's free rate limit was reached. Wait a minute and ask again."
                    ) from exc
                raise LLMError(f"Groq API failed after {attempt} attempts: {exc}") from exc
            time.sleep(retry_wait(exc, attempt, llm))
        except APIStatusError as exc:
            raise LLMError(f"Groq API error ({exc.status_code}): {exc.message}") from exc
    raise LLMError("Groq API call did not return a response.")


def format_history(history: list[tuple[str, str]], cfg: dict[str, Any]) -> str:
    """Recent question/answer turns as plain text (answers trimmed to save tokens)."""
    turns = history[-cfg["retrieval"]["history_turns"]:]
    if not turns:
        return "(no earlier questions)"
    return "\n".join(f"User: {q}\nReadBot: {a[:400]}" for q, a in turns)


def expand_query(
    question: str, history: list[tuple[str, str]], cfg: dict[str, Any], prompts: dict[str, str]
) -> tuple[list[str], bool]:
    """Ask a small model for a standalone query plus alternative wordings.

    Returns (queries, broad). Falls back to the original question if anything fails.
    """
    if not cfg["retrieval"]["query_expansion"]:
        return [question], False
    prompt = prompts["query_rewrite_prompt"].format(
        history=format_history(history, cfg), question=question
    )
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}], cfg,
            model=cfg["llm"]["helper_model"], max_tokens=600, reasoning_effort="low",
        )
        data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
    except (RAGError, AttributeError, ValueError):
        return [question], False
    queries = [question, str(data.get("standalone", ""))] + [str(a) for a in data.get("alternatives", [])]
    unique = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
    return unique[:5], bool(data.get("broad", False))


def build_context(chunks: list[Chunk], prompts: dict[str, str]) -> str:
    """Join chunks into one context string labelled with page numbers."""
    template = prompts["context_chunk_template"]
    return "\n\n".join(template.format(page=c.page, text=c.text) for c in chunks)


def system_message(prompts: dict[str, str]) -> dict[str, str]:
    """Build the system message with the not-found phrase filled in."""
    content = prompts["system_prompt"].format(not_found_message=prompts["not_found_message"])
    return {"role": "system", "content": content}


def cited_pages(answer: str, valid: set[int] | None = None) -> list[int]:
    """Page numbers the answer cites, e.g. "(p. 3)", "(pp. 4-5)", "(p. 2, 7)".

    Pages not in `valid` (when given) are dropped, so a typo can't show a page that doesn't exist.
    """
    dash = r"\-‐‑‒–—"  # hyphen and the Unicode dashes LLMs often use
    pages: set[int] = set()
    for group in re.findall(rf"\(pp?\.\s*([\d\s,{dash}and]+)\)", answer):
        for a, b in re.findall(rf"(\d+)\s*[{dash}]\s*(\d+)", group):
            if int(b) - int(a) < 500:
                pages.update(range(int(a), int(b) + 1))
        pages.update(int(n) for n in re.findall(r"\d+", group))
    return sorted(p for p in pages if valid is None or p in valid)


def answer_question(
    question: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    embedder: SentenceTransformer,
    cfg: dict[str, Any],
    prompts: dict[str, str],
    history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Expand the question, retrieve with hybrid search, and ask the LLM for a cited answer."""
    history = history or []
    queries, broad = expand_query(question, history, cfg, prompts)
    order, best_sim = rank_chunks(queries, chunks, embeddings, embedder, cfg)
    picked = select_context(order, chunks, cfg, broad)
    context_chunks = [chunks[i] for i in picked]

    user_prompt = prompts["answer_template"].format(
        history=format_history(history, cfg),
        context=build_context(context_chunks, prompts),
        question=question,
        not_found_message=prompts["not_found_message"],
    )
    answer = call_llm([system_message(prompts), {"role": "user", "content": user_prompt}], cfg)

    rank_of = {i: r for r, i in enumerate(order)}
    shown = sorted(picked, key=lambda i: rank_of.get(i, len(order)))[: cfg["retrieval"]["top_k"]]
    page_scores: dict[int, float] = {}
    for chunk, score in zip(chunks, best_sim):
        page_scores[chunk.page] = max(page_scores.get(chunk.page, -1.0), float(score))
    not_found = answer.strip().strip('"') == prompts["not_found_message"]
    return {
        "answer": answer,
        "not_found": not_found,
        "sources": [(chunks[i], float(best_sim[i])) for i in shown],
        "cited_pages": [] if not_found else cited_pages(answer, set(page_scores)),
        "page_scores": page_scores,
        "queries": queries,
        "context_pages": sorted({c.page for c in context_chunks}),
    }


def summarize_document(chunks: list[Chunk], cfg: dict[str, Any], prompts: dict[str, str]) -> str:
    """Summarise the document from chunks spread evenly from start to end."""
    r = cfg["retrieval"]
    count = r["max_context_chars"] // max(r["chunk_size_hint"], 1)
    picks = [chunks[i] for i in spread_indices(len(chunks), count)]
    while len(picks) > 1 and sum(len(c.text) for c in picks) > r["max_context_chars"]:
        picks = picks[::2]
    user_prompt = prompts["summary_prompt"].format(context=build_context(picks, prompts))
    return call_llm([system_message(prompts), {"role": "user", "content": user_prompt}], cfg)


# ---------- 6. Document insights ----------

def top_keywords(pages: list[Page], n: int = 10) -> list[tuple[str, int]]:
    """Most frequent meaningful words in the document (no stopwords, numbers or short words)."""
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update(t for t in tokenize(page.text) if len(t) > 3 and not t.isdigit())
    return counts.most_common(n)
