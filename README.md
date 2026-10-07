# ReadBot: Document Q&A with RAG

ReadBot lets you upload a PDF and ask questions about it in plain English. It finds the parts of the document that relate to your question and sends only those parts to a Large Language Model (Groq API). The model answers **only from the document**, cites the **page numbers** it used, and says *"I couldn't find this in the document."* when the answer is not there.

This method is called **Retrieval-Augmented Generation (RAG)**.

## Workflow

```
PDF upload
  -> 1. Extract text per page (pypdf) and clean it
  -> 2. Split into overlapping, sentence-aware chunks (each keeps its page number)
  -> 3. Embed chunks with multilingual-e5-small (cached on disk)
User question
  -> 4. A small LLM rewrites it into a standalone query + alternative wordings
        (uses the chat history, so follow-ups like "what about it?" work)
  -> 5. Hybrid search: meaning (embeddings) + keywords (BM25) for every wording,
        merged with reciprocal rank fusion
  -> 6. Build the context: best matches + neighbouring chunks (whole document if small;
        chunks spread over the document for broad questions)
  -> 7. Send context + history + question to the Groq LLM using prompts.yaml
  -> Answer with page citations + the search details shown for checking
```

## Project files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `rag.py` | The NLP pipeline (extract, clean, chunk, embed, retrieve, generate) |
| `prompts.yaml` | All prompts sent to the LLM |
| `config.yaml` | All settings (model, temperature, chunk size, top_k, ...) |
| `requirements.txt` | Python packages |
| `.env.example` | Template for your API key |
| `.gitignore` | Keeps `.env` and caches out of Git |

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Get a free API key from https://console.groq.com/keys, then:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your key. **Never commit `.env`.**

## Usage

```bash
streamlit run app.py
```

1. Open the link shown in the terminal (usually http://localhost:8501).
2. Upload a PDF that has selectable text (scanned image PDFs need OCR first).
3. Type a question and click **Ask**.
4. Read the answer, the cited pages, and open **Retrieved chunks** to see the exact text the answer was based on.
5. Click **Summarize document** for a short summary in bullet points.

The first run downloads the embedding model (about 470 MB). Settings can be changed in `config.yaml` without touching the code.

## Example

Sample PDF: a 3-page note about solar energy (page 2 covers costs and warranties).

> **Q:** How long is the warranty on solar panels?
>
> **A:** Solar panels typically come with a 25-year performance warranty (p. 2).
>
> *Sources: page 2*

> **Q:** Who invented the telephone?
>
> **A:** I couldn't find this in the document.

## Where the NLP is

| NLP step | Where | What it does |
|----------|-------|--------------|
| **Text cleaning** | `rag.clean_text` | Joins words split by hyphens at line ends, turns line breaks into spaces, collapses extra spaces, removes control characters. |
| **Chunking** | `rag.split_sentences`, `rag.chunk_page` | Splits text into sentences, then groups sentences into ~800-character chunks. Neighbouring chunks share ~150 characters of overlap so an idea cut at a border is not lost. Each chunk keeps its page number. |
| **Embeddings** | `rag.embed_texts`, `rag.embed_chunks` | Turns each chunk into a 384-dimensional vector with `intfloat/multilingual-e5-small`, which understands English, Hindi and ~100 other languages. Vectors are cached in `.cache/` so the same PDF is not embedded twice. |
| **Query expansion** | `rag.expand_query`, `prompts.yaml` | A small model (`helper_model`) rewrites the question into a standalone query plus up to 3 alternative wordings, translates non-English questions, and flags broad questions. |
| **Hybrid search** | `rag.bm25_scores`, `rag.rank_chunks` | Ranks chunks by meaning (cosine similarity) and by exact keywords (BM25) for every query wording, then merges the rankings with reciprocal rank fusion. Finds both paraphrases and exact names/numbers. |
| **Context building** | `rag.select_context` | Sends the whole document when it fits the budget (`max_context_chars`); otherwise the best matches plus their neighbouring chunks, and for broad questions chunks spread across the document. |
| **Grounded generation** | `rag.answer_question`, `prompts.yaml` | The LLM gets the context (labelled with page numbers) and recent conversation, is told to read all of it, combine chunks, give complete and partial answers, cite pages, and only say "not found" when nothing is relevant. |

## Error handling

- **Empty or scanned PDF:** shows "No text found in this PDF".
- **Missing API key:** warning in the sidebar and a clear error when you ask.
- **API failure:** each call has a timeout. Temporary errors (timeouts, rate limits, server errors) are retried; rate limits wait as long as Groq asks (`retry-after`), others back off 2s, 4s, ... A wrong API key is reported straight away.
