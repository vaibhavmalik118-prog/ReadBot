"""Streamlit UI for ReadBot: upload a PDF, ask questions, get cited answers."""

from __future__ import annotations

import hashlib
import html
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from streamlit import config as st_config

import rag

CFG = rag.load_config()
PROMPTS = rag.load_prompts()

SUGGESTIONS = [
    "📝 Summarize the main points",
    "🔢 What are the key numbers and dates?",
    "📖 List the important terms and what they mean",
]

# Streamlit's own theme (widgets, inputs, file chips, buttons) + extra colours for our CSS/charts.
THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "streamlit": {
            "base": "light", "primaryColor": "#6C5CE7", "backgroundColor": "#F5F4FB",
            "secondaryBackgroundColor": "#ECEAF8", "textColor": "#1E1E2F", "borderColor": "#E2DFF3",
        },
        "css": {
            "bg": "#F5F4FB", "glow": "#E6E1FF", "surface": "#FFFFFF", "surface2": "#F6F5FD",
            "text": "#1E1E2F", "muted": "#6E6E87", "border": "#E6E3F5", "accent": "#6C5CE7",
            "pill_bg": "#EFEBFF", "pill_text": "#5B4BD6", "user_bg": "#EFEBFF",
            "shadow": "rgba(40, 30, 90, 0.08)",
        },
        "chart": {"mark": "#6C5CE7", "faded": "#CFC9F5", "axis": "#6E6E87", "grid": "#ECEAF4"},
    },
    "dark": {
        "streamlit": {
            "base": "dark", "primaryColor": "#8B7CF0", "backgroundColor": "#0F1020",
            "secondaryBackgroundColor": "#1E1D36", "textColor": "#ECEBFA", "borderColor": "#2E2D4D",
        },
        "css": {
            "bg": "#0F1020", "glow": "#1F1B45", "surface": "#18182C", "surface2": "#1E1D36",
            "text": "#ECEBFA", "muted": "#A3A2C2", "border": "#2E2D4D", "accent": "#8B7CF0",
            "pill_bg": "#2D2860", "pill_text": "#D2CBFF", "user_bg": "#26224F",
            "shadow": "rgba(0, 0, 0, 0.35)",
        },
        "chart": {"mark": "#8B7CF0", "faded": "#3A3666", "axis": "#A3A2C2", "grid": "#26253F"},
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {{
    --bg: {bg}; --glow: {glow}; --surface: {surface}; --surface2: {surface2};
    --text: {text}; --muted: {muted}; --border: {border}; --accent: {accent};
    --pill-bg: {pill_bg}; --pill-text: {pill_text}; --user-bg: {user_bg}; --shadow: {shadow};
}}

html, body, .stApp, button, input, textarea, [data-testid="stMarkdownContainer"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
}}
.stApp {{
    background:
        radial-gradient(1200px 500px at 15% -10%, var(--glow) 0, transparent 60%),
        radial-gradient(900px 500px at 100% 0%, var(--glow) 0, transparent 55%),
        var(--bg);
}}
[data-testid="stHeader"] {{background: transparent;}}
.block-container {{padding-top: 2.2rem; max-width: 960px;}}
#MainMenu, footer, [data-testid="stToolbar"] {{visibility: hidden;}}

/* ---------- sidebar ---------- */
.brand {{display: flex; align-items: center; gap: 12px; margin: 4px 0 10px;}}
.brand .logo {{
    width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center; font-size: 1.3rem;
    background: linear-gradient(135deg, #6C5CE7, #00B4D8); box-shadow: 0 6px 16px rgba(108, 92, 231, .35);
}}
.brand b {{font-size: 1.2rem; color: var(--text);}}
.brand small {{display: block; color: var(--muted); font-size: .78rem;}}
.step {{display: flex; gap: 10px; align-items: center; margin: 10px 0; color: var(--text);}}
.step .num {{
    min-width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center;
    background: var(--pill-bg); color: var(--pill-text); font-size: .78rem; font-weight: 700;
}}

/* ---------- hero ---------- */
@keyframes shift {{0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}}}}
@keyframes float {{0%, 100% {{transform: translateY(0);}} 50% {{transform: translateY(-8px);}}}}
.hero {{
    position: relative; overflow: hidden; color: #fff;
    background: linear-gradient(120deg, #6C5CE7, #A06CF5, #00B4D8, #6C5CE7);
    background-size: 300% 300%; animation: shift 14s ease infinite;
    border-radius: 24px; padding: 34px 38px; margin-bottom: 22px;
    box-shadow: 0 20px 44px rgba(108, 92, 231, 0.30);
}}
.hero::before, .hero::after {{
    content: ""; position: absolute; border-radius: 50%; background: rgba(255, 255, 255, 0.12);
}}
.hero::before {{width: 240px; height: 240px; right: -70px; top: -80px;}}
.hero::after {{width: 120px; height: 120px; right: 140px; bottom: -60px; background: rgba(255,255,255,.08);}}
.hero .doc {{
    position: absolute; right: 44px; top: 50%; margin-top: -42px; font-size: 4.2rem;
    animation: float 4s ease-in-out infinite; filter: drop-shadow(0 10px 18px rgba(0,0,0,.2));
}}
.hero h1 {{color: #fff; margin: 0; padding: 0; font-size: 2.5rem; font-weight: 800; letter-spacing: -.02em;}}
.hero p {{color: rgba(255,255,255,.92); margin: 8px 0 0; font-size: 1.05rem; max-width: 560px;}}
.hero .badges {{margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap;}}
.hero .badge {{
    padding: 5px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600; color: #fff;
    background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.25); backdrop-filter: blur(6px);
}}
@media (max-width: 700px) {{.hero .doc {{display: none;}} .hero h1 {{font-size: 1.9rem;}}}}

/* ---------- cards ---------- */
.card-row {{display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 6px 0 18px;}}
.stat, .feature {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
    box-shadow: 0 6px 18px var(--shadow); transition: transform .18s ease, box-shadow .18s ease;
}}
.stat:hover, .feature:hover {{transform: translateY(-3px); box-shadow: 0 12px 26px var(--shadow);}}
.stat {{padding: 16px 18px; display: flex; gap: 14px; align-items: center;}}
.stat .icon {{
    width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center;
    background: var(--pill-bg); font-size: 1.25rem; flex-shrink: 0;
}}
.stat .label {{font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; font-weight: 600;}}
.stat .value {{font-size: 1.35rem; font-weight: 800; color: var(--text); line-height: 1.2;}}
.feature {{padding: 20px;}}
.feature .icon {{font-size: 1.6rem;}}
.feature b {{display: block; margin: 8px 0 4px; color: var(--text);}}
.feature span {{color: var(--muted); font-size: .9rem;}}

.pill {{
    display: inline-block; background: var(--pill-bg); color: var(--pill-text); border-radius: 999px;
    padding: 3px 12px; margin: 8px 6px 0 0; font-size: .8rem; font-weight: 600;
}}
.chunk-text {{
    color: var(--text); background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 10px 14px; font-size: .9rem; line-height: 1.55; margin-bottom: 10px;
}}
.legend {{color: var(--muted); font-size: .82rem; margin: -4px 0 6px;}}
.legend i {{display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin: 0 4px 0 10px;}}
.section-title {{font-weight: 700; color: var(--text); margin: 4px 0 2px; font-size: 1.02rem;}}
.section-sub {{color: var(--muted); font-size: .85rem; margin-bottom: 6px;}}

.st-key-summary_card {{
    background: var(--surface); border: 1px solid var(--border); border-left: 5px solid var(--accent);
    border-radius: 16px; padding: 18px 22px; box-shadow: 0 6px 18px var(--shadow);
}}
.st-key-chart_words, .st-key-chart_keywords {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
    padding: 16px 18px 6px; box-shadow: 0 6px 18px var(--shadow);
}}

/* ---------- widgets ---------- */
[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed var(--accent); border-radius: 18px; background: var(--surface);
}}
.stButton > button, .stDownloadButton > button {{border-radius: 12px; font-weight: 600;}}
.st-key-summarize button {{
    background: linear-gradient(120deg, #6C5CE7, #A06CF5); color: #fff; border: none;
    box-shadow: 0 8px 20px rgba(108, 92, 231, .30);
}}
.st-key-summarize button:hover {{filter: brightness(1.08); color: #fff; border: none;}}
.st-key-suggestions button {{
    border-radius: 999px; background: var(--surface); border: 1px solid var(--border);
    box-shadow: 0 3px 10px var(--shadow); font-weight: 500;
}}
.st-key-suggestions button:hover {{border-color: var(--accent); color: var(--accent);}}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{gap: 6px;}}
[data-testid="stTabs"] button[role="tab"] {{
    padding: 6px 16px; border-radius: 12px 12px 0 0; font-weight: 600;
}}

[data-testid="stChatMessage"] {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
    padding: 14px 16px; box-shadow: 0 4px 14px var(--shadow); margin-bottom: 10px;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{
    background: var(--user-bg); border-color: transparent;
    margin-left: auto; width: 85%; max-width: 85%;
}}
[data-testid="stChatInput"] {{border-radius: 16px; box-shadow: 0 6px 18px var(--shadow);}}
[data-testid="stExpander"] details {{border-radius: 14px; background: var(--surface2);}}
</style>
"""


# ---------- cached work ----------

@st.cache_resource(show_spinner="Loading the search model (first time only)...")
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per server process."""
    return rag.load_embedder(CFG)


@st.cache_data(show_spinner="Reading and indexing the PDF...")
def index_pdf(pdf_bytes: bytes) -> tuple[list[rag.Chunk], np.ndarray, list[rag.Page]]:
    """Extract, chunk and embed a PDF. Returns chunks, embeddings and pages."""
    pages = rag.extract_pages(pdf_bytes)
    chunks = rag.chunk_pages(pages, CFG)
    embeddings = rag.embed_chunks(chunks, get_embedder(), CFG)
    return chunks, embeddings, pages


# ---------- helpers ----------

def show_error(exc: Exception) -> None:
    """Display a friendly error message for known pipeline errors."""
    if isinstance(exc, rag.EmptyDocumentError):
        st.error(f"Empty PDF: {exc}", icon="📭")
    elif isinstance(exc, rag.MissingAPIKeyError):
        st.error(f"Missing API key: {exc}", icon="🔑")
    elif isinstance(exc, rag.LLMError):
        st.error(f"API failure: {exc}", icon="⚠️")
    else:
        st.error(str(exc))


def md_safe(text: str) -> str:
    """Stop Streamlit treating "$5 and $10" as a LaTeX formula."""
    return text.replace("$", "\\$")


def theme() -> dict[str, Any]:
    """The active theme (light or dark)."""
    return THEMES[st.session_state.get("theme_mode", "light")]


def word_count(pages: list[rag.Page]) -> int:
    """Total words in the document."""
    return sum(len(p.text.split()) for p in pages)


# ---------- theme & layout ----------

def apply_theme() -> None:
    """Dark mode switch in the sidebar; switches Streamlit's own theme and our CSS."""
    current = st_config.get_option("theme.base") or "light"
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="logo">📚</div>'
            "<div><b>ReadBot</b><small>Chat with any PDF</small></div></div>",
            unsafe_allow_html=True,
        )
        dark = st.toggle("🌙 Dark mode", value=current == "dark", key="dark_mode")
    mode = "dark" if dark else "light"
    st.session_state.theme_mode = mode
    if current != mode:
        for name, value in THEMES[mode]["streamlit"].items():
            st_config.set_option(f"theme.{name}", value)
        st.rerun()
    st.markdown(CSS.format(**THEMES[mode]["css"]), unsafe_allow_html=True)


def check_api_key() -> bool:
    """Show the Groq key status in the sidebar."""
    try:
        rag.get_api_key()
        st.sidebar.success("Groq API key connected", icon="🔑")
        return True
    except rag.MissingAPIKeyError as exc:
        st.sidebar.error(str(exc), icon="🔑")
        return False


def render_sidebar() -> None:
    """How-to steps, active settings and the clear-chat button."""
    steps = ["Upload a PDF", "Ask a question or pick a suggestion", "Check the cited pages"]
    with st.sidebar:
        st.markdown("#### How it works")
        st.markdown(
            "".join(f'<div class="step"><span class="num">{i}</span><span>{s}</span></div>'
                    for i, s in enumerate(steps, 1)),
            unsafe_allow_html=True,
        )
        st.divider()
        with st.expander("⚙️ Settings"):
            st.caption(f"Model: `{CFG['llm']['model']}`")
            st.caption(f"Search model: `{CFG['embeddings']['model'].split('/')[-1]}`")
            st.caption(f"Text sent per question: `{CFG['retrieval']['max_context_chars']:,}` chars")
            st.caption("Change these in `config.yaml`.")
        if st.session_state.get("history"):
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.history = []
                st.rerun()


def render_hero() -> None:
    """Animated gradient header at the top of the page."""
    st.markdown(
        '<div class="hero"><div class="doc">📄</div><h1>Chat with your PDF</h1>'
        "<p>Upload a document and ask anything. Every answer comes only from your file, "
        "with page citations you can check.</p>"
        '<div class="badges"><span class="badge">⚡ Powered by Groq</span>'
        '<span class="badge">🌐 English &amp; Hindi</span>'
        '<span class="badge">🔎 Hybrid search</span></div></div>',
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Feature cards shown before a PDF is uploaded."""
    features = [
        ("🔎", "Smart search", "Finds passages by meaning and exact keywords, even if you word it differently."),
        ("📌", "Page citations", "Every answer tells you which pages it used, so you can check it."),
        ("💬", "Real conversation", "Ask follow-ups like “explain more” - ReadBot remembers the chat."),
        ("🛡️", "No guessing", "Says so when the answer isn't in the document instead of making it up."),
    ]
    cards = "".join(
        f'<div class="feature"><div class="icon">{icon}</div><b>{title}</b><span>{text}</span></div>'
        for icon, title, text in features
    )
    st.markdown(f'<div class="card-row">{cards}</div>', unsafe_allow_html=True)


def render_doc_header(pages: list[rag.Page], chunks: list[rag.Chunk]) -> None:
    """Stat cards for the loaded document."""
    words = word_count(pages)
    minutes = max(1, round(words / 220))
    cards = [("📑", "Pages", len(pages)), ("✍️", "Words", f"{words:,}"),
             ("⏱️", "Read time", f"{minutes} min"), ("🧩", "Chunks", len(chunks))]
    html_cards = "".join(
        f'<div class="stat"><div class="icon">{icon}</div>'
        f'<div><div class="label">{label}</div><div class="value">{value}</div></div></div>'
        for icon, label, value in cards
    )
    st.markdown(f'<div class="card-row">{html_cards}</div>', unsafe_allow_html=True)


# ---------- charts ----------

def style_chart(chart: alt.Chart, height: int) -> alt.Chart:
    """Apply theme colours: transparent background, quiet axes and grid."""
    c = theme()["chart"]
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=c["axis"], titleColor=c["axis"], gridColor=c["grid"],
                        domain=False, tickColor=c["grid"], labelFontSize=11, titleFontSize=11)
    )


def words_per_page_chart(pages: list[rag.Page]) -> alt.Chart:
    """Bar chart: how much text each page has."""
    df = pd.DataFrame({"Page": [p.number for p in pages], "Words": [len(p.text.split()) for p in pages]})
    bars = alt.Chart(df).mark_bar(
        color=theme()["chart"]["mark"], cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X("Page:O", axis=alt.Axis(labelAngle=0, labelOverlap="greedy")),
        y=alt.Y("Words:Q", title="Words"),
        tooltip=[alt.Tooltip("Page:O"), alt.Tooltip("Words:Q", format=",")],
    )
    return style_chart(bars, 220)


def keywords_chart(keywords: list[tuple[str, int]]) -> alt.Chart:
    """Horizontal bar chart of the most frequent words."""
    df = pd.DataFrame(keywords, columns=["Word", "Mentions"])
    bars = alt.Chart(df).mark_bar(
        color=theme()["chart"]["mark"], cornerRadiusTopRight=4, cornerRadiusBottomRight=4
    ).encode(
        x=alt.X("Mentions:Q", title="Mentions"),
        y=alt.Y("Word:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=140)),
        tooltip=["Word", "Mentions"],
    )
    return style_chart(bars, max(180, 28 * len(df)))


def relevance_chart(page_scores: dict[int, float], cited: list[int]) -> alt.Chart:
    """Bar chart: how well each page matches the question; cited pages are highlighted."""
    c = theme()["chart"]
    df = pd.DataFrame({
        "Page": list(page_scores),
        "Match": [max(0.0, s) for s in page_scores.values()],
        "Status": ["Cited in answer" if p in cited else "Other page" for p in page_scores],
    })
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Page:O", axis=alt.Axis(labelAngle=0, labelOverlap="greedy")),
        y=alt.Y("Match:Q", title="Match", axis=alt.Axis(format="%")),
        color=alt.Color("Status:N", legend=None, scale=alt.Scale(
            domain=["Cited in answer", "Other page"], range=[c["mark"], c["faded"]])),
        tooltip=[alt.Tooltip("Page:O"), alt.Tooltip("Match:Q", format=".0%"), "Status"],
    )
    return style_chart(bars, 160)


# ---------- chat ----------

def render_sources(result: dict[str, Any]) -> None:
    """Cited pages as pills, plus how the answer was found inside an expander."""
    sources = result.get("sources") or []
    if result.get("not_found") or not sources:
        return
    pages = result.get("cited_pages") or []
    if pages:
        pills = "".join(f'<span class="pill">📌 Page {p}</span>' for p in pages)
        st.markdown(pills, unsafe_allow_html=True)
    with st.expander("🔍 How this was found"):
        if result.get("page_scores") and len(result["page_scores"]) > 1:
            c = theme()["chart"]
            st.markdown(
                '<div class="section-title">Where in the document</div>'
                f'<div class="legend"><i style="background:{c["mark"]}"></i>cited in the answer'
                f'<i style="background:{c["faded"]}"></i>other pages</div>',
                unsafe_allow_html=True,
            )
            st.altair_chart(relevance_chart(result["page_scores"], pages), use_container_width=True)
        if result.get("queries"):
            st.markdown("**Searched for:** " + " · ".join(f"`{q}`" for q in result["queries"]))
        if result.get("context_pages"):
            st.caption("Pages read by the AI: " + ", ".join(map(str, result["context_pages"])))
        st.markdown('<div class="section-title">Best matching passages</div>', unsafe_allow_html=True)
        for chunk, score in sources:
            st.caption(f"Page {chunk.page} · {max(score, 0):.0%} match")
            st.markdown(f'<div class="chunk-text">{html.escape(chunk.text)}</div>', unsafe_allow_html=True)


def render_suggestions() -> None:
    """Clickable example questions shown before the first question."""
    st.markdown('<div class="section-sub">Not sure where to start? Try one of these:</div>',
                unsafe_allow_html=True)
    with st.container(key="suggestions"):
        cols = st.columns(len(SUGGESTIONS))
        for col, text in zip(cols, SUGGESTIONS):
            if col.button(text, use_container_width=True):
                st.session_state.pending_question = text.split(" ", 1)[1]
                st.rerun()


def render_qa(chunks: list[rag.Chunk], embeddings: np.ndarray) -> None:
    """Chat-style Q&A: past turns, suggestions, then a new question."""
    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    if not history:
        render_suggestions()
    for turn in history:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(md_safe(turn["question"]))
        with st.chat_message("assistant", avatar="📚"):
            st.markdown(md_safe(turn["answer"]))
            render_sources(turn)

    typed = st.chat_input("Ask anything about the document...")
    question = (typed or st.session_state.pop("pending_question", "") or "").strip()
    if not question:
        return
    with st.chat_message("user", avatar="🧑"):
        st.markdown(md_safe(question))
    with st.chat_message("assistant", avatar="📚"):
        try:
            with st.spinner("Searching the document and thinking..."):
                result: dict[str, Any] = rag.answer_question(
                    question, chunks, embeddings, get_embedder(), CFG, PROMPTS,
                    history=[(t["question"], t["answer"]) for t in history],
                )
        except rag.RAGError as exc:
            show_error(exc)
            return
    history.append({"question": question, **result})
    st.rerun()  # redraw so the new turn, charts and the sidebar "Clear chat" button all appear


# ---------- tabs ----------

def render_insights(pages: list[rag.Page]) -> None:
    """Charts about the document's content."""
    left, right = st.columns(2)
    with left, st.container(key="chart_words"):
        st.markdown('<div class="section-title">📊 Words per page</div>'
                    '<div class="section-sub">Where the document has the most text</div>',
                    unsafe_allow_html=True)
        st.altair_chart(words_per_page_chart(pages), use_container_width=True)
    with right, st.container(key="chart_keywords"):
        st.markdown('<div class="section-title">🏷️ Top keywords</div>'
                    '<div class="section-sub">Most frequent meaningful words</div>',
                    unsafe_allow_html=True)
        keywords = rag.top_keywords(pages, 10)
        if keywords:
            st.altair_chart(keywords_chart(keywords), use_container_width=True)
        else:
            st.caption("Not enough text to find keywords.")


def render_summary(chunks: list[rag.Chunk]) -> None:
    """Button that asks the LLM for a short document summary."""
    st.markdown('<div class="section-sub">Get a quick overview of the whole document in a few bullet '
                "points, with page numbers.</div>", unsafe_allow_html=True)
    label = "🔄 Regenerate summary" if st.session_state.get("summary") else "✨ Summarize document"
    if st.button(label, key="summarize"):
        try:
            with st.spinner("Reading the document and summarizing..."):
                st.session_state.summary = rag.summarize_document(chunks, CFG, PROMPTS)
        except rag.RAGError as exc:
            show_error(exc)
    if st.session_state.get("summary"):
        with st.container(key="summary_card"):
            st.markdown("**📝 Summary**")
            st.markdown(md_safe(st.session_state.summary))
        st.download_button("⬇️ Download summary", st.session_state.summary,
                           file_name="summary.md", mime="text/markdown")


def main() -> None:
    """Page layout and flow."""
    st.set_page_config(page_title=CFG["app"]["title"], page_icon="📚", layout="centered")
    apply_theme()
    check_api_key()
    render_sidebar()
    render_hero()

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded is None:
        render_empty_state()
        return

    pdf_bytes = uploaded.getvalue()
    if len(pdf_bytes) > CFG["app"]["max_upload_mb"] * 1024 * 1024:
        st.error(f"File is larger than {CFG['app']['max_upload_mb']} MB.")
        return

    doc_id = hashlib.sha256(pdf_bytes).hexdigest()
    if st.session_state.get("doc_id") != doc_id:
        st.session_state.doc_id, st.session_state.summary = doc_id, None
        st.session_state.history = []

    try:
        chunks, embeddings, pages = index_pdf(pdf_bytes)
    except rag.RAGError as exc:
        show_error(exc)
        return

    render_doc_header(pages, chunks)
    chat_tab, summary_tab, insights_tab = st.tabs(["💬 Chat", "📝 Summary", "📊 Insights"])
    with chat_tab:
        render_qa(chunks, embeddings)
    with summary_tab:
        render_summary(chunks)
    with insights_tab:
        render_insights(pages)


if __name__ == "__main__":
    main()
