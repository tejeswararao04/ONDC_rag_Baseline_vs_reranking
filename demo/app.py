"""ONDC eB2B protocol checker: baseline retrieval vs Cross-Encoder re-ranking.

Ask whether a feature is allowed for buyer apps (BNP) or seller apps (SNP).
Each question runs both pipelines side by side with the identical prompt.
Needs GEMINI_API_KEY in Streamlit secrets. Offline artifacts chunks.pkl and
index.faiss were built once from ONDC_eB2B_Contract.pdf (328 chunks).
"""
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

# files sit next to this script, so resolve against it, not the cwd
BASE = Path(__file__).parent

# fixed prompt used by both pipelines, only chunks differ
PROMPT_HEAD = (
    "You are a helper for the ONDC eB2B protocol document.\n"
    "Answer only from given context.\n"
    "If answer is not in context, say not found.\n"
)

SAMPLES = [
    "Can a buyer app add convenience fees to an order",
    "Can a new retailer place an order before distributor mapping is complete",
    "What happens if provider id is omitted without descriptor type new",
]


@st.cache_resource
def load_all():
    # chunks, index and both models load once per session
    with open(BASE / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(str(BASE / "index.faiss"))
    bi = SentenceTransformer("all-MiniLM-L6-v2")
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return chunks, index, bi, cross


def search_top10(query, chunks, index, bi):
    # baseline: embed query, cosine search, return Top 10
    q = bi.encode([query], convert_to_numpy=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    scores, ids = index.search(q.astype("float32"), 10)
    return [
        {
            "rank": r + 1,
            "id": chunks[int(ids[0][r])]["id"],
            "page": chunks[int(ids[0][r])]["page"],
            "score": float(scores[0][r]),
            "text": chunks[int(ids[0][r])]["text"],
        }
        for r in range(10)
    ]


def rerank_top10(query, hits, cross):
    # re-rank: score each (query, chunk) pair jointly, sort descending
    pairs = [[query, h["text"]] for h in hits]
    scores = cross.predict(pairs)
    for i, h in enumerate(hits):
        h["cross_score"] = float(scores[i])
    return sorted(hits, key=lambda x: x["cross_score"], reverse=True)


def make_prompt(query, top3):
    context = "".join(
        f"\nChunk {i + 1}:\n{top3[i]['text']}\n" for i in range(3)
    )
    return f"{PROMPT_HEAD}\nContext:\n{context}\nQuestion: {query}\nAnswer:"


def answer(query, top3):
    # Gemini key comes from Streamlit secrets, never from code
    import google.generativeai as genai

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    llm = genai.GenerativeModel("models/gemini-flash-lite-latest")
    return llm.generate_content(make_prompt(query, top3)).text


st.set_page_config(page_title="ONDC Protocol Checker", layout="wide")
st.title("ONDC eB2B Protocol Checker")
st.caption(
    "Baseline dense retrieval vs Cross-Encoder re-ranking over the ONDC eB2B "
    "API contract (328 chunks). Same prompt, only the chunks differ."
)

chunks, index, bi, cross = load_all()

for s in SAMPLES:
    if st.button(s, key=s):
        st.session_state["q"] = s

query = st.text_input(
    "Ask if a buyer-app or seller-app feature is protocol-allowed:",
    value=st.session_state.get("q", SAMPLES[0]),
)

if st.button("Check") and query.strip():
    t0 = time.time()
    hits = search_top10(query, chunks, index, bi)
    top3_base = hits[:3]
    t1 = time.time()
    ranked = rerank_top10(query, hits, cross)
    top3_new = ranked[:3]
    t2 = time.time()
    try:
        ans1 = answer(query, top3_base)
    except Exception as e:  # quota or key errors stay visible, not silent
        ans1 = f"LLM call failed: {e}"
    try:
        ans2 = answer(query, top3_new)
    except Exception as e:
        ans2 = f"LLM call failed: {e}"
    t3 = time.time()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Answer 1: baseline Top 3")
        st.write(ans1)
        st.caption(f"Retrieval: {t1 - t0:.3f}s")
    with col2:
        st.subheader("Answer 2: re-ranked Top 3")
        st.write(ans2)
        st.caption(f"Re-rank extra: {t2 - t1:.3f}s, total: {t3 - t0:.3f}s")

    st.subheader("Rank shifts (old to new)")
    st.write(
        " | ".join(
            f"{c['id']} (p.{c['page']}): "
            f"#{next(h['rank'] for h in hits if h['id'] == c['id'])} "
            f"to #{i + 1}"
            for i, c in enumerate(top3_new)
        )
    )
    with st.expander("Baseline Top 3 chunks"):
        for h in top3_base:
            st.markdown(f"**{h['id']}**, page {h['page']}")
            st.text(h["text"][:800])
    with st.expander("Re-ranked Top 3 chunks"):
        for h in top3_new:
            st.markdown(f"**{h['id']}**, page {h['page']}")
            st.text(h["text"][:800])
