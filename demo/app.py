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


st.set_page_config(
    page_title="ONDC Protocol Checker", layout="wide", page_icon="✅"
)
st.markdown(
    """
    <style>
    /* green Check button (primary buttons only, samples stay grey) */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #16a34a;
        border-color: #16a34a;
        color: white;
        font-weight: 600;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #15803d;
        border-color: #15803d;
        color: white;
    }
    /* color badges for the two pipelines */
    .badge-base {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background-color: #fef3c7; color: #92400e;
        font-size: 13px; font-weight: 600;
    }
    .badge-rerank {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background-color: #dcfce7; color: #166534;
        font-size: 13px; font-weight: 600;
    }
    .shift-chip {
        display: inline-block; padding: 2px 10px; border-radius: 8px;
        background-color: #f1f5f9; color: #0f172a;
        font-size: 13px; margin: 2px 4px 2px 0;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("ONDC eB2B Protocol Checker")
st.caption(
    "Baseline dense retrieval vs Cross-Encoder re-ranking over the ONDC eB2B "
    "API contract (328 chunks). Same prompt, only the chunks differ."
)

chunks, index, bi, cross = load_all()

if "q" not in st.session_state:
    st.session_state["q"] = SAMPLES[0]
if "history" not in st.session_state:
    # every answered question stays on screen like a chat thread
    st.session_state["history"] = []


def run_check(query):
    # runs both pipelines once, returns everything the UI shows
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
    return {
        "query": query,
        "hits": hits,
        "ranked": ranked,
        "top3_base": top3_base,
        "top3_new": top3_new,
        "ans1": ans1,
        "ans2": ans2,
        "t_base": t1 - t0,
        "t_rerank": t2 - t1,
        "t_total": t3 - t0,
    }


st.write("Try a sample, or type your own question below:")
for s in SAMPLES:
    if st.button(s, key=s):
        # one click fills the box AND runs both pipelines
        st.session_state["q"] = s
        st.session_state["history"].append(run_check(s))

# textbox with the Check button right beside it, Enter also works
with st.form("ask", clear_on_submit=False):
    input_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
    with input_col:
        query = st.text_input(
            "Ask if a buyer-app or seller-app feature is protocol-allowed:",
            value=st.session_state["q"],
            label_visibility="collapsed",
            placeholder="Ask if a buyer-app or seller-app feature is protocol-allowed…",
        )
    with btn_col:
        go = st.form_submit_button("Check", type="primary")
    if go and query.strip():
        st.session_state["q"] = query
        st.session_state["history"].append(run_check(query))

for res in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(res["query"])
    with st.chat_message("assistant"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<span class="badge-base">PIPELINE 1 · BASELINE</span>',
                unsafe_allow_html=True,
            )
            st.write(res["ans1"])
            st.caption(f"Retrieval: {res['t_base']:.3f}s")
        with col2:
            st.markdown(
                '<span class="badge-rerank">PIPELINE 2 · RE-RANKED</span>',
                unsafe_allow_html=True,
            )
            st.write(res["ans2"])
            st.caption(
                f"Re-rank extra: {res['t_rerank']:.3f}s, "
                f"total: {res['t_total']:.3f}s"
            )
        st.markdown(
            "".join(
                f"<span class='shift-chip'>{c['id']} (p.{c['page']}): "
                f"#{next(h['rank'] for h in res['hits'] if h['id'] == c['id'])} "
                f"→ #{i + 1}</span>"
                for i, c in enumerate(res["top3_new"])
            ),
            unsafe_allow_html=True,
        )
        # one organized source view: union of both Top 3s with old/new ranks
        st.markdown("**Sources compared**")
        union_ids = [c["id"] for c in res["top3_new"]]
        for h in res["top3_base"]:
            if h["id"] not in union_ids:
                union_ids.append(h["id"])
        base_ids = [h["id"] for h in res["top3_base"]]
        new_ids = [h["id"] for h in res["top3_new"]]
        table = []
        for cid in union_ids:
            old = next(h["rank"] for h in res["hits"] if h["id"] == cid)
            new = next(
                i + 1 for i, h in enumerate(res["ranked"]) if h["id"] == cid
            )
            page = next(h["page"] for h in res["hits"] if h["id"] == cid)
            move = old - new
            arrow = f"↑ {move}" if move > 0 else (f"↓ {-move}" if move < 0 else "=")
            table.append(
                {
                    "Chunk": f"{cid} (p. {page})",
                    "Baseline": "in Top 3" if cid in base_ids else f"rank {old}",
                    "Re-ranked": "in Top 3" if cid in new_ids else f"rank {new}",
                    "Move": arrow,
                }
            )
        st.table(table)
        with st.expander("Chunk texts"):
            for cid in union_ids:
                h = next(x for x in res["hits"] if x["id"] == cid)
                tag = []
                if cid in base_ids:
                    tag.append("baseline")
                if cid in new_ids:
                    tag.append("re-ranked")
                st.markdown(f"**{cid}**, page {h['page']} ({', '.join(tag)})")
                st.text(h["text"][:800])
