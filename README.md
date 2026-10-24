# ONDC RAG: Baseline Retrieval vs Re-Ranking

Two-stage RAG over the ONDC eB2B API Contract (v1.2.5), built so buyer-app and
seller-app teams can ask whether a feature is allowed by ONDC protocol instead
of reading the full spec. Idea came from internship work at Qwipo, where every
new feature for the retailer buyer app and distributor seller app had to be
checked against protocol before launch.

## Dataset

`data/ONDC_eB2B_Contract.pdf` — 99 pages, 97 with text, ~259K characters.
Chunking (frozen, reused by both pipelines): 1000 chars, 150 overlap, skip
under 200 chars. Result: 328 chunks with id and page number.

## Pipelines (same corpus, identical prompt, only chunks differ)

- **Pipeline 1 (baseline):** `all-MiniLM-L6-v2` Bi-Encoder to 384-dim vectors,
  L2-normalized, FAISS `IndexFlatIP` (cosine). Top 10 per query, first 3 into
  the fixed prompt for Answer 1.
- **Pipeline 2 (re-rank):** same Top 10 scored pairwise by
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, re-sorted, new Top 3 into the
  identical prompt for Answer 2. Strict rule: answer only from context, else
  say not found.

Top-10 cosine scores sit at 0.29-0.40 (JSON-schema heavy text), so re-ranking
matters more here than on prose documents.

## Failure queries (buyer/seller feature-allowedness)

| Query (type) | Shift | Answer 1 | Answer 2 |
|---|---|---|---|
| Misc fees at fulfillment + title customization (keyword trap) | p82_283: 4 to 1 | Incomplete (title part not found) | Complete: misc convenience fees, title by BNP |
| New retailer ordering before mapping (exception) | p3_9: 4 to 2 | Bare yes | Grounded with Seller-NP picks distributor, creation pending, P1 |
| Error codes on lapse/decline (buried) | 10 to 2, 9 to 3 | 50015/50016 correct | Same codes, rank-restored |
| Provider-id omission (negation) | 6 to 1, 7 to 3 | 30000 NACK | 30000 NACK |
| Distributor must-buy rule | 2 to 1 | Correct rule | Same rule, promoted |
| Feature codes 030/031 | 4 to 1, 6 to 2 | Complete | Concise, correct |

Honest limitations kept in notes: one tax question went partial to Not found,
one phrasing scored Not found both ways. Re-ranking promotes keyword-dense
chunks, which usually helps but can narrow answers.

## Latency (Colab T4, retrieval mean of 5)

Baseline Top 10: 0.0075 s. Re-rank extra: 0.0437 s (about 6x). Retrieval total
with re-rank: about 0.051 s. Pipeline 1 with LLM: 0.951 s. Pipeline 2 with
LLM: 0.732 s (LLM variance across runs; retrieval overhead is small, LLM
dominates). Same pattern as the NITI run.

## Scaling

Cross-Encoder needs a full forward pass per pair with nothing precomputable:
10 pairs cost ~44 ms here, so all 328 chunks would cost ~1.4 s per query.
Bi-Encoder vectors are precomputed with fast FAISS search. Prefilter to 10,
then re-rank: speed plus accuracy.

## Run

Colab, T4 GPU, Internet ON. Upload the PDF, add a Gemini key as a secret,
install `requirements.txt`, Run All top to bottom.

## Files

- `Rag_ondc.ipynb` — full notebook (setup, chunking, both pipelines, failure
  analysis, latency).
- `rag_ondc.py` — same flow as a plain script.
- `ondc_rag_report.pdf` — final report.
- `REPORT_ONDC.txt` — report text with all metrics.
- `ONDC_interview_notes.docx` — decision log with reasons, quota lessons,
  interview Q&A.
