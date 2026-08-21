# BioAssess AI: PYQ-Centric Intelligent System for Long and Short Answer Evaluation in NCERT Biology

An NCERT-grounded Biology learning and formative-assessment app for CBSE Class 12 students, built with Streamlit, LangChain (RAG), ChromaDB, and Groq/Llama 3.1.

Instead of generic AI answers, BioAssist keeps every explanation, question, and evaluation tied back to the actual NCERT Class 12 Biology textbook — and it clearly separates real Previous Year Questions (PYQs) from AI-generated practice questions, so students always know which is which.

Live app: https://bioassessai.streamlit.app/

---

## What it does

**Practice** — Pick a chapter and a question level (MCQ, 1–5 Mark). BioAssist pulls real CBSE PYQs for that chapter where available, or falls back to a clearly labelled AI-generated question grounded in the NCERT text. You answer, and it's graded on the spot.

**Formative evaluation** — Answers are scored with mark-aware depth expectations (a 2-mark answer isn't held to 5-mark standards), with structured feedback: what you got right, what's missing, key terminology you should have used, and a model answer grounded in NCERT content. MCQs are graded with normalized answer matching, so "B", "b)", and the full option text are all accepted as equivalent.

**RAG-grounded explanations** — When useful, BioAssist retrieves the most relevant chunks of the NCERT chapter via ChromaDB before generating an explanation, rather than answering from the LLM's generic knowledge. Retrieved evidence is shown alongside the explanation.

**Progress tracking** — A lightweight history of attempted questions, scores, and chapter-level activity, stored locally in SQLite.

---

## Tech stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| RAG orchestration | LangChain |
| Vector store | ChromaDB (one store per chapter) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace), bundled locally in the repo so the app never depends on a live connection to huggingface.co |
| LLM | Llama 3.1 8B Instant via the Groq API |
| PDF processing | PyMuPDF |
| PYQ storage | Structured JSON question bank, built from CBSE previous-year papers |
| Progress storage | SQLite |

---

## How it's organized

```
src/
  main.py                    Streamlit UI — navigation, session state, orchestration
  rag_service.py              NCERT RAG: embeddings, chapter retrieval, LLM calls
  question_generator.py       AI-generated practice/case-study questions
  answer_evaluator.py         Mark-aware descriptive + MCQ evaluation
  pyq_mapper.py                Loads/filters/groups real PYQs by chapter and marks
  pyq_question_bank_builder.py Converts raw PYQ PDFs into structured JSON records
  progress_tracker.py          SQLite progress history + mastery breakdown
  vectorize_book.py            One-time script: NCERT PDFs → chapter-wise ChromaDB
  chatbot_utility.py           Reads available chapters from the NCERT PDF folder

data/
  class_12/biology/            NCERT chapter PDFs (source of truth for chapter list)
  pyqs/class_12/biology/       Raw CBSE previous-year question papers
  model_answers/               Reference answers used for evaluation grounding

chapters_vector_db/            Chroma vector store, one collection per chapter
pyq_vector_db/, model_answer_vector_db/   Supporting Chroma stores
models/all-MiniLM-L6-v2/       Bundled embedding model (no network call needed at runtime)
pyq_questions.json             Structured PYQ question bank
scripts/download_embedding_model.py   One-time local setup script (see below)
```

Scope note: the app currently supports **Class 12 Biology only**. A Capstone 2 feature set (Case Study mode, Critical Thinking questions, a Recall-vs-Critical-Thinking progress breakdown) is fully implemented in the code but hidden behind a `SHOW_CAPSTONE_2_FEATURES` flag at the top of `main.py` for the current demo scope.

---

## Running it locally

**1. Clone and install**
```bash
git clone <this-repo-url>
cd project-1-studymate
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Set up your Groq API key**

Copy `src/env_template` to `src/.env` and fill in your own key:
```
GROQ_API_KEY=your_groq_api_key
CLASS_SUBJECT_NAME=class_12/biology
DEVICE=cpu
```
Get a free key at [console.groq.com](https://console.groq.com). Never commit `src/.env` — it's already git-ignored.

**3. (One-time) bundle the embedding model**

The repo already ships a local copy of the embedding model under `models/all-MiniLM-L6-v2/`, so this step is usually not needed. If that folder is missing or you want to refresh it:
```bash
pip install sentence-transformers
python scripts/download_embedding_model.py
```

**4. Run the app**
```bash
streamlit run src/main.py
```

Open `http://localhost:8501`.

---

## Deployment

The live app runs on Streamlit Community Cloud, auto-deploying from the `main` branch. For a self-hosted EC2 setup instead, see `EC2_DEPLOYMENT.md`.

Required secret in your deployment environment: `GROQ_API_KEY`. On Streamlit Cloud this goes in **App settings → Secrets** as a top-level key (no section header), not as an environment variable file.

---

## Notes for anyone extending this

- The NCERT RAG flow (`rag_service.py`) and the PYQ flow (`pyq_mapper.py` / `pyq_question_bank_builder.py`) are intentionally separate pipelines — don't merge them. Real PYQs must never be relabelled as AI-generated, or vice versa.
- Rebuilding `chapters_vector_db/` or `pyq_questions.json` is disruptive and should be a deliberate, announced step, not a side effect of an unrelated change.
- Class/chapter changes reset only the state that depends on that context (current question, current evaluation) — not unrelated session state.
