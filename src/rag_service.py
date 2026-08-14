import os
import time
from functools import lru_cache

# Must be set before huggingface_hub is ever imported anywhere in
# the process (even transitively) -- it reads HF_HUB_OFFLINE into
# a fixed constant at import time, so setting it later (e.g. inside
# get_embeddings(), right before instantiating HuggingFaceEmbeddings)
# is too late to have any effect. Skips the network "check for
# updates" call that was adding ~90s to loading an already-cached
# local model. get_embeddings() below removes this if the model
# turns out not to be cached yet, so a genuinely first-ever run
# still works (just slower, while it downloads).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv


def _log_timing(label, seconds):
    # Temporary diagnostic logging -- prints to the terminal
    # running `streamlit run`, not the browser. Evaluation is
    # taking 1+ minutes with no errors even after switching to a
    # fast non-reasoning model, which doesn't match normal LLM or
    # retrieval latency. These prints pin down exactly which step
    # (embedding-model load, Chroma retrieval, or the Groq call
    # itself) is actually consuming the time, instead of guessing
    # further. Safe to remove once the real bottleneck is found.
    print(f"[BioAssist timing] {label}: {seconds:.2f}s")

# NOTE: langchain_chroma / langchain_huggingface / langchain_groq
# are imported lazily, inside the functions that need them
# (get_embeddings, get_llm, load_chapter_db) rather than here
# at module level. Those packages pull in torch/transformers,
# which is the slowest part of starting the Streamlit app.
# Importing them lazily means the app UI renders immediately,
# and that cost is only paid the first time a Learn / Generate /
# Evaluate action actually needs a model.


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =========================================================
# LOAD .ENV FROM src/.env
# =========================================================

ENV_PATH = os.path.join(
    PROJECT_DIR,
    "src",
    ".env"
)

load_dotenv(
    dotenv_path=ENV_PATH
)


# =========================================================
# EMBEDDINGS
# =========================================================

@lru_cache(maxsize=1)
def get_embeddings():

    _start = time.time()

    from langchain_huggingface import HuggingFaceEmbeddings

    # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set at the top of
    # this module (before this import ever runs anywhere in the
    # process) to skip the network "check for updates" call that
    # was adding ~90s to loading an already-cached local model.
    # If the model somehow isn't cached yet, this raises and the
    # except-block below retries with network access allowed.
    try:

        result = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    except Exception as error:

        print(
            "Offline-mode embeddings load failed "
            f"(model may not be cached locally yet): {error}\n"
            "Retrying with network access allowed -- this one "
            "load may be slow while it downloads."
        )

        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        result = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    # lru_cache means this body only runs once per process -- if
    # this print appears on every evaluation (not just the first),
    # caching is not behaving as expected.
    _log_timing("get_embeddings() model load (first call only)", time.time() - _start)

    return result


# =========================================================
# GROQ LLM
# =========================================================

@lru_cache(maxsize=1)
def get_llm():

    from langchain_groq import ChatGroq

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY not found.\n"
            f"Expected .env at:\n{ENV_PATH}"
        )

    return ChatGroq(
        # Switched to openai/gpt-oss-20b earlier (Groq's suggested
        # replacement ahead of llama-3.1-8b-instant's 2026-08-16
        # deprecation), but that made things WORSE: gpt-oss-20b is
        # a reasoning model -- by default it generates a hidden
        # internal "thinking" pass before every answer, and that
        # reasoning generation consumes real time and eats into
        # the token budget on every single call, regardless of how
        # short the final JSON is (console.groq.com/docs/reasoning).
        # That's a bad fit for a short structured-output grading
        # task that needs to be fast every time.
        #
        # llama-3.1-8b-instant is still fully live today (checked
        # console.groq.com/docs/models directly -- listed as a
        # Production model at full speed, not yet actually shut
        # down) and is a plain non-reasoning "instant" model, which
        # is what this app actually needs. Using it until the
        # 2026-08-16 shutdown; revisit before then (candidates at
        # that point: gpt-oss-20b with reasoning_effort="low" to
        # cut the reasoning overhead, or qwen/qwen3.6-27b with
        # reasoning_effort="none").
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=700,
        groq_api_key=api_key,
        # Bounds worst-case latency: one attempt, 20s to fail, one
        # retry, 20s to fail again -- under a minute before
        # evaluate_answer()'s existing except-block returns a
        # clean error instead of hanging indefinitely (there was
        # no timeout configured before this at all).
        timeout=20,
        max_retries=1
    )


# =========================================================
# CLASS FOLDER
# =========================================================

def get_class_folder(
    selected_class
):

    return (
        selected_class
        .lower()
        .replace(
            " ",
            "_"
        )
    )


# =========================================================
# LOAD CHAPTER DB
# =========================================================

@lru_cache(maxsize=64)
def load_chapter_db(
    selected_class,
    chapter
):

    _start = time.time()

    from langchain_chroma import Chroma

    class_folder = get_class_folder(
        selected_class
    )

    db_path = os.path.join(
        PROJECT_DIR,
        "chapters_vector_db",
        class_folder,
        "biology",
        chapter
    )

    if not os.path.exists(
        db_path
    ):

        print(
            "ChromaDB not found:",
            db_path
        )

        return None

    result = Chroma(
        persist_directory=db_path,
        embedding_function=get_embeddings()
    )

    # Cached per (class, chapter) -- should only print once per
    # chapter per process, not on every evaluation for that chapter.
    _log_timing(f"load_chapter_db({chapter}) (first call only)", time.time() - _start)

    return result


# =========================================================
# RETRIEVE NCERT
# =========================================================

def retrieve_ncert(
    question,
    selected_class,
    chapter,
    k=3
):

    db = load_chapter_db(
        selected_class,
        chapter
    )

    if db is None:
        return []

    try:

        _start = time.time()

        result = db.similarity_search(
            question,
            k=k
        )

        # Runs on EVERY evaluation/question-generation call, not
        # just the first -- this is the one to watch if the other
        # two only print once.
        _log_timing("retrieve_ncert() similarity_search (every call)", time.time() - _start)

        return result

    except Exception as error:

        print(
            "NCERT retrieval error:",
            error
        )

        return []


# =========================================================
# EXPLANATION LEVEL
#
# The retrieved NCERT context always comes from the actual
# Class 12 chapter (source of truth never changes). What
# changes here is how simply the LLM is asked to explain
# that same content, based on the student's demonstrated
# capacity (see progress_tracker.suggest_explanation_level).
# =========================================================

EXPLANATION_LEVEL_INSTRUCTIONS = {

    "Class 8": """
Explain this at a Class 8 level.
Use simple, everyday language and short sentences.
Avoid heavy biological/technical terminology. If a technical
term is essential, briefly explain it in plain words the
first time it is used.
Focus on the core idea rather than exam-level depth.
""",

    "Class 10": """
Explain this at a Class 10 level.
Use moderately simple language. Standard biology terms can
be used, but briefly explain each one.
Give a clear, structured explanation without the full
Class 12 exam-level technical depth.
""",

    "Class 12": """
Explain this at full Class 12 NCERT depth.
Use accurate Class 12 biological terminology, structured for
exam preparation.
"""
}


def get_explanation_level_instruction(explanation_level):

    return EXPLANATION_LEVEL_INSTRUCTIONS.get(
        explanation_level,
        EXPLANATION_LEVEL_INSTRUCTIONS["Class 12"]
    )


# =========================================================
# BUILD COMPACT CONTEXT
# =========================================================

def build_context(
    docs,
    max_chars_per_chunk=1200
):

    if not docs:
        return ""

    return "\n\n".join(
        doc.page_content[
            :max_chars_per_chunk
        ]
        for doc in docs
    )


# =========================================================
# ANSWER QUESTION
# =========================================================

def answer_question(
    question,
    selected_class,
    chapter,
    explanation_level="Class 12"
):

    docs = retrieve_ncert(
        question,
        selected_class,
        chapter,
        k=3
    )

    if not docs:

        return (
            "No indexed NCERT content was found for this chapter. "
            "Please run `python src/vectorize_book.py` first.",
            []
        )

    context = build_context(
        docs,
        max_chars_per_chunk=1200
    )

    level_instruction = get_explanation_level_instruction(
        explanation_level
    )

    prompt = f"""
You are BioAssist AI, an NCERT Biology learning assistant.

CLASS:
{selected_class}

CHAPTER:
{chapter}

Use ONLY the retrieved NCERT content below.

NCERT CONTEXT:
{context}

STUDENT QUESTION:
{question}

EXPLANATION LEVEL:
{explanation_level}

{level_instruction}

RULES:

1. Answer only from the retrieved NCERT content.
2. Stay within the selected class and chapter.
3. Explain in simple student-friendly language, matched to
   the EXPLANATION LEVEL above.
4. Keep the answer concise but complete.
5. Do not invent information.
6. Do not use outside knowledge.
7. If the retrieved context is insufficient, say:

"The retrieved NCERT evidence is insufficient to answer this question."
"""

    try:

        response = get_llm().invoke(
            prompt
        )

        answer = response.content.strip()

        return answer, docs

    except Exception as error:

        print(
            "Groq error:",
            error
        )

        return (
            "Unable to generate the answer right now.\n\n"
            f"Error: {error}",
            docs
        )