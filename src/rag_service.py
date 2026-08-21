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

# If a local copy of the model has been bundled into the repo (see
# scripts/download_embedding_model.py), use that folder directly --
# this is what actually removes the huggingface.co network
# dependency, rather than just retrying it. Falls back to the
# hub model id below if the bundle isn't present, so local dev
# still works before anyone has run the download step.
LOCAL_EMBEDDING_MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "all-MiniLM-L6-v2"
)

HUB_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings():

    _start = time.time()

    from langchain_huggingface import HuggingFaceEmbeddings

    if os.path.isdir(LOCAL_EMBEDDING_MODEL_DIR) and os.listdir(LOCAL_EMBEDDING_MODEL_DIR):

        print(
            "Loading embeddings from bundled local copy "
            f"({LOCAL_EMBEDDING_MODEL_DIR}) -- no network required."
        )

        result = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL_DIR,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        _log_timing("get_embeddings() model load (first call only, local bundle)", time.time() - _start)

        return result

    # No bundled copy found -- fall back to loading from the
    # HuggingFace Hub id. HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are
    # set at the top of this module (before this import ever runs
    # anywhere in the process) to skip the network "check for
    # updates" call that was adding ~90s to loading an already-cached
    # local model. If the model somehow isn't cached yet, this
    # raises and the except-block below retries with network access
    # allowed.
    try:

        result = HuggingFaceEmbeddings(
            model_name=HUB_EMBEDDING_MODEL_NAME,
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

        # A fresh Streamlit Cloud container has no HF cache, so this
        # online retry is the ONLY path on first load there -- if
        # huggingface.co has a transient blip at that exact moment
        # (seen live: "couldn't connect to huggingface.co ... and
        # couldn't find them in cached files"), the whole app goes
        # down with it. Retrying a few times with backoff costs
        # nothing when the network is fine (succeeds on attempt 1)
        # and meaningfully improves odds of surviving a one-off
        # blip during something like a live demo.
        last_network_error = None
        result = None

        for attempt in range(1, 4):

            try:
                result = HuggingFaceEmbeddings(
                    model_name=HUB_EMBEDDING_MODEL_NAME,
                    model_kwargs={
                        "device": "cpu"
                    },
                    encode_kwargs={
                        "normalize_embeddings": True
                    }
                )
                break

            except Exception as network_error:

                last_network_error = network_error

                print(
                    f"Embeddings download attempt {attempt}/3 failed: "
                    f"{network_error}"
                )

                if attempt < 3:
                    time.sleep(attempt * 3)

        if result is None:
            raise last_network_error

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
        # llama-3.1-8b-instant was shut down on schedule (2026-08-16,
        # confirmed via the "model_not_found" 404 this app started
        # throwing) and no longer appears on console.groq.com/docs/models
        # at all -- it's not even in the Deprecated table anymore, just
        # gone. Current Production models (checked 2026-08-20) are only
        # openai/gpt-oss-120b and openai/gpt-oss-20b; qwen/qwen3.6-27b
        # is Preview-only, not Production, so it's a worse fit for a
        # capstone app that needs to keep working through submission.
        #
        # Using openai/gpt-oss-20b with reasoning_effort="low" this
        # time (not left at the default) -- that's what avoids the
        # earlier problem noted below: at default reasoning effort,
        # gpt-oss-20b spends real time and tokens on a hidden internal
        # "thinking" pass before every answer, which is a bad fit for
        # a short structured-output grading task that needs to be fast
        # every time. "low" keeps that thinking pass minimal.
        model="openai/gpt-oss-20b",
        reasoning_effort="low",
        temperature=0.2,
        # Was 700, tuned for llama-3.1-8b-instant (a plain
        # non-reasoning model, so every token went straight to the
        # visible answer). gpt-oss-20b is a reasoning model: even at
        # reasoning_effort="low" it spends some of this same
        # max_tokens budget on its hidden "thinking" pass before
        # writing the JSON. With only 700 tokens total, that left
        # too little room to finish the LAST field in the JSON
        # schema -- which is "model_answer" in evaluate_answer() and
        # "answer" in question_generator.py -- so it was coming back
        # empty ("Suggested NCERT-Aligned Answer: Not available.")
        # even though everything earlier in the JSON (score,
        # correct_points, missing_points, etc.) was fine. Groq's own
        # docs flag this exact tradeoff for reasoning models. Raised
        # to 1536 for headroom; still well under a second of
        # generation time at this model's ~1000 tokens/sec.
        max_tokens=1536,
        groq_api_key=api_key,
        # Bounds worst-case latency: one attempt, 20s to fail, one
        # retry, 20s to fail again -- under a minute before
        # evaluate_answer()'s existing except-block returns a
        # clean error instead of hanging indefinitely (there was
        # no timeout configured before this at all).
        timeout=20,
        max_retries=1
    )


@lru_cache(maxsize=1)
def get_fallback_llm():
    """
    Second, independent Groq production model. Only used if the
    primary model in get_llm() fails for any reason at call time --
    most importantly, if Groq deprecates/renames it again like they
    did to llama-3.1-8b-instant on 2026-08-16 with no advance code
    change on our side. openai/gpt-oss-120b is the other current
    Production-tier model on Groq (console.groq.com/docs/models),
    deliberately different from the primary so a single bad model ID
    can't take down both.
    """

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
        model="openai/gpt-oss-120b",
        reasoning_effort="low",
        temperature=0.2,
        # Kept in sync with get_llm()'s max_tokens -- see the
        # comment there for why 700 wasn't enough for a reasoning
        # model's hidden thinking pass plus a full JSON response.
        max_tokens=1536,
        groq_api_key=api_key,
        timeout=20,
        max_retries=1
    )


def invoke_llm(prompt):
    """
    Single entry point for every LLM call in the app -- use this
    instead of calling get_llm().invoke(...) directly.

    Tries the primary model (get_llm()) first. If that call raises
    ANY exception -- the model being deprecated/renamed by Groq, a
    rate limit, a transient network error -- it automatically retries
    once against a second, independent model (get_fallback_llm())
    before giving up. This is the direct fix for what happened on
    2026-08-16: Groq deprecated llama-3.1-8b-instant with no warning
    and every LLM call in the app broke at once. With this wrapper,
    the app quietly falls back to a working model instead of failing
    outright the next time a model gets deprecated mid-project.

    Callers keep their existing try/except around this call for
    display purposes (a friendly "couldn't grade this right now"
    message) -- this function only adds the extra attempt in between,
    it doesn't change what gets raised if BOTH models fail.
    """

    try:

        return get_llm().invoke(
            prompt
        )

    except Exception as primary_error:

        print(
            "[BioAssist] Primary LLM (openai/gpt-oss-20b) call failed, "
            "retrying once on fallback model (openai/gpt-oss-120b):",
            primary_error
        )

        try:

            return get_fallback_llm().invoke(
                prompt
            )

        except Exception as fallback_error:

            print(
                "[BioAssist] Fallback LLM (openai/gpt-oss-120b) also "
                "failed:",
                fallback_error
            )

            raise fallback_error from primary_error


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

        response = invoke_llm(
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