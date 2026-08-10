import os
from functools import lru_cache

from dotenv import load_dotenv

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

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


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
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=700,
        groq_api_key=api_key
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

    return Chroma(
        persist_directory=db_path,
        embedding_function=get_embeddings()
    )


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

        return db.similarity_search(
            question,
            k=k
        )

    except Exception as error:

        print(
            "NCERT retrieval error:",
            error
        )

        return []


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
    chapter
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

RULES:

1. Answer only from the retrieved NCERT content.
2. Stay within the selected class and chapter.
3. Explain in simple student-friendly language.
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