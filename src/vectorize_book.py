import os
import glob
import re
import shutil

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

CHAPTER_DB_DIR = os.path.join(
    PROJECT_DIR,
    "chapters_vector_db"
)

PYQ_DB_DIR = os.path.join(
    PROJECT_DIR,
    "pyq_vector_db"
)

MODEL_ANSWER_DB_DIR = os.path.join(
    PROJECT_DIR,
    "model_answer_vector_db"
)


# =========================================================
# EMBEDDINGS
# =========================================================

def get_embeddings():
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
# PDF LOADER
# =========================================================

def load_pdf(pdf_path):
    return PyMuPDFLoader(
        pdf_path
    ).load()


# =========================================================
# NCERT SPLITTER
# =========================================================

def split_ncert_documents(documents):
    """
    NCERT is still chunked because chapter text is long.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120
    )

    return splitter.split_documents(
        documents
    )


# =========================================================
# GENERAL CLEANING
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# PYQ CLEANING
# =========================================================

BAD_SYMBOLS = [
    "§", "¡", "©", "ñ", "Ê", "Ý",
    "¿", "ø", "å", "à", "ì", "ò",
    "¤", "¦", "¨", "µ", "¢", "¥"
]


COMMON_ENGLISH_WORDS = {
    "the", "and", "of", "to", "in", "is", "are",
    "was", "were", "which", "what", "how", "why",
    "where", "when", "with", "for", "from", "on",
    "a", "an", "this", "that", "these", "those",
    "identify", "explain", "describe", "mention",
    "state", "name", "given", "following",
    "correct", "incorrect", "option", "choose",
    "biology", "cell", "cells", "plant", "plants",
    "human", "humans", "organism", "organisms",
    "gene", "genes", "dna", "rna", "chromosome",
    "chromosomes", "hormone", "hormones",
    "reproduction", "fertilisation", "fertilization",
    "gamete", "gametes", "ovum", "sperm",
    "zygote", "embryo", "answer", "question",
    "figure", "diagram", "process", "function",
    "role", "during", "between", "type", "called"
}


def is_english_line(line):

    if not line:
        return False

    line = line.strip()

    if len(line) < 8:
        return False

    if any(
        symbol in line
        for symbol in BAD_SYMBOLS
    ):
        return False

    ascii_count = sum(
        1
        for char in line
        if ord(char) < 128
    )

    ascii_ratio = (
        ascii_count /
        max(len(line), 1)
    )

    if ascii_ratio < 0.92:
        return False

    words = re.findall(
        r"[A-Za-z]+",
        line.lower()
    )

    if len(words) < 3:
        return False

    english_hits = sum(
        1
        for word in words
        if word in COMMON_ENGLISH_WORDS
    )

    if english_hits < 2:
        return False

    return True


def clean_pyq_page(text):
    """
    Clean one complete CBSE PYQ page.

    Important:
    We DO NOT split the page afterwards.
    """

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    good_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if is_english_line(
            line
        ):
            good_lines.append(
                line
            )

    cleaned = " ".join(
        good_lines
    )

    # remove page labels
    cleaned = re.sub(
        r"\bPage\s+\d+\b",
        " ",
        cleaned,
        flags=re.IGNORECASE
    )

    # remove repeated whitespace
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    )

    return cleaned.strip()


def is_valid_pyq_page(text):

    if not text:
        return False

    text = text.strip()

    if len(text) < 100:
        return False

    if any(
        symbol in text
        for symbol in BAD_SYMBOLS
    ):
        return False

    words = re.findall(
        r"[A-Za-z]+",
        text.lower()
    )

    if len(words) < 15:
        return False

    return True


# =========================================================
# NCERT VECTORIZATION
# =========================================================

def vectorize_ncert(
    class_name,
    subject="biology"
):

    source_dir = os.path.join(
        DATA_DIR,
        class_name,
        subject
    )

    if not os.path.exists(
        source_dir
    ):
        print(
            "NCERT folder not found:",
            source_dir
        )
        return

    pdf_files = sorted(
        glob.glob(
            os.path.join(
                source_dir,
                "*.pdf"
            )
        )
    )

    if not pdf_files:
        print(
            "No NCERT PDFs found:",
            source_dir
        )
        return

    for pdf_path in pdf_files:

        chapter_name = os.path.splitext(
            os.path.basename(
                pdf_path
            )
        )[0]

        print(
            "Processing NCERT:",
            class_name,
            chapter_name
        )

        try:

            documents = load_pdf(
                pdf_path
            )

            for doc in documents:

                doc.page_content = clean_text(
                    doc.page_content
                )

            chunks = split_ncert_documents(
                documents
            )

            valid_chunks = []

            for chunk in chunks:

                text = clean_text(
                    chunk.page_content
                )

                if len(text) < 50:
                    continue

                chunk.page_content = text

                chunk.metadata.update({
                    "class": class_name,
                    "subject": subject,
                    "chapter": chapter_name,
                    "source_type": "NCERT",
                    "source_file": os.path.basename(
                        pdf_path
                    )
                })

                valid_chunks.append(
                    chunk
                )

            if not valid_chunks:
                continue

            persist_dir = os.path.join(
                CHAPTER_DB_DIR,
                class_name,
                subject,
                chapter_name
            )

            if os.path.exists(
                persist_dir
            ):
                shutil.rmtree(
                    persist_dir
                )

            os.makedirs(
                os.path.dirname(
                    persist_dir
                ),
                exist_ok=True
            )

            Chroma.from_documents(
                documents=valid_chunks,
                embedding=get_embeddings(),
                persist_directory=persist_dir
            )

            print(
                f"Indexed {len(valid_chunks)} NCERT chunks"
            )

        except Exception as e:

            print(
                "NCERT error:",
                e
            )


# =========================================================
# PYQ PAGE-LEVEL VECTORIZATION
# =========================================================

def vectorize_pyqs():

    source_root = os.path.join(
        DATA_DIR,
        "pyqs",
        "class_12",
        "biology"
    )

    if not os.path.exists(
        source_root
    ):
        print(
            "PYQ folder not found:",
            source_root
        )
        return

    pdf_files = sorted(
        glob.glob(
            os.path.join(
                source_root,
                "**",
                "*.pdf"
            ),
            recursive=True
        )
    )

    if not pdf_files:
        print(
            "No PYQ PDFs found:",
            source_root
        )
        return

    all_pages = []

    print(
        f"\nFound {len(pdf_files)} PYQ PDFs"
    )

    for pdf_path in pdf_files:

        relative_path = os.path.relpath(
            pdf_path,
            source_root
        )

        path_parts = relative_path.split(
            os.sep
        )

        year = ""

        if (
            path_parts
            and path_parts[0].isdigit()
        ):
            year = path_parts[0]

        print(
            "Processing PYQ:",
            year,
            os.path.basename(
                pdf_path
            )
        )

        try:

            pages = load_pdf(
                pdf_path
            )

            for page_number, page in enumerate(
                pages,
                start=1
            ):

                cleaned = clean_pyq_page(
                    page.page_content
                )

                if not is_valid_pyq_page(
                    cleaned
                ):
                    continue

                page.page_content = cleaned

                page.metadata.update({
                    "class": "class_12",
                    "subject": "biology",
                    "source_type": "PYQ",
                    "year": year,
                    "source_file": os.path.basename(
                        pdf_path
                    ),
                    "page_number": page_number
                })

                # IMPORTANT:
                # add the COMPLETE cleaned page.
                all_pages.append(
                    page
                )

        except Exception as e:

            print(
                "PYQ error:",
                e
            )

    if not all_pages:

        print(
            "No readable PYQ pages were generated."
        )
        return

    persist_dir = os.path.join(
        PYQ_DB_DIR,
        "class_12",
        "biology"
    )

    if os.path.exists(
        persist_dir
    ):
        shutil.rmtree(
            persist_dir
        )

    os.makedirs(
        os.path.dirname(
            persist_dir
        ),
        exist_ok=True
    )

    Chroma.from_documents(
        documents=all_pages,
        embedding=get_embeddings(),
        persist_directory=persist_dir
    )

    print(
        f"PYQ database created with "
        f"{len(all_pages)} complete pages"
    )


# =========================================================
# MODEL ANSWER VECTORIZATION
# =========================================================

def vectorize_model_answers():

    source_root = os.path.join(
        DATA_DIR,
        "model_answers",
        "class_12",
        "biology"
    )

    if not os.path.exists(
        source_root
    ):
        print(
            "MODEL_ANSWER folder not found:",
            source_root
        )
        return

    pdf_files = sorted(
        glob.glob(
            os.path.join(
                source_root,
                "**",
                "*.pdf"
            ),
            recursive=True
        )
    )

    if not pdf_files:
        print(
            "No MODEL_ANSWER PDFs found:",
            source_root
        )
        return

    all_pages = []

    for pdf_path in pdf_files:

        relative_path = os.path.relpath(
            pdf_path,
            source_root
        )

        path_parts = relative_path.split(
            os.sep
        )

        year = ""

        if (
            path_parts
            and path_parts[0].isdigit()
        ):
            year = path_parts[0]

        print(
            "Processing MODEL_ANSWER:",
            year,
            os.path.basename(
                pdf_path
            )
        )

        try:

            pages = load_pdf(
                pdf_path
            )

            for page_number, page in enumerate(
                pages,
                start=1
            ):

                cleaned = clean_pyq_page(
                    page.page_content
                )

                if not is_valid_pyq_page(
                    cleaned
                ):
                    continue

                page.page_content = cleaned

                page.metadata.update({
                    "class": "class_12",
                    "subject": "biology",
                    "source_type": "MODEL_ANSWER",
                    "year": year,
                    "source_file": os.path.basename(
                        pdf_path
                    ),
                    "page_number": page_number
                })

                all_pages.append(
                    page
                )

        except Exception as e:

            print(
                "MODEL_ANSWER error:",
                e
            )

    if not all_pages:

        print(
            "No readable MODEL_ANSWER pages generated."
        )
        return

    persist_dir = os.path.join(
        MODEL_ANSWER_DB_DIR,
        "class_12",
        "biology"
    )

    if os.path.exists(
        persist_dir
    ):
        shutil.rmtree(
            persist_dir
        )

    os.makedirs(
        os.path.dirname(
            persist_dir
        ),
        exist_ok=True
    )

    Chroma.from_documents(
        documents=all_pages,
        embedding=get_embeddings(),
        persist_directory=persist_dir
    )

    print(
        f"MODEL_ANSWER database created with "
        f"{len(all_pages)} pages"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "BIOASSIST VECTORIZATION"
    )

    print(
        "==============================\n"
    )

    vectorize_ncert(
        "class_12"
    )

    vectorize_pyqs()

    vectorize_model_answers()

    print(
        "\nALL VECTORIZATION COMPLETED"
    )