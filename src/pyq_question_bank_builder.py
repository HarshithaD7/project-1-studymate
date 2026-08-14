import os
import glob
import json
import re
import sys

import numpy as np

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PYQ_SOURCE_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "pyqs",
    "class_12",
    "biology"
)

CHAPTER_DB_DIR = os.path.join(
    PROJECT_DIR,
    "chapters_vector_db",
    "class_12",
    "biology"
)

OUTPUT_JSON = os.path.join(
    PROJECT_DIR,
    "pyq_questions.json"
)


# =========================================================
# CLASS 12 BIOLOGY CHAPTERS
# =========================================================

VALID_CHAPTERS = [
    "Sexual Reproduction in Flowering Plants",
    "Human Reproduction",
    "Reproductive Health",
    "Principles of Inheritance and Variation",
    "Molecular Basis of Inheritance",
    "Evolution",
    "Human Health and Diseases",
    "Microbes in Human Welfare",
    "Biotechnology: Principles and Processes",
    "Biotechnology and its Applications",
    "Organisms and Populations",
    "Ecosystem",
    "Biodiversity and Conservation"
]


# =========================================================
# EMBEDDINGS
# =========================================================

_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        _embeddings = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    return _embeddings


# =========================================================
# BASIC HELPERS
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = text.replace(
        "\x00",
        " "
    )

    text = text.replace(
        "\u200b",
        ""
    )

    text = text.replace(
        "\ufeff",
        ""
    )

    text = text.replace(
        "￾",
        "-"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def normalize_text(value):

    value = str(
        value or ""
    ).lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def strip_number_prefix(value):

    value = str(
        value or ""
    ).strip()

    value = re.sub(
        r"^\s*\d+\s*[\.\)\-:]\s*",
        "",
        value
    )

    return value.strip()


# =========================================================
# LOAD PDF
# =========================================================

def load_pdf(pdf_path):

    return PyMuPDFLoader(
        pdf_path
    ).load()


# =========================================================
# YEAR
# =========================================================

def detect_year(pdf_path):

    match = re.search(
        r"\b(20\d{2})\b",
        pdf_path
    )

    if match:
        return match.group(1)

    return ""


# =========================================================
# REMOVE SOLUTIONS
# =========================================================

def strip_answer_blocks(text):
    """
    Remove supplied solutions from solved question papers.

    Stops skipping when the next TOP-LEVEL question begins.
    """

    if not text:
        return ""

    lines = text.splitlines()

    output = []

    skipping_answer = False

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        # ---------------------------------------------
        # START PROVIDED ANSWER
        #
        # Some solved-paper PDFs use "Solution:" (followed
        # by "Step 1:", "Step 2:", "Quick Tip") or "Correct
        # Answer:" instead of "Ans:"/"Answer:". Without
        # recognising those too, that text bled into the
        # question -- either merging unrelated questions
        # together (Solution:...Quick Tip runs right into
        # the next question with no blank line) or literally
        # printing the MCQ answer key inside the question
        # text shown to the student (Correct Answer: ...).
        # ---------------------------------------------

        if re.match(
            r"^(Ans|Answer|Solution|Correct\s+Answer)\s*:",
            line,
            re.IGNORECASE
        ):

            skipping_answer = True
            continue


        # ---------------------------------------------
        # NEW TOP-LEVEL QUESTION
        #
        # Supports:
        # 15. Question...
        # 15 Question...
        #
        # Requires actual sentence content afterwards.
        # ---------------------------------------------

        new_question = re.match(
            r"^\s*(\d{1,3})\s*[\.\)]?\s+([A-Za-z(])",
            line
        )


        if (
            skipping_answer
            and new_question
        ):

            skipping_answer = False


        # ---------------------------------------------
        # OR ALTERNATIVE
        # ---------------------------------------------

        if (
            skipping_answer
            and line.lower() == "or"
        ):

            skipping_answer = False

            output.append(
                "OR"
            )

            continue


        if skipping_answer:
            continue


        output.append(
            line
        )


    return "\n".join(
        output
    )


# =========================================================
# REMOVE PAGE NOISE
# =========================================================

def remove_noise(text):

    lines = []

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            continue


        lower = line.lower()


        if "www.vedantu.com" in lower:
            continue


        if re.fullmatch(
            r"class xii biology\s*\d*",
            lower
        ):
            continue


        if re.fullmatch(
            r"page\s*\d+",
            lower
        ):
            continue


        if lower.startswith(
            "previous year question paper"
        ):
            continue


        if lower.startswith(
            "cbse class 12"
        ):
            continue


        lines.append(
            line
        )


    return "\n".join(
        lines
    )


# =========================================================
# BILINGUAL (HINDI/ENGLISH) FILTERING
#
# CBSE Class 12 Biology papers from ~2015 onward print the
# Hindi translation of every question alongside the English
# one. We only want the English version. Real Devanagari
# Unicode text is detected and dropped line-by-line before
# question blocks are built, so Hindi content never reaches
# the question bank (this is also what was causing garbled
# text / rendering errors in the browser).
# =========================================================

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F


def contains_devanagari(text):

    for char in str(
        text or ""
    ):

        codepoint = ord(
            char
        )

        if (
            DEVANAGARI_START
            <= codepoint
            <= DEVANAGARI_END
        ):

            return True

    return False


def strip_devanagari_lines(text):

    lines = []

    for raw_line in str(
        text or ""
    ).splitlines():

        if contains_devanagari(
            raw_line
        ):

            continue

        lines.append(
            raw_line
        )

    return "\n".join(
        lines
    )


def sanitize_for_display(text):
    """
    Removes control / non-printable characters that can
    survive PDF text extraction and break JSON storage or
    Streamlit rendering.
    """

    text = str(
        text or ""
    )

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        text
    )

    return text.strip()


# =========================================================
# PAGE FOOTER TRUNCATION
#
# On some source PDFs, a question's block runs straight into
# the next page's footer/barcode without a clean line break --
# and sometimes into several following questions that turned
# out to be diagram-only (no real text between their number and
# marks label), e.g.:
#   "...encourage cross-pollination. 3 011111111111111111112
#    Page 14 of 23 [ ] 24. 3 25. 3 26. 3 27. 3 ... Page 15 of 23"
# Stripping individual known-junk substrings after the fact is
# fragile (each new footer style needs its own pattern and the
# leaked "24. 3 25. 3..." fragments are hard to tell apart from
# real content). Instead, the block is hard-truncated at the
# FIRST footer signature found -- nothing past a page footer is
# ever part of the actual question.
# =========================================================

PAGE_FOOTER_SIGNATURE = re.compile(
    r"_{5,}"
    # Real "P.T.O." always keeps its periods in extracted text --
    # the periods here are mandatory (not optional) specifically
    # so this can never match ordinary words like "hel(p) (t)(o)
    # encourage" where P/T/O just happen to appear with a space
    # between them (this was a real false-positive that truncated
    # a genuine question).
    r"|\bP\s*\.\s*T\s*\.\s*O\s*\.?"
    r"|\bP\s+a\s+g\s+e\b\s*\d*"
    r"|\bPage\s+\d+\s+of\s+\d+\b"
    r"|\b\d{2}/\d/\d(?:[-/]\d+)?\b"
    r"|\b0+1{8,}2\b",
    re.IGNORECASE
)


def truncate_at_page_footer(block):

    match = PAGE_FOOTER_SIGNATURE.search(block)

    if match and match.start() > 15:

        return block[:match.start()]

    return block


# =========================================================
# TOP LEVEL QUESTION DETECTION
# =========================================================

def find_question_starts(text):
    """
    Detect only likely top-level question numbers.

    We then validate numbering sequence to avoid treating
    internal lists such as 1), 2), 3) as board questions.
    """

    pattern = re.compile(
        r"""
        (?m)
        ^
        \s*
        (\d{1,3})
        \s*
        [\.\)]
        \s+
        (?=
            [A-Za-z('"\[]
        )
        """,
        re.VERBOSE
    )

    return list(
        pattern.finditer(
            text
        )
    )


# =========================================================
# QUESTION BLOCK EXTRACTION
# =========================================================

def split_question_blocks(text):

    matches = find_question_starts(
        text
    )

    if not matches:
        return []


    raw_blocks = []


    for index, match in enumerate(
        matches
    ):

        start = match.start()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )


        number = int(
            match.group(1)
        )


        block = text[
            start:end
        ].strip()


        raw_blocks.append(
            {
                "number": number,
                "block": block
            }
        )


    # =====================================================
    # SEQUENCE FILTER
    #
    # Board questions should generally progress:
    # 1,2,3,4...
    #
    # Internal numbered lists usually restart at 1.
    # =====================================================

    accepted = []

    expected = None


    for item in raw_blocks:

        number = item[
            "number"
        ]


        if expected is None:

            # Find plausible start.
            # Most CBSE papers begin at Q1.
            if number == 1:

                accepted.append(
                    item
                )

                expected = 2

            continue


        if number == expected:

            accepted.append(
                item
            )

            expected += 1

            continue


        # Sometimes parsing misses a question because
        # it begins on an image/diagram page.
        # Permit a small forward jump.
        if (
            number > expected
            and number <= expected + 2
        ):

            accepted.append(
                item
            )

            expected = (
                number + 1
            )

            continue


        # Ignore numbers that restart inside a solution,
        # case-study paragraph, list, etc.


    return accepted


# =========================================================
# MARKS
# =========================================================

def extract_marks(block):

    patterns = [

        r"\b([1-5])\s*Marks?\b",

        r"\bMarks?\s*[:\-]?\s*([1-5])\b",

        r"\[\s*([1-5])\s*\]"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            block,
            re.IGNORECASE
        )

        if match:

            return match.group(
                1
            )


    return "Unknown"


# =========================================================
# OPTIONS
# =========================================================

def extract_options(block):

    options = []


    option_pattern = re.compile(
        r"""
        (?m)
        ^
        \s*
        \(?
        ([A-Da-d])
        \)?
        [\.\)]
        \s*
        (.+?)
        \s*
        $
        """,
        re.VERBOSE
    )


    for match in option_pattern.finditer(
        block
    ):

        option_text = str(
            match.group(2)
        ).strip()


        if option_text:

            options.append(
                option_text
            )


    if len(options) >= 4:

        return options[
            :4
        ]


    return []


# =========================================================
# ANSWER KEY (when the source explicitly states one)
#
# A few solved-paper PDFs print "Correct Answer: (X) <option
# text>" right after the options. That is a genuine, verified
# answer key -- far more reliable than asking an LLM to guess
# the correct option at evaluation time. When present, it is
# captured here and later lets the app grade that MCQ through
# the same deterministic path used for system-generated
# questions, instead of the unverified-MCQ LLM fallback.
# =========================================================

def extract_answer_key(block, options):

    match = re.search(
        r"Correct\s+Answer\s*:\s*\(?([A-Da-d])\)?\s*(.*)",
        block,
        re.IGNORECASE
    )

    if not match:
        return ""

    letter = match.group(1).strip().upper()
    stated_text = match.group(2).strip()

    index = ord(letter) - ord("A")

    if 0 <= index < len(options):
        return options[index]

    if stated_text:
        return stated_text

    return ""


# =========================================================
# QUESTION TYPE
# =========================================================

def detect_question_type(
    block,
    options,
    marks
):
    """
    IMPORTANT — ordering matters here.

    Visual/diagram detection now runs BEFORE the MCQ check.
    Previously "len(options) == 4" was checked first, which
    meant any question block that happened to contain four
    lettered (a)/(b)/(c)/(d) lines got tagged "MCQ" even when
    those lines were really: (1) sub-parts of a diagram-
    labelling question ("Label the following: (a)...(d)..."),
    or (2) a genuine MCQ whose options describe a diagram/
    graph/pyramid the student can't actually see in the app.
    Both cases were slipping past the "exclude Diagram Based"
    filter in main.py and being served as answerable MCQs,
    which is the main reason MCQ practice was unreliable.

    A real CBSE MCQ is also always worth 1 mark. If four
    options were extracted but the block's own marks value
    explicitly says 2-5, that contradicts a genuine MCQ and
    almost always means extract_options() picked up lettered
    sub-parts of a longer descriptive/diagram question instead
    of real answer choices -- so it is not trusted as MCQ.
    """

    lower = block.lower()


    if any(
        phrase in lower
        for phrase in [
            "study the diagram",
            "study the given diagram",
            "draw a",
            "draw the",
            "diagram",
            "figure given",
            # "figure" alone (not just "figure given") was missing
            # here -- questions phrased "study the figure below",
            # "in the figure shown", "identify A, B, C in the
            # figure" etc. were slipping through un-tagged and
            # being served as normal answerable questions even
            # though the actual image was never captured.
            "figure",
            "diagrammatic representation",
            "schematic representation",
            "graph",
            "chart",
            "flowchart",
            "pyramid",
            "picture given",
            "picture shown",
            "observe the picture",
            "image given",
            "image shown",
            "map given",
            "structure shown",
            "photograph"
        ]
    ):

        return "Diagram Based"


    if (
        "assertion" in lower
        and "reason" in lower
    ):

        return "Assertion-Reason"


    if any(
        phrase in lower
        for phrase in [
            "case based",
            "case-based",
            "read the following passage",
            "read the passage",
            "study the following passage"
        ]
    ):

        return "Case Based"


    if (
        len(options) == 4
        and marks in ("Unknown", "1")
    ):

        return "MCQ"


    if marks in [
        "4",
        "5"
    ]:

        return "Long Answer"


    if marks in [
        "2",
        "3"
    ]:

        return "Short Answer"


    return "Other"


# =========================================================
# CLEAN QUESTION DISPLAY TEXT
# =========================================================

def clean_question_text(
    block,
    options
):

    lines = []


    for raw_line in block.splitlines():

        line = raw_line.strip()


        if not line:
            continue


        # Remove separate option lines.
        if re.match(
            r"^\s*\(?[A-Da-d]\)?[\.\)]\s+",
            line
        ):

            continue


        # Remove Answer/Solution/Correct Answer lines if
        # somehow retained.
        if re.match(
            r"^(Ans|Answer|Solution|Correct\s+Answer)\s*:",
            line,
            re.IGNORECASE
        ):

            break


        lines.append(
            line
        )


    text = " ".join(
        lines
    )


    # Remove leading question number.
    text = re.sub(
        r"^\s*\d{1,3}\s*[\.\)]\s*",
        "",
        text
    )


    # Remove marks label from displayed question.
    text = re.sub(
        r"\s*\b[1-5]\s*Marks?\b\s*",
        " ",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# =========================================================
# VALID QUESTION
# =========================================================

def valid_question(question):

    question = str(
        question or ""
    ).strip()


    if len(question) < 12:
        return False


    words = re.findall(
        r"[A-Za-z0-9]+",
        question
    )


    if len(words) < 3:
        return False


    return True


# =========================================================
# GENERAL INSTRUCTIONS / BOILERPLATE FILTER
#
# Some CBSE papers number their "General Instructions"
# section (1, 2, 3...) the same way real questions are
# numbered. These get matched by find_question_starts()
# but are not actual Biology questions.
# =========================================================

INSTRUCTION_PHRASES = [
    "all questions are compulsory",
    "there is no overall choice",
    "internal choice has been provided",
    "wherever necessary, the diagrams",
    "this question paper consists of",
    "question paper is divided into",
    "question paper contains",
    "questions number",
    "read the following instructions",
    "use of calculators is not permitted",
    "use of log tables"
]


def is_instruction_boilerplate(question):

    lower = str(
        question or ""
    ).strip().lower()

    return any(
        phrase in lower
        for phrase in INSTRUCTION_PHRASES
    )


# =========================================================
# SOLVED-ANSWER ARTIFACT FILTER
#
# Some solved PYQ compilation PDFs (e.g. certain "solution
# walkthrough" style papers) don't use a simple "Ans:" /
# "Answer:" prefix -- they use "Solution: Step 1: ... Step
# 2: ... Quick Tip". strip_answer_blocks() never learns to
# skip those (it only recognises "Ans:"/"Answer:"), and
# because the PDF's text extraction flattens multi-column
# solved layouts into one blob, the solution text for one
# question runs straight into the NEXT question's number
# without a clean line break the question-boundary regex
# can detect. The result is a single garbled "question"
# record that actually merges several unrelated questions
# and their full worked solutions together -- unusable, and
# it also poisons chapter mapping (mixed topics -> ambiguous
# embedding). Rather than risk a fragile re-parse of that
# format, blocks showing these markers are dropped entirely
# so they never reach students.
# =========================================================

SOLVED_ANSWER_MARKERS = [
    "solution:",
    "quick tip",
    "step 1:",
    "step 2:",
]


def contains_solved_answer_artifacts(text):

    lower = str(
        text or ""
    ).lower()

    hits = sum(
        1
        for marker in SOLVED_ANSWER_MARKERS
        if marker in lower
    )

    return hits >= 2


# =========================================================
# KEYWORD CHAPTER HINT
#
# map_question_to_chapter() picks a chapter purely from
# embedding similarity against NCERT chunks, which can
# misfire on short, fact-heavy MCQs or on chapters that
# share a lot of vocabulary (e.g. Biotechnology: Principles
# vs Biotechnology and its Applications; Human Reproduction
# vs Sexual Reproduction in Flowering Plants). This is a
# second, independent signal built from distinguishing
# NCERT terms per chapter -- when it strongly disagrees with
# the embedding result, it wins, since a clear keyword match
# ("Bt cotton", "Meselson and Stahl") is a stronger signal
# than an averaged nearest-neighbour score.
# =========================================================

CHAPTER_KEYWORDS = {

    "Sexual Reproduction in Flowering Plants": [
        "pollen grain", "pollination", "anther", "ovule", "embryo sac",
        "double fertilization", "double fertilisation", "endosperm",
        "pistil", "stigma", "style", "megasporogenesis", "microsporogenesis",
        "self-pollination", "self pollination", "cross pollination",
        "apomixis", "parthenocarpy", "dicot embryo", "synergid",
        "antipodal", "filiform apparatus", "megaspore mother cell",
        "microspore mother cell", "pollen tube", "emasculation",
    ],

    "Human Reproduction": [
        "testis", "testes", "ovary", "spermatogenesis", "oogenesis",
        "menstrual cycle", "fallopian tube", "oviduct", "uterus",
        "implantation", "gastrulation", "placenta", "parturition",
        "gametogenesis", "seminiferous tubule", "corpus luteum",
        "graafian follicle", "acrosome", "blastocyst", "gonad",
        "epididymis", "vas deferens",
    ],

    "Reproductive Health": [
        "contraceptive", "contraception", "medical termination of pregnancy",
        "sexually transmitted", "infertility", "in vitro fertilisation",
        "in vitro fertilization", "amniocentesis", "reproductive health",
        "family planning", "zift", "gift", "surrogate", "semen freezing",
        "copper t", "oral pill",
    ],

    "Principles of Inheritance and Variation": [
        "mendel", "monohybrid", "dihybrid", "allele", "genotype",
        "phenotype", "sex determination", "pedigree", "linkage",
        "down syndrome", "down's syndrome", "klinefelter", "turner syndrome",
        "dominant", "recessive", "test cross", "codominance",
        "law of dominance", "law of segregation", "independent assortment",
        "haemophilia", "hemophilia", "sickle cell anaemia",
        "sickle cell anemia", "punnett square",
    ],

    "Molecular Basis of Inheritance": [
        "dna replication", "transcription", "translation", "operon",
        "lac operon", "meselson", "stahl", "hershey", "chase", "griffith",
        "hnrna", "mrna", "trna", "rrna", "central dogma", "exon", "intron",
        "genome", "human genome project", "genetic code", "okazaki",
        "semiconservative", "nucleosome", "promoter", "transcription unit",
        "template strand", "codon", "anticodon", "rna polymerase",
    ],

    "Evolution": [
        "darwin", "natural selection", "lamarck", "hardy-weinberg",
        "hardy weinberg", "adaptive radiation", "homologous organ",
        "analogous organ", "fossil", "human evolution", "origin of life",
        "speciation", "miller", "urey", "industrial melanism",
        "survival of the fittest",
    ],

    "Human Health and Diseases": [
        "pathogen", "immunity", "antigen", "antibody", "aids", "hiv",
        "cancer", "malaria", "typhoid", "pneumonia", "common cold",
        "amoebiasis", "ringworm", "drug abuse", "vaccine", "allergy",
        "autoimmune", "plasmodium", "entamoeba", "lymphocyte",
        "interferon", "innate immunity", "acquired immunity",
    ],

    "Microbes in Human Welfare": [
        "fermentation", "biogas", "sewage treatment", "antibiotic",
        "curd", "cheese", "penicillin", "streptococcus", "saccharomyces",
        "biofertiliser", "biofertilizer", "rhizobium", "nitrobacter",
        "lactobacillus", "toddy", "swiss cheese", "statin",
    ],

    "Biotechnology: Principles and Processes": [
        "recombinant dna", "restriction enzyme", "cloning vector", "pcr",
        "gene cloning", "bioreactor", "competent host",
        "downstream processing", "ti plasmid", "restriction endonuclease",
        "origin of replication", "selectable marker", "electroporation",
        "gene gun", "biolistics", "ligase", "palindromic",
    ],

    "Biotechnology and its Applications": [
        "bt cotton", "bt toxin", "gm crop", "genetically modified crop",
        "gene therapy", "ada deficiency", "transgenic", "insulin",
        "humulin", "molecular diagnosis", "elisa", "golden rice", "rnai",
        "flavr savr", "bollworm", "cry protein", "transgenic animal",
    ],

    "Organisms and Populations": [
        "population growth", "carrying capacity", "logistic growth",
        "birth rate", "death rate", "age pyramid", "population density",
        "hibernation", "aestivation", "commensalism", "mutualism",
        "parasitism", "predation", "niche", "habitat",
        "population interaction", "exponential growth",
    ],

    "Ecosystem": [
        "energy flow", "food chain", "food web", "trophic level",
        "primary productivity", "decomposition", "ecological pyramid",
        "nutrient cycling", "carbon cycle", "phosphorus cycle",
        "standing crop", "gross primary productivity", "detritus",
        "humification", "litter",
    ],

    "Biodiversity and Conservation": [
        "biodiversity", "species richness", "biodiversity hotspot",
        "endangered species", "extinction", "iucn", "red data book",
        "in-situ conservation", "in situ conservation",
        "ex-situ conservation", "ex situ conservation", "sacred grove",
        "western ghats", "biosphere reserve",
    ],
}


def keyword_chapter_scores(text):

    lower = str(
        text or ""
    ).lower()

    scores = {}

    for chapter, words in CHAPTER_KEYWORDS.items():

        count = sum(
            1
            for word in words
            if word in lower
        )

        if count:
            scores[chapter] = count

    return scores


def resolve_chapter_with_keyword_hint(
    text,
    embedding_chapter
):
    """
    Trusts the embedding-based chapter UNLESS the keyword
    signal clearly points somewhere else and is stronger
    there than at the embedding chapter, by a margin of at
    least 2 -- a small margin is treated as noise.
    """

    scores = keyword_chapter_scores(
        text
    )

    if not scores:
        return embedding_chapter

    best_chapter = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_chapter]

    embedding_score = scores.get(
        embedding_chapter,
        0
    )

    if (
        best_chapter != embedding_chapter
        and best_score >= embedding_score + 2
    ):

        return best_chapter

    return embedding_chapter


# =========================================================
# CHAPTER DIRECTORY NORMALIZATION
# =========================================================

def chapter_key(value):

    value = strip_number_prefix(
        value
    )

    value = normalize_text(
        value
    )

    return value


# =========================================================
# FIND AVAILABLE VECTOR CHAPTERS
# =========================================================

def get_available_chapter_dbs():

    if not os.path.exists(
        CHAPTER_DB_DIR
    ):

        raise RuntimeError(
            "Class 12 chapter vector DB folder not found:\n"
            f"{CHAPTER_DB_DIR}"
        )


    folders = [
        path
        for path in glob.glob(
            os.path.join(
                CHAPTER_DB_DIR,
                "*"
            )
        )
        if os.path.isdir(
            path
        )
    ]


    if not folders:

        raise RuntimeError(
            "No Class 12 Biology chapter vector DBs found."
        )


    result = []


    for folder in folders:

        folder_name = os.path.basename(
            folder
        )


        result.append(
            {
                "folder_name":
                    folder_name,

                "path":
                    folder
            }
        )


    return result


# =========================================================
# MATCH VECTOR FOLDER TO DISPLAY CHAPTER
# =========================================================

def resolve_display_chapter(folder_name):

    folder_norm = chapter_key(
        folder_name
    )


    # Exact normalized match
    for chapter in VALID_CHAPTERS:

        if folder_norm == chapter_key(
            chapter
        ):

            return chapter


    # Compact comparison handles:
    # HumanReproduction
    # Human Reproduction
    compact_folder = re.sub(
        r"\s+",
        "",
        folder_norm
    )


    for chapter in VALID_CHAPTERS:

        compact_chapter = re.sub(
            r"\s+",
            "",
            chapter_key(
                chapter
            )
        )


        if compact_folder == compact_chapter:

            return chapter


    return ""


# =========================================================
# LOAD CHAPTER STORES
# =========================================================

def load_chapter_stores():

    stores = []


    embeddings = get_embeddings()


    for item in get_available_chapter_dbs():

        display_chapter = resolve_display_chapter(
            item[
                "folder_name"
            ]
        )


        if not display_chapter:

            print(
                "Skipping unrecognized chapter DB:",
                item[
                    "folder_name"
                ]
            )

            continue


        try:

            store = Chroma(
                persist_directory=item[
                    "path"
                ],
                embedding_function=embeddings
            )


            stores.append(
                {
                    "chapter":
                        display_chapter,

                    "store":
                        store
                }
            )


        except Exception as error:

            print(
                "Could not load chapter DB:",
                item[
                    "folder_name"
                ],
                error
            )


    if not stores:

        raise RuntimeError(
            "Could not load any usable Class 12 "
            "Biology chapter vector DBs."
        )


    print(
        "Loaded chapter DBs:",
        len(
            stores
        )
    )


    for item in stores:

        print(
            "  -",
            item[
                "chapter"
            ]
        )


    return stores


# =========================================================
# FAST IN-MEMORY CHAPTER INDEX
#
# Querying 13 separate Chroma DBs per question does not
# scale once the PYQ bank covers many years (thousands of
# questions x 13 chapter DBs = tens of thousands of DB
# round-trips). Instead we pull every chapter's already-
# computed embeddings into memory ONCE, and do a single
# vectorized nearest-neighbour lookup per question.
#
# This keeps the exact same "nearest NCERT chunk decides
# the chapter" logic as before, just without the repeated
# per-chapter DB queries.
# =========================================================

def build_chapter_index(chapter_stores):

    all_vectors = []
    all_chapters = []


    for item in chapter_stores:

        try:

            raw = item[
                "store"
            ]._collection.get(
                include=[
                    "embeddings"
                ]
            )

            vectors = raw.get(
                "embeddings"
            )

        except Exception as error:

            print(
                "Could not read embeddings for",
                item["chapter"],
                ":",
                error
            )

            continue


        if (
            vectors is None
            or len(vectors) == 0
        ):

            continue


        for vector in vectors:

            all_vectors.append(
                vector
            )

            all_chapters.append(
                item["chapter"]
            )


    if not all_vectors:

        raise RuntimeError(
            "No chapter embeddings were available "
            "for PYQ chapter mapping."
        )


    matrix = np.array(
        all_vectors,
        dtype=np.float32
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    matrix = matrix / norms


    print(
        "Built in-memory chapter index:",
        matrix.shape[0],
        "chunks across",
        len(set(all_chapters)),
        "chapters"
    )


    return {
        "matrix": matrix,
        "chapters": all_chapters
    }


def map_question_to_chapter(
    question,
    embeddings,
    chapter_index,
    k=6
):
    """
    Embed the question once, compare it against every
    stored NCERT chunk embedding in one vectorized op,
    and take the chapter that dominates the top-k nearest
    chunks (mirrors the previous per-chapter-DB approach,
    just computed as a single matrix multiply).

    Higher cosine similarity = stronger semantic match.
    """

    try:

        query_vector = np.array(
            embeddings.embed_query(
                question
            ),
            dtype=np.float32
        )

    except Exception:

        return "", None


    norm = np.linalg.norm(
        query_vector
    )

    if norm == 0:
        return "", None

    query_vector = query_vector / norm


    similarities = (
        chapter_index["matrix"]
        @ query_vector
    )

    top_k = min(
        k,
        len(similarities)
    )

    top_indices = np.argpartition(
        similarities,
        -top_k
    )[-top_k:]


    scores_by_chapter = {}

    for index in top_indices:

        chapter = chapter_index[
            "chapters"
        ][index]

        scores_by_chapter.setdefault(
            chapter,
            []
        ).append(
            float(
                similarities[index]
            )
        )


    best_chapter = ""
    best_score = None

    for chapter, scores in scores_by_chapter.items():

        average_score = (
            sum(scores)
            / len(scores)
        )

        if (
            best_score is None
            or average_score > best_score
        ):

            best_score = average_score
            best_chapter = chapter


    return (
        best_chapter,
        best_score
    )


# =========================================================
# LOCAL TOPIC
# =========================================================

def derive_topic(
    question,
    chapter_store
):

    try:

        docs = (
            chapter_store
            .similarity_search(
                question,
                k=1
            )
        )


        if not docs:
            return ""


        metadata = getattr(
            docs[0],
            "metadata",
            {}
        )


        topic = metadata.get(
            "topic",
            ""
        )


        if topic:

            return str(
                topic
            )


    except Exception:

        pass


    return ""


# =========================================================
# EXTRACT ONE PDF
# =========================================================

def extract_pdf_questions(
    pdf_path
):

    filename = os.path.basename(
        pdf_path
    )


    year = detect_year(
        pdf_path
    )


    pages = load_pdf(
        pdf_path
    )


    print(
        "Pages:",
        len(
            pages
        )
    )


    # =====================================================
    # COMBINE PDF TEXT
    # =====================================================

    text_parts = []


    for page in pages:

        content = clean_text(
            page.page_content
            or ""
        )


        if content:

            text_parts.append(
                content
            )


    full_text = "\n".join(
        text_parts
    )


    full_text = remove_noise(
        full_text
    )


    full_text = strip_devanagari_lines(
        full_text
    )


    # =====================================================
    # ANSWER KEYS (must be read BEFORE stripping)
    #
    # extract_answer_key() looks for "Correct Answer: (X)"
    # lines. strip_answer_blocks() removes exactly those
    # lines (so they never leak into the displayed question)
    # -- so the answer key has to be captured from the raw,
    # pre-strip blocks first, then matched back onto the
    # cleaned blocks by question number.
    # =====================================================

    raw_blocks = split_question_blocks(
        full_text
    )

    answer_keys_by_number = {}

    for raw_item in raw_blocks:

        raw_options = extract_options(
            raw_item["block"]
        )

        key = extract_answer_key(
            raw_item["block"],
            raw_options
        )

        if key:

            answer_keys_by_number[
                raw_item["number"]
            ] = key


    # =====================================================
    # STRIP PROVIDED ANSWERS
    # =====================================================

    question_text = strip_answer_blocks(
        full_text
    )


    # =====================================================
    # SPLIT
    # =====================================================

    blocks = split_question_blocks(
        question_text
    )


    print(
        "Validated top-level questions:",
        len(
            blocks
        )
    )


    records = []


    for item in blocks:

        question_number = str(
            item[
                "number"
            ]
        )


        block = item[
            "block"
        ]

        block = truncate_at_page_footer(
            block
        )


        marks = extract_marks(
            block
        )


        options = extract_options(
            block
        )


        answer_key = answer_keys_by_number.get(
            item["number"],
            ""
        )


        question_type = detect_question_type(
            block,
            options,
            marks
        )


        question = clean_question_text(
            block,
            options
        )

        question = sanitize_for_display(
            question
        )


        if not valid_question(
            question
        ):

            continue


        if is_instruction_boilerplate(
            question
        ):

            continue


        if contains_solved_answer_artifacts(
            block
        ):

            continue


        # MCQ must have 4 usable options.
        if (
            question_type == "MCQ"
            and len(options) != 4
        ):

            question_type = "Other"


        records.append(
            {
                "year":
                    year,

                "question_number":
                    question_number,

                "question":
                    question,

                "options":
                    options,

                "answer_key":
                    answer_key,

                "marks":
                    marks,

                "chapter":
                    "",

                "topic":
                    "",

                "question_type":
                    question_type,

                "source_file":
                    filename
            }
        )


    return records


# =========================================================
# DEDUPLICATE
# =========================================================

def deduplicate_records(records):

    seen = set()

    output = []


    for item in records:

        key = (
            item.get(
                "year",
                ""
            ),
            item.get(
                "source_file",
                ""
            ).lower(),
            normalize_text(
                item.get(
                    "question",
                    ""
                )
            )
        )


        if key in seen:
            continue


        seen.add(
            key
        )


        output.append(
            item
        )


    return output


# =========================================================
# LOAD EXISTING BANK (FOR INCREMENTAL BUILDS)
# =========================================================

def load_existing_records():

    if not os.path.exists(
        OUTPUT_JSON
    ):

        return []


    try:

        with open(
            OUTPUT_JSON,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

    except Exception as error:

        print(
            "Could not read existing pyq_questions.json:",
            error
        )

        return []


    if not isinstance(
        data,
        list
    ):

        return []


    return [
        item
        for item in data
        if isinstance(
            item,
            dict
        )
    ]


# =========================================================
# SAVE
# =========================================================

def save_records(records):

    temporary = (
        OUTPUT_JSON
        + ".tmp"
    )


    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2
        )


    os.replace(
        temporary,
        OUTPUT_JSON
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(mapped):

    chapter_counts = {
        chapter: 0
        for chapter in VALID_CHAPTERS
    }


    mark_counts = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "Unknown": 0
    }


    year_counts = {}


    unmapped = 0


    for item in mapped:

        chapter = item.get(
            "chapter",
            ""
        )


        if chapter in chapter_counts:

            chapter_counts[
                chapter
            ] += 1

        else:

            unmapped += 1


        marks = str(
            item.get(
                "marks",
                "Unknown"
            )
        )


        if marks not in mark_counts:

            marks = "Unknown"


        mark_counts[
            marks
        ] += 1


        year = item.get(
            "year",
            "Unknown"
        )


        year_counts[
            year
        ] = (
            year_counts.get(
                year,
                0
            )
            + 1
        )


    print(
        "\n====================================="
    )

    print(
        "PYQ BUILD COMPLETE"
    )

    print(
        "====================================="
    )


    print(
        "\nTotal questions:",
        len(
            mapped
        )
    )


    print(
        "Unmapped:",
        unmapped
    )


    print(
        "\nQUESTIONS BY YEAR"
    )


    for year in sorted(
        year_counts
    ):

        print(
            f"{year}: "
            f"{year_counts[year]}"
        )


    print(
        "\nQUESTIONS BY MARKS"
    )


    for marks in [
        "1",
        "2",
        "3",
        "4",
        "5",
        "Unknown"
    ]:

        print(
            f"{marks}: "
            f"{mark_counts[marks]}"
        )


    print(
        "\nQUESTIONS BY CHAPTER"
    )


    for chapter in VALID_CHAPTERS:

        print(
            f"{chapter}: "
            f"{chapter_counts[chapter]}"
        )


    print(
        "\nSaved to:"
    )


    print(
        OUTPUT_JSON
    )


def build_question_bank(force_rebuild=False):

    pdf_files = sorted(
        glob.glob(
            os.path.join(
                PYQ_SOURCE_DIR,
                "**",
                "*.pdf"
            ),
            recursive=True
        )
    )


    print(
        "\n====================================="
    )

    print(
        "BIOASSIST LOCAL PYQ BUILDER"
    )

    print(
        "====================================="
    )


    print(
        "PYQ folder:",
        PYQ_SOURCE_DIR
    )


    print(
        "PDF files found:",
        len(
            pdf_files
        )
    )


    print(
        "Chapter DB:",
        CHAPTER_DB_DIR
    )


    if not pdf_files:

        print(
            "ERROR: No PYQ PDFs found."
        )

        return []


    # =====================================================
    # INCREMENTAL: SKIP ALREADY-PROCESSED PDFs
    #
    # Rebuilding from scratch every time does not scale to
    # many years of papers. By default we only process PDFs
    # whose filename is not already present in the saved
    # question bank. Pass force_rebuild=True (or run with
    # --rebuild-all) to reprocess everything, which is
    # needed whenever the extraction/mapping logic itself
    # changes.
    # =====================================================

    existing_records = (
        []
        if force_rebuild
        else load_existing_records()
    )

    processed_files = {
        str(
            item.get(
                "source_file",
                ""
            )
        ).lower()
        for item in existing_records
        if item.get(
            "source_file"
        )
    }

    new_pdf_files = [
        path
        for path in pdf_files
        if os.path.basename(
            path
        ).lower()
        not in processed_files
    ]

    print(
        "Already in question bank:",
        len(
            existing_records
        ),
        "questions from",
        len(
            processed_files
        ),
        "PDF(s)"
    )

    print(
        "New PDFs to process:",
        len(
            new_pdf_files
        )
    )

    if force_rebuild:

        print(
            "force_rebuild=True: reprocessing everything."
        )


    if not new_pdf_files:

        print(
            "\nNothing new to process. "
            "Question bank is already up to date."
        )

        mapped = existing_records

        print_summary(
            mapped
        )

        return mapped


    # =====================================================
    # LOAD CHAPTER VECTOR DATABASES + FAST INDEX
    # =====================================================

    print(
        "\nLoading Class 12 NCERT chapter databases..."
    )


    chapter_stores = load_chapter_stores()


    chapter_store_map = {
        item[
            "chapter"
        ]:
        item[
            "store"
        ]
        for item in chapter_stores
    }

    chapter_index = build_chapter_index(
        chapter_stores
    )

    embeddings = get_embeddings()


    # =====================================================
    # EXTRACT ONLY NEW QUESTIONS LOCALLY
    # =====================================================

    all_records = []


    for index, pdf_path in enumerate(
        new_pdf_files,
        start=1
    ):

        filename = os.path.basename(
            pdf_path
        )


        year = detect_year(
            pdf_path
        )


        print(
            "\n-------------------------------------"
        )


        print(
            f"PDF {index}/{len(new_pdf_files)}"
        )


        print(
            "Processing:",
            year,
            filename
        )


        try:

            records = extract_pdf_questions(
                pdf_path
            )


        except Exception as error:

            print(
                "Extraction failed:",
                error
            )

            continue


        print(
            "Extracted:",
            len(
                records
            )
        )


        all_records.extend(
            records
        )


    # =====================================================
    # DEDUPLICATE (within the new batch, and against the
    # already-saved question bank)
    # =====================================================

    all_records = deduplicate_records(
        all_records
    )

    existing_keys = {
        (
            item.get("year", ""),
            str(item.get("source_file", "")).lower(),
            normalize_text(item.get("question", ""))
        )
        for item in existing_records
    }

    all_records = [
        item
        for item in all_records
        if (
            item.get("year", ""),
            str(item.get("source_file", "")).lower(),
            normalize_text(item.get("question", ""))
        )
        not in existing_keys
    ]


    print(
        "\nNew questions after extraction/dedup:",
        len(
            all_records
        )
    )


    # =====================================================
    # LOCAL CHAPTER MAPPING (new questions only)
    # =====================================================

    print(
        "\nMapping new questions to NCERT chapters locally..."
    )


    newly_mapped = []


    for index, item in enumerate(
        all_records,
        start=1
    ):

        chapter, similarity = map_question_to_chapter(
            item[
                "question"
            ],
            embeddings,
            chapter_index
        )

        chapter = resolve_chapter_with_keyword_hint(
            item["question"] + " " + " ".join(item.get("options") or []),
            chapter
        )


        item[
            "chapter"
        ] = chapter


        # Optional topic metadata from nearest chapter.
        if (
            chapter
            and chapter in chapter_store_map
        ):

            item[
                "topic"
            ] = derive_topic(
                item[
                    "question"
                ],
                chapter_store_map[
                    chapter
                ]
            )


        newly_mapped.append(
            item
        )


        if similarity is None:

            similarity_text = "N/A"

        else:

            similarity_text = (
                f"{similarity:.4f}"
            )


        print(
            f"{index}/{len(all_records)} "
            f"Q{item['question_number']} "
            f"({item['year']}) "
            f"→ {chapter or 'UNMAPPED'} "
            f"[{similarity_text}]"
        )


        # Checkpoint regularly.
        if (
            index % 10 == 0
            or index == len(
                all_records
            )
        ):

            save_records(
                existing_records
                + newly_mapped
            )


    mapped = (
        existing_records
        + newly_mapped
    )


    # =====================================================
    # SORT
    # =====================================================

    def sort_key(item):

        try:

            year = int(
                item.get(
                    "year",
                    0
                )
            )

        except Exception:

            year = 0


        try:

            number = int(
                item.get(
                    "question_number",
                    9999
                )
            )

        except Exception:

            number = 9999


        return (
            year,
            item.get(
                "source_file",
                ""
            ),
            number
        )


    mapped.sort(
        key=sort_key
    )


    save_records(
        mapped
    )


    print_summary(
        mapped
    )


    return mapped


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    rebuild_all = (
        "--rebuild-all"
        in sys.argv
    )

    if rebuild_all:

        print(
            "Running FULL rebuild "
            "(--rebuild-all flag detected)."
        )

    build_question_bank(
        force_rebuild=rebuild_all
    )