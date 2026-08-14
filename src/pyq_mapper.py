import os
import json
import re
from collections import defaultdict

import streamlit as st


# =========================================================
# PATH
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PYQ_JSON_PATH = os.path.join(
    PROJECT_DIR,
    "pyq_questions.json"
)


# =========================================================
# NORMALIZE CHAPTER
# =========================================================

def normalize_chapter(
    value
):

    value = str(
        value or ""
    ).strip().lower()

    # Removes:
    # "1. Human Reproduction"
    # "1) Human Reproduction"
    # "1 - Human Reproduction"
    value = re.sub(
        r"^\s*\d+\s*[\.\)\-:]\s*",
        "",
        value
    )

    value = value.replace(
        "–",
        "-"
    )

    value = value.replace(
        "—",
        "-"
    )

    value = value.replace(
        ":",
        " "
    )

    value = value.replace(
        "-",
        " "
    )

    value = value.replace(
        "_",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# =========================================================
# NORMALIZE QUESTION
# =========================================================

def normalize_question(
    value
):

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


# =========================================================
# LOAD BANK
# =========================================================

@st.cache_data(ttl=300)
def load_question_bank():
    # Streamlit reruns the whole script on every widget
    # interaction, and this file now holds 500+ records --
    # re-reading and re-parsing it from disk on every single
    # rerun (chapter box caption, practice question fetch,
    # progress page, etc.) added up across the app. Cached for
    # 5 minutes so a manual question-bank rebuild is picked up
    # without requiring a full app restart.

    if not os.path.exists(
        PYQ_JSON_PATH
    ):

        return []

    try:

        with open(
            PYQ_JSON_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

    except Exception as error:

        print(
            "Unable to load PYQ question bank:",
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
# QUESTION NUMBER SORT
# =========================================================

def question_number_value(
    value
):

    match = re.search(
        r"\d+",
        str(
            value or ""
        )
    )

    if not match:
        return 9999

    try:

        return int(
            match.group()
        )

    except Exception:

        return 9999


# =========================================================
# GET ALL QUESTIONS FOR CHAPTER
# =========================================================

def get_questions_for_chapter(
    selected_class,
    selected_chapter
):

    if selected_class != "Class 12":
        return []

    target_chapter = normalize_chapter(
        selected_chapter
    )

    if not target_chapter:
        return []

    matches = []

    seen = set()


    for item in load_question_bank():

        stored_chapter = normalize_chapter(
            item.get(
                "chapter",
                ""
            )
        )

        if stored_chapter != target_chapter:
            continue


        question = str(
            item.get(
                "question",
                ""
            )
        ).strip()

        if not question:
            continue


        # Keep the same question if it appeared in a
        # different year/paper.
        duplicate_key = (
            str(
                item.get(
                    "year",
                    ""
                )
            ),
            str(
                item.get(
                    "source_file",
                    ""
                )
            ).lower(),
            normalize_question(
                question
            )
        )

        if duplicate_key in seen:
            continue

        seen.add(
            duplicate_key
        )

        matches.append(
            item
        )


    # Newest papers first.
    # Within a paper, question-number order.
    matches.sort(
        key=lambda item: (
            -int(
                item.get(
                    "year",
                    "0"
                )
                if str(
                    item.get(
                        "year",
                        ""
                    )
                ).isdigit()
                else 0
            ),
            str(
                item.get(
                    "source_file",
                    ""
                )
            ).lower(),
            question_number_value(
                item.get(
                    "question_number",
                    ""
                )
            ),
            int(
                item.get(
                    "page",
                    0
                )
                or 0
            )
        )
    )

    return matches


# =========================================================
# MARK GROUP
# =========================================================

def get_mark_group(
    item
):

    question_type = str(
        item.get(
            "question_type",
            ""
        )
    ).strip().lower()


    # Reliable overrides.
    if question_type == "mcq":
        return "1"

    if question_type == "assertion-reason":
        return "1"


    raw_marks = str(
        item.get(
            "marks",
            ""
        )
    ).strip()


    # Important:
    # Require the actual field to represent 1-5.
    # Do NOT search arbitrary text such as "15 marks".
    match = re.fullmatch(
        r"\s*([1-5])(?:\.0)?(?:\s*marks?)?\s*",
        raw_marks,
        re.IGNORECASE
    )

    if match:

        return match.group(
            1
        )


    return "Unknown"


# =========================================================
# GROUP BY MARKS
# =========================================================

def group_questions_by_marks(
    questions
):

    grouped = defaultdict(
        list
    )


    for item in questions:

        mark_group = get_mark_group(
            item
        )

        grouped[
            mark_group
        ].append(
            item
        )


    return {

        "1":
            grouped.get(
                "1",
                []
            ),

        "2":
            grouped.get(
                "2",
                []
            ),

        "3":
            grouped.get(
                "3",
                []
            ),

        "4":
            grouped.get(
                "4",
                []
            ),

        "5":
            grouped.get(
                "5",
                []
            ),

        "Unknown":
            grouped.get(
                "Unknown",
                []
            )
    }


# =========================================================
# OPTIONAL SUMMARY
# =========================================================

def get_question_counts(
    selected_class,
    selected_chapter
):

    questions = get_questions_for_chapter(
        selected_class,
        selected_chapter
    )

    grouped = group_questions_by_marks(
        questions
    )

    return {
        "total":
            len(
                questions
            ),

        "1":
            len(
                grouped["1"]
            ),

        "2":
            len(
                grouped["2"]
            ),

        "3":
            len(
                grouped["3"]
            ),

        "4":
            len(
                grouped["4"]
            ),

        "5":
            len(
                grouped["5"]
            ),

        "Unknown":
            len(
                grouped["Unknown"]
            )
    }