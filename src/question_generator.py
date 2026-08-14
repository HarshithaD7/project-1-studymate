import json
import random
import re

from rag_service import (
    retrieve_ncert,
    get_llm
)


# =========================================================
# JSON PARSER
# =========================================================

def parse_json_response(raw):

    if not raw:
        return None

    raw = (
        str(raw)
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(raw)

    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        return None

    try:
        return json.loads(
            raw[start:end + 1]
        )

    except Exception:
        return None


# =========================================================
# QUESTION LEVEL RULES
# =========================================================

def get_level_instruction(
    question_level
):

    rules = {

        "MCQ / 1 Mark": """
Generate ONE 1-mark Biology question.

Prefer an MCQ when appropriate.

If it is an MCQ:
- provide exactly 4 meaningful options
- provide the correct answer
""",

        "1 Mark": """
Generate ONE 1-mark Biology short-answer question -- NOT
a multiple-choice question.

The expected answer should be a single precise fact,
term, or concept from NCERT. Keep it short and specific.

Set "options" to an empty array.
""",

        "2 Mark": """
Generate ONE 2-mark short-answer Biology question.

The expected answer should require approximately
two important NCERT points.
""",

        "3 Mark": """
Generate ONE 3-mark Biology question.

The expected answer should require approximately
three important NCERT concepts, steps, or points.
""",

        "4 Mark": """
Generate ONE 4-mark Biology question.

The expected answer should require a structured response
covering several relevant NCERT concepts.
""",

        "5 Mark": """
Generate ONE 5-mark Biology question.

The expected answer should require a detailed,
well-structured NCERT-grounded explanation.
"""
    }

    return rules.get(
        question_level,
        rules["3 Mark"]
    )


# =========================================================
# RECONCILE MCQ ANSWER AGAINST OPTIONS
#
# The prompt instructs the LLM to make "answer" an exact
# copy of one of the "options" strings, but an 8B model is
# not guaranteed to follow that format perfectly. Without
# this, evaluate_mcq_answer() in answer_evaluator.py cannot
# reliably match a student's correct selection against a
# reworded/prefixed answer string. This snaps "answer" back
# to the exact option text whenever possible.
# =========================================================

def _normalize_for_match(value):

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value or "").lower()
    ).strip()


def reconcile_mcq_answer(
    answer,
    options
):

    if not options:
        return answer

    normalized_answer = _normalize_for_match(
        answer
    )

    if not normalized_answer:
        return answer

    # Exact normalized match.
    for option in options:

        if _normalize_for_match(option) == normalized_answer:
            return option

    # Partial containment (e.g. answer came back as
    # "B) DNA" or wrapped in a short sentence).
    best_option = ""
    best_overlap = 0

    for option in options:

        normalized_option = _normalize_for_match(
            option
        )

        if not normalized_option:
            continue

        if (
            normalized_option in normalized_answer
            or normalized_answer in normalized_option
        ):

            overlap = min(
                len(normalized_option),
                len(normalized_answer)
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_option = option

    if best_option:
        return best_option

    return answer


# =========================================================
# GENERATE EVALUATION QUESTION
# =========================================================

def generate_evaluation_question(
    selected_class,
    chapter,
    question_level,
    previous_question=""
):

    # Retrieve a larger pool from the selected chapter.
    # We then randomly select a few chunks to improve variety.
    docs = retrieve_ncert(
        chapter,
        selected_class,
        chapter,
        # Only 3 of these are ever sampled into the prompt below;
        # k=8 was retrieving more candidates than needed for that,
        # adding retrieval time without adding variety.
        k=5
    )

    if not docs:

        return {
            "question": "",
            "options": [],
            "answer": "",
            "error":
                "No NCERT evidence was found for the selected chapter."
        }


    docs = list(
        docs
    )


    # =====================================================
    # RANDOM NCERT CONTEXT
    # =====================================================

    sample_size = min(
        3,
        len(docs)
    )

    selected_docs = random.sample(
        docs,
        sample_size
    )


    context_parts = []

    for doc in selected_docs:

        page_content = getattr(
            doc,
            "page_content",
            ""
        )

        if page_content:

            context_parts.append(
                page_content[:1400]
            )


    context = "\n\n".join(
        context_parts
    )


    if not context.strip():

        return {
            "question": "",
            "options": [],
            "answer": "",
            "error":
                "Retrieved NCERT documents did not contain readable content."
        }


    # =====================================================
    # LEVEL INSTRUCTION
    # =====================================================

    level_instruction = get_level_instruction(
        question_level
    )


    # =====================================================
    # AVOID PREVIOUS QUESTION
    # =====================================================

    previous_instruction = ""

    if previous_question:

        previous_instruction = f"""
PREVIOUSLY GENERATED QUESTION:

{previous_question}

Generate a DIFFERENT question.

Do not repeat the previous question.
Do not simply paraphrase it.
Prefer another concept from the supplied NCERT context.
"""


    # =====================================================
    # PROMPT
    # =====================================================

    prompt = f"""
You are BioAssist AI.

Generate exactly ONE Biology examination question
using the retrieved NCERT evidence.

CLASS:
{selected_class}

CHAPTER:
{chapter}

QUESTION LEVEL:
{question_level}


RETRIEVED NCERT CONTEXT:

{context}


{previous_instruction}


QUESTION REQUIREMENT:

{level_instruction}


STRICT RULES:

1. Use ONLY the supplied NCERT context.

2. The question must belong to the selected chapter:
   {chapter}

3. Generate exactly ONE question.

4. Do not use outside knowledge.

5. Do not ask something unsupported by the supplied context.

6. Make the question clear and exam-oriented.

7. Also generate a concise NCERT-grounded model answer.

8. If a previous question is supplied:
   generate a genuinely different question.

9. Do not repeatedly ask about the same small concept
   when another concept is available.

10. For MCQ / 1 Mark:
    - generate exactly four options if using MCQ
    - options must be meaningful.
    - the "answer" field MUST be an EXACT, character-for-
      character copy of one of the 4 strings in "options".
    - do NOT add a letter prefix (no "A)", no "B."), do NOT
      rephrase it, do NOT turn it into a sentence. Copy the
      matching option text exactly as it appears in "options".

11. For 2/3/4/5 Mark:
    - options must be an empty array.

12. Return ONLY valid JSON.

Return exactly:

{{
    "question": "",
    "options": [],
    "answer": ""
}}
"""


    # =====================================================
    # CALL LLM
    # =====================================================

    try:

        response = get_llm().invoke(
            prompt
        )

        parsed = parse_json_response(
            response.content
        )


        if not parsed:

            return {
                "question": "",
                "options": [],
                "answer": "",
                "error":
                    "Unable to parse the generated question."
            }


        question = str(
            parsed.get(
                "question",
                ""
            )
        ).strip()


        options = parsed.get(
            "options",
            []
        )


        answer = str(
            parsed.get(
                "answer",
                ""
            )
        ).strip()


        if not isinstance(
            options,
            list
        ):

            options = []


        options = [
            str(option).strip()
            for option in options
            if str(option).strip()
        ]


        if not question:

            return {
                "question": "",
                "options": [],
                "answer": "",
                "error":
                    "BioAssist did not generate a valid question."
            }


        # =====================================================
        # MCQ ANSWER MUST MATCH AN OPTION
        # =====================================================

        if options:

            answer = reconcile_mcq_answer(
                answer,
                options
            )

            if _normalize_for_match(
                answer
            ) not in [
                _normalize_for_match(option)
                for option in options
            ]:

                return {
                    "question": "",
                    "options": [],
                    "answer": "",
                    "error":
                        "BioAssist generated an MCQ answer that "
                        "did not match any option. Please try "
                        "generating the question again."
                }


        return {
            "question": question,
            "options": options,
            "answer": answer,
            "error": ""
        }


    except Exception as error:

        return {
            "question": "",
            "options": [],
            "answer": "",
            "error": str(error)
        }