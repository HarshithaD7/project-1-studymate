import json
import random

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
        k=8
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