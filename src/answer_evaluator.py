import json
import re

from rag_service import (
    retrieve_ncert,
    get_llm
)


# =========================================================
# HELPERS
# =========================================================

def safe_list(value):
    """
    Always return a list.
    Prevents errors such as:
    TypeError: 'int' object is not iterable
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        return [value]

    return [str(value)]


def safe_score(value):
    """
    Convert LLM score safely to 0-10.
    """

    try:
        score = float(value)
    except Exception:
        score = 0.0

    score = max(
        0.0,
        min(10.0, score)
    )

    # Keep whole numbers as ints
    if score.is_integer():
        return int(score)

    return round(score, 1)


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
# MCQ NORMALIZATION
# =========================================================

def normalize_text(value):

    value = str(
        value or ""
    ).strip().lower()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


def extract_option_letter(value):
    """
    Accepts:
        B
        b
        B)
        b)
        B.
        (B)
        B) A small letter
    """

    value = normalize_text(
        value
    )

    match = re.match(
        r"^\(?\s*([a-d])\s*[\)\.\:\-]?",
        value
    )

    if match:
        return match.group(1)

    return ""


def remove_option_letter(value):
    """
    B) A small letter
        ->
    a small letter
    """

    value = normalize_text(
        value
    )

    value = re.sub(
        r"^\(?\s*[a-d]\s*[\)\.\:\-]?\s*",
        "",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_mcq_text(value):

    value = normalize_text(
        value
    )

    value = value.replace(
        "(",
        ""
    )

    value = value.replace(
        ")",
        ""
    )

    value = value.replace(
        ".",
        ""
    )

    value = value.replace(
        ":",
        ""
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# =========================================================
# MCQ CHECKING
# =========================================================

def evaluate_mcq_answer(
    student_answer,
    expected_answer
):

    if not str(
        student_answer
    ).strip():

        return False


    student_normalized = normalize_mcq_text(
        student_answer
    )

    expected_normalized = normalize_mcq_text(
        expected_answer
    )


    student_letter = extract_option_letter(
        student_answer
    )

    expected_letter = extract_option_letter(
        expected_answer
    )


    student_text = remove_option_letter(
        student_answer
    )

    expected_text = remove_option_letter(
        expected_answer
    )


    # -----------------------------------------------------
    # EXACT NORMALIZED MATCH
    # -----------------------------------------------------

    if (
        student_normalized
        and
        student_normalized
        ==
        expected_normalized
    ):
        return True


    # -----------------------------------------------------
    # LETTER MATCH
    #
    # B
    # b
    # B)
    # b)
    # -----------------------------------------------------

    if (
        student_letter
        and
        expected_letter
        and
        student_letter == expected_letter
    ):

        # Student entered only option letter
        if student_text in [
            "",
            student_letter
        ]:
            return True


        # Student entered correct letter + text
        if (
            expected_text
            and
            student_text == expected_text
        ):
            return True


    # -----------------------------------------------------
    # TEXT ONLY MATCH
    #
    # A small letter
    # a small letter
    # -----------------------------------------------------

    if (
        student_text
        and
        expected_text
        and
        student_text == expected_text
    ):
        return True


    # Student answer may not contain option prefix
    if (
        student_normalized
        and
        expected_text
        and
        student_normalized == expected_text
    ):
        return True


    return False


# =========================================================
# EXPECTED DEPTH
# =========================================================

def get_depth_instruction(
    question_level
):

    rules = {

        "2 Mark": """
This is a 2-mark question.

A complete answer should normally contain approximately
2 distinct important NCERT points.

A one-line or extremely incomplete response should not
receive full marks merely because it contains keywords.
""",

        "3 Mark": """
This is a 3-mark question.

A complete answer should normally contain approximately
3 distinct important NCERT points, concepts, steps,
features, or explanations.

A very short answer covering only one concept must not
receive full marks.
""",

        "4 Mark": """
This is a 4-mark question.

A complete answer should normally contain approximately
4 substantial relevant NCERT points, steps, concepts,
or an equivalently developed explanation.

A short response mentioning only one or two facts must
not receive full marks.
""",

        "5 Mark": """
This is a 5-mark question.

A complete answer should normally contain approximately
5 substantial NCERT points, steps, concepts, or a
properly developed explanation.

A response containing only one or two short points must
not receive full marks even if those points are correct.
"""
    }

    return rules.get(
        question_level,
        ""
    )


# =========================================================
# DEPTH / LENGTH CAP
# =========================================================

def apply_answer_depth_cap(
    score,
    student_answer,
    question_level
):
    """
    Content correctness remains the main scoring factor.

    Word count acts only as a CAP to prevent an obviously
    underdeveloped descriptive answer from receiving 10/10.
    """

    score = safe_score(
        score
    )

    words = re.findall(
        r"\b[\w'-]+\b",
        str(student_answer)
    )

    word_count = len(
        words
    )


    # MCQ is evaluated deterministically elsewhere.
    if question_level == "MCQ / 1 Mark":
        return score


    # -----------------------------------------------------
    # 2 MARK
    # -----------------------------------------------------

    if question_level == "2 Mark":

        if word_count < 8:
            return min(
                score,
                4
            )

        if word_count < 15:
            return min(
                score,
                7
            )


    # -----------------------------------------------------
    # 3 MARK
    # -----------------------------------------------------

    elif question_level == "3 Mark":

        if word_count < 12:
            return min(
                score,
                4
            )

        if word_count < 25:
            return min(
                score,
                7
            )

        if word_count < 35:
            return min(
                score,
                9
            )


    # -----------------------------------------------------
    # 4 MARK
    # -----------------------------------------------------

    elif question_level == "4 Mark":

        if word_count < 15:
            return min(
                score,
                4
            )

        if word_count < 30:
            return min(
                score,
                6
            )

        if word_count < 45:
            return min(
                score,
                8
            )

        if word_count < 55:
            return min(
                score,
                9
            )


    # -----------------------------------------------------
    # 5 MARK
    # -----------------------------------------------------

    elif question_level == "5 Mark":

        if word_count < 20:
            return min(
                score,
                3
            )

        if word_count < 35:
            return min(
                score,
                5
            )

        if word_count < 50:
            return min(
                score,
                7
            )

        if word_count < 65:
            return min(
                score,
                8
            )

        if word_count < 80:
            return min(
                score,
                9
            )


    return score


# =========================================================
# EXAM MARK CONVERSION
# =========================================================

def get_max_marks(
    question_level
):

    mapping = {
        "MCQ / 1 Mark": 1,
        "2 Mark": 2,
        "3 Mark": 3,
        "4 Mark": 4,
        "5 Mark": 5
    }

    return mapping.get(
        question_level,
        1
    )


def convert_score_to_exam_marks(
    score,
    question_level
):

    max_marks = get_max_marks(
        question_level
    )

    try:
        score = float(
            score
        )

    except Exception:
        score = 0.0


    marks = (
        score / 10.0
    ) * max_marks


    return round(
        marks,
        1
    )


# =========================================================
# MAIN EVALUATOR
# =========================================================

def evaluate_answer(
    question,
    student_answer,
    selected_class,
    chapter,
    expected_answer="",
    question_level="3 Mark"
):

    question = str(
        question or ""
    ).strip()

    student_answer = str(
        student_answer or ""
    ).strip()

    expected_answer = str(
        expected_answer or ""
    ).strip()


    # =====================================================
    # EMPTY ANSWER
    # =====================================================

    if not student_answer:

        return {
            "score": 0,
            "exam_marks": 0,
            "max_marks":
                get_max_marks(
                    question_level
                ),
            "correct_points": [],
            "missing_points": [
                "No answer was provided."
            ],
            "missing_keywords": [],
            "improvement":
                "Write your answer before submitting.",
            "model_answer":
                expected_answer
        }


    # =====================================================
    # MCQ
    #
    # IMPORTANT:
    # Do NOT ask the LLM to decide the answer again.
    # Use the generated expected answer.
    # =====================================================

    if question_level == "MCQ / 1 Mark":

        if not expected_answer:

            return {
                "score": 0,
                "exam_marks": 0,
                "max_marks": 1,
                "correct_points": [],
                "missing_points": [
                    "The expected answer was not available."
                ],
                "missing_keywords": [],
                "improvement":
                    "Generate a new question and try again.",
                "model_answer": ""
            }


        is_correct = evaluate_mcq_answer(
            student_answer,
            expected_answer
        )


        if is_correct:

            return {
                "score": 10,
                "exam_marks": 1,
                "max_marks": 1,
                "correct_points": [
                    "You selected the correct answer."
                ],
                "missing_points": [],
                "missing_keywords": [],
                "improvement":
                    "Your answer is correct.",
                "model_answer":
                    expected_answer
            }


        return {
            "score": 0,
            "exam_marks": 0,
            "max_marks": 1,
            "correct_points": [],
            "missing_points": [
                f"The correct answer is {expected_answer}."
            ],
            "missing_keywords": [],
            "improvement":
                "Review the relevant NCERT concept and try again.",
            "model_answer":
                expected_answer
        }


    # =====================================================
    # RETRIEVE NCERT EVIDENCE FOR DESCRIPTIVE ANSWERS
    # =====================================================

    try:

        docs = retrieve_ncert(
            question,
            selected_class,
            chapter,
            k=5
        )

    except Exception:

        docs = []


    context_parts = []

    for doc in docs or []:

        content = getattr(
            doc,
            "page_content",
            ""
        )

        if content:

            context_parts.append(
                content[:1600]
            )


    ncert_context = "\n\n".join(
        context_parts
    )


    depth_instruction = get_depth_instruction(
        question_level
    )


    # =====================================================
    # DESCRIPTIVE EVALUATION PROMPT
    # =====================================================

    prompt = f"""
You are BioAssist, an NCERT Biology formative evaluator.

Evaluate the student's answer fairly and consistently.

CLASS:
{selected_class}

CHAPTER:
{chapter}

QUESTION LEVEL:
{question_level}


QUESTION:

{question}


REFERENCE / EXPECTED ANSWER:

{expected_answer}


RETRIEVED NCERT EVIDENCE:

{ncert_context}


STUDENT ANSWER:

{student_answer}


DEPTH REQUIREMENT:

{depth_instruction}


SCORING RULES:

1. Score from 0 to 10.

2. Evaluate conceptual correctness first.

3. Compare the student's answer with the expected answer
   and the retrieved NCERT evidence.

4. Do NOT penalize a student merely because their wording
   differs from the model answer.

5. Accept scientifically equivalent NCERT-grounded wording.

6. Do NOT reward irrelevant padding.

7. The answer must have sufficient DEPTH for the selected
   mark level.

8. A 5-mark answer containing only one or two short points
   must NOT receive 10/10.

9. A 4-mark answer containing only one short fact must NOT
   receive full marks.

10. A 3-mark answer should normally contain around three
    important relevant ideas/steps/points.

11. A 2-mark answer should normally contain around two
    important relevant ideas/points.

12. If the student's answer fully covers the expected
    answer accurately, it may receive 10/10.

13. Do not invent missing concepts that are not supported
    by the reference answer or NCERT evidence.

14. "correct_points" must contain ONLY things the student
    actually wrote correctly.

15. "missing_points" must contain important expected ideas
    that the student omitted or explained incorrectly.

16. "missing_keywords" should contain only genuinely useful
    Biology terms that would strengthen the student's answer.

17. Keep feedback understandable to a school student.

Return ONLY valid JSON:

{{
    "score": 0,
    "correct_points": [],
    "missing_points": [],
    "missing_keywords": [],
    "improvement": "",
    "model_answer": ""
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
                "score": 0,
                "exam_marks": 0,
                "max_marks":
                    get_max_marks(
                        question_level
                    ),
                "correct_points": [],
                "missing_points": [
                    "BioAssist could not evaluate the answer."
                ],
                "missing_keywords": [],
                "improvement":
                    "Please submit the answer again.",
                "model_answer":
                    expected_answer
            }


        score = safe_score(
            parsed.get(
                "score",
                0
            )
        )


        # =================================================
        # APPLY DEPTH CAP
        # =================================================

        score = apply_answer_depth_cap(
            score,
            student_answer,
            question_level
        )


        correct_points = safe_list(
            parsed.get(
                "correct_points",
                []
            )
        )


        missing_points = safe_list(
            parsed.get(
                "missing_points",
                []
            )
        )


        missing_keywords = safe_list(
            parsed.get(
                "missing_keywords",
                []
            )
        )


        improvement = str(
            parsed.get(
                "improvement",
                ""
            )
        ).strip()


        model_answer = str(
            parsed.get(
                "model_answer",
                ""
            )
        ).strip()


        if not model_answer:

            model_answer = (
                expected_answer
            )


        exam_marks = convert_score_to_exam_marks(
            score,
            question_level
        )


        return {
            "score":
                score,

            "exam_marks":
                exam_marks,

            "max_marks":
                get_max_marks(
                    question_level
                ),

            "correct_points":
                correct_points,

            "missing_points":
                missing_points,

            "missing_keywords":
                missing_keywords,

            "improvement":
                improvement,

            "model_answer":
                model_answer
        }


    except Exception as error:

        return {
            "score": 0,
            "exam_marks": 0,
            "max_marks":
                get_max_marks(
                    question_level
                ),
            "correct_points": [],
            "missing_points": [
                "Unable to evaluate the answer."
            ],
            "missing_keywords": [],
            "improvement":
                str(error),
            "model_answer":
                expected_answer
        }