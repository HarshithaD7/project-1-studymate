import json
import re
import time

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


def is_low_content_gibberish(letters_and_digits_only):
    """
    Catches keyboard-mash / placeholder input like "nnnn",
    "asdasd", "xxxxx", "1111" that carries no real Biology
    content -- BEFORE it ever reaches the LLM.

    Why this exists: the LLM was scoring "nnnn" a 4/10 on a
    3-mark question. The word-count depth cap (see
    apply_answer_depth_cap) only limits the CEILING for short
    answers -- for a <12-word 3-mark answer that ceiling is
    min(score, 4), so a lenient LLM score of 4+ came through
    as exactly 4. The post-parse "empty correct_points" clamp
    didn't catch it either, because the LLM sometimes invents
    a correct_points entry even for nonsense input. None of
    that is reliable for zero-content input, so it is rejected
    deterministically first.
    """

    text = re.sub(
        r"[^a-z0-9]",
        "",
        letters_and_digits_only
    )

    if len(text) < 3:
        return True

    unique_chars = set(text)

    if len(unique_chars) <= 2:
        return True

    most_common_count = max(
        text.count(char)
        for char in unique_chars
    )

    if most_common_count / len(text) >= 0.6:
        return True

    has_letters = any(
        char.isalpha()
        for char in text
    )

    has_vowel = any(
        char in "aeiou"
        for char in text
    )

    if (
        has_letters
        and not has_vowel
        and len(text) >= 4
    ):
        return True

    return False


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

    IMPORTANT:
    The delimiter after the letter (), ., :, -) or the end
    of the string is REQUIRED here. Without that requirement,
    any word that simply starts with a/b/c/d -- DNA, Diabetes,
    Auxin, Biotechnology, Cytokinin, Chromosome, and plenty of
    other real Biology terms -- would be misread as "option
    letter + leftover text", corrupting the comparison.
    """

    value = normalize_text(
        value
    )

    # Whole string is just the letter, e.g. "b", "(b)"
    exact_match = re.fullmatch(
        r"\(?\s*([a-d])\s*\)?",
        value
    )

    if exact_match:
        return exact_match.group(1)

    # Letter immediately followed by a real delimiter, and
    # then either more text or the end of the string, e.g.
    # "b)", "b.", "b) a small letter"
    prefix_match = re.match(
        r"^\(?\s*([a-d])[\)\.\:\-](?:\s+|$)",
        value
    )

    if prefix_match:
        return prefix_match.group(1)

    return ""


def remove_option_letter(value):
    """
    B) A small letter
        ->
    a small letter

    Only strips a leading letter when it is followed by a
    genuine delimiter (or is the entire string). A word like
    "DNA" or "Diabetes" is left untouched -- see the note in
    extract_option_letter() for why this guard is required.
    """

    value = normalize_text(
        value
    )

    if re.fullmatch(
        r"\(?\s*[a-d]\s*\)?",
        value
    ):
        return ""

    value = re.sub(
        r"^\(?\s*[a-d][\)\.\:\-](?:\s+|$)",
        "",
        value,
        count=1
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
# UNVERIFIED MCQ (real previous-year question, no stored
# answer key)
#
# IMPORTANT DESIGN NOTE:
# Earlier this asked the LLM to grade the student directly
# ("is this correct? score it"), which is exactly the kind
# of self-graded judgment call small LLMs tend to answer
# leniently/agreeably on -- reported symptom was "even a
# wrong option gets accepted as right". To remove that
# failure mode, the LLM is now asked ONLY the narrower,
# more factual question "which option is correct", and the
# actual accept/reject decision is made deterministically
# here in code by comparing letters -- the LLM's own score
# self-assessment is never trusted.
# =========================================================

def evaluate_unverified_mcq(
    question,
    student_answer,
    selected_class,
    chapter,
    options
):

    try:

        docs = retrieve_ncert(
            question,
            selected_class,
            chapter,
            # Identifying the correct option out of 4 doesn't need
            # as much grounding as a full descriptive evaluation --
            # trimmed the same way as evaluate_answer() above to
            # keep this LLM round trip fast.
            k=3
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
                content[:900]
            )

    ncert_context = "\n\n".join(
        context_parts
    )

    lettered_options = []

    for index, option_text in enumerate(options):

        letter = chr(65 + index)

        lettered_options.append(
            f"{letter}) {option_text}"
        )

    options_block = "\n".join(
        lettered_options
    )

    prompt = f"""
You are BioAssist, an NCERT Biology expert.

This is a 1-mark multiple choice question from a real CBSE
Class 12 Biology previous-year paper. There is no stored
answer key, so you must work out the correct option yourself.

CLASS:
{selected_class}

CHAPTER:
{chapter}

QUESTION:

{question}

OPTIONS:

{options_block}

RETRIEVED NCERT EVIDENCE (may be empty or incomplete):

{ncert_context}

TASK:

Identify which ONE of the four options (A, B, C, or D) is
scientifically correct. Prefer the retrieved NCERT evidence
when it is relevant; otherwise rely on accurate NCERT Class
12 Biology knowledge. Exactly one option must be chosen, even
if you are not fully certain.

Return ONLY valid JSON in this exact shape:

{{
    "correct_option": "A",
    "explanation": ""
}}
"""

    student_letter = extract_option_letter(
        student_answer
    )

    try:

        _llm_start = time.time()

        response = get_llm().invoke(
            prompt
        )

        print(f"[BioAssist timing] LLM call (evaluate_unverified_mcq, every call): {time.time() - _llm_start:.2f}s")

        parsed = parse_json_response(
            response.content
        )

    except Exception:

        parsed = None

    if not parsed or not str(
        parsed.get("correct_option", "")
    ).strip():

        return {
            "score": 0,
            "exam_marks": 0,
            "max_marks": 1,
            "correct_points": [],
            "missing_points": [
                "BioAssist could not verify this question "
                "right now."
            ],
            "missing_keywords": [],
            "improvement":
                "Please submit your answer again.",
            "model_answer": ""
        }

    raw_letter = normalize_text(
        parsed.get("correct_option", "")
    ).replace(")", "").replace(".", "").strip()

    correct_letter = raw_letter[:1] if raw_letter else ""

    explanation = str(
        parsed.get("explanation", "")
    ).strip()

    correct_option_text = ""

    if correct_letter:

        index = ord(correct_letter) - 97

        if 0 <= index < len(options):
            correct_option_text = options[index]

    model_answer = (
        f"{correct_letter.upper()}) {correct_option_text}"
        if correct_option_text
        else correct_letter.upper()
    )

    is_correct = (
        bool(student_letter)
        and bool(correct_letter)
        and student_letter == correct_letter
    )

    if is_correct:

        correct_points = [
            "You selected the correct option."
        ]

        if explanation:
            correct_points.append(explanation)

        return {
            "score": 10,
            "exam_marks": 1,
            "max_marks": 1,
            "correct_points": correct_points,
            "missing_points": [],
            "missing_keywords": [],
            "improvement":
                "Your answer is correct.",
            "model_answer": model_answer
        }

    missing_points = [
        f"The correct answer is {model_answer}."
    ]

    if explanation:
        missing_points.append(explanation)

    return {
        "score": 0,
        "exam_marks": 0,
        "max_marks": 1,
        "correct_points": [],
        "missing_points": missing_points,
        "missing_keywords": [],
        "improvement":
            "Review the relevant NCERT concept and try again.",
        "model_answer": model_answer
    }


# =========================================================
# EXPECTED DEPTH
# =========================================================

def get_depth_instruction(
    question_level
):

    rules = {

        "1 Mark": """
This is a 1-mark question.

A complete answer should normally be one precise NCERT
fact, term, or concept -- not a full explanation.

Do not penalize brevity here the way longer answers are
penalized; a short, correct, on-topic answer can receive
full marks.
""",

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
        "1 Mark": 1,
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
    question_level="3 Mark",
    options=None
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

    options = [
        str(option).strip()
        for option in (options or [])
        if str(option).strip()
    ]


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
    # NON-ANSWER ("not sure", "don't know", etc.)
    #
    # These carry zero Biology content, so there is nothing
    # for an LLM to legitimately score above 0 -- but LLMs
    # tend to be agreeable/generous even here, and even a
    # generic safety clamp can leave a non-zero floor. This
    # is graded deterministically instead of ever reaching
    # the LLM, for MCQ and descriptive questions alike.
    # =====================================================

    NON_ANSWER_PHRASES = {
        "not sure", "notsure", "not sure sir", "not sure about this",
        "dont know", "don't know", "do not know", "i dont know",
        "i don't know", "idk", "no idea", "not aware", "unaware",
        "cant say", "can't say", "no answer", "skip", "pass",
        "i have no idea", "not known", "unknown", "na", "n/a", "?",
    }

    normalized_student_answer = re.sub(
        r"[^a-z0-9\s]",
        "",
        student_answer.lower()
    ).strip()

    normalized_student_answer = re.sub(
        r"\s+",
        " ",
        normalized_student_answer
    )

    is_non_answer = (
        normalized_student_answer in NON_ANSWER_PHRASES
        or is_low_content_gibberish(
            normalized_student_answer
        )
    )

    if (
        question_level != "MCQ / 1 Mark"
        and is_non_answer
    ):

        return {
            "score": 0,
            "exam_marks": 0,
            "max_marks":
                get_max_marks(
                    question_level
                ),
            "correct_points": [],
            "missing_points": [
                "No real attempt was made at this question."
            ],
            "missing_keywords": [],
            "improvement":
                "Try writing what you remember, even if "
                "you're unsure -- a partial, on-topic attempt "
                "can still earn marks, but random text or "
                "\"not sure\" cannot.",
            "model_answer":
                expected_answer
        }


    # =====================================================
    # MCQ WITH A KNOWN ANSWER KEY
    #
    # IMPORTANT:
    # Do NOT ask the LLM to decide the answer again.
    # Use the generated expected answer.
    #
    # This only applies when expected_answer is available
    # (system-generated questions always provide one). Real
    # previous-year MCQs have no stored answer key, so those
    # fall through to the NCERT-grounded LLM evaluation
    # below instead of erroring out.
    # =====================================================

    if question_level == "MCQ / 1 Mark" and expected_answer:

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
    # MCQ WITHOUT A KNOWN ANSWER KEY (real previous-year MCQ)
    #
    # Handled entirely by evaluate_unverified_mcq(), which
    # makes the correct/incorrect decision deterministically
    # in code rather than trusting an LLM self-reported score.
    # This short-circuits before the generic descriptive
    # prompt below, which is no longer MCQ-aware.
    # =====================================================

    if (
        question_level == "MCQ / 1 Mark"
        and not expected_answer
        and options
    ):

        return evaluate_unverified_mcq(
            question,
            student_answer,
            selected_class,
            chapter,
            options
        )


    # =====================================================
    # RETRIEVE NCERT EVIDENCE FOR DESCRIPTIVE ANSWERS
    # =====================================================

    try:

        docs = retrieve_ncert(
            question,
            selected_class,
            chapter,
            # Fewer, more targeted chunks: k=5 at 1600 chars each
            # (up to 8000 chars of context) was making every
            # evaluation prompt large enough to noticeably slow
            # down the LLM round trip. The top 3 chunks carry
            # almost all the relevant grounding; the 4th/5th
            # added bulk more than signal.
            k=3
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
                content[:900]
            )


    ncert_context = "\n\n".join(
        context_parts
    )


    depth_instruction = get_depth_instruction(
        question_level
    )


    # =====================================================
    # MARK-DEPTH SCORING RULES
    # =====================================================

    mark_depth_rules = """
8. A 5-mark answer containing only one or two short points
   must NOT receive 10/10.

9. A 4-mark answer containing only one short fact must NOT
   receive full marks.

10. A 3-mark answer should normally contain around three
    important relevant ideas/steps/points.

11. A 2-mark answer should normally contain around two
    important relevant ideas/points.
"""


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

6a. If the student's answer is factually wrong, contradicts
    the reference answer / NCERT evidence, or is off-topic,
    you MUST score it 0-2 regardless of length, fluency, or
    confident tone. A long, well-written but incorrect answer
    is still incorrect -- do not award marks for effort alone.

6b. If "correct_points" would be empty (the student got
    nothing meaningfully right), the score must be 3 or below.

7. The answer must have sufficient DEPTH for the selected
   mark level.
{mark_depth_rules}
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

        _llm_start = time.time()

        response = get_llm().invoke(
            prompt
        )

        print(f"[BioAssist timing] LLM call (evaluate_answer, every call): {time.time() - _llm_start:.2f}s")

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


        # =================================================
        # DETERMINISTIC SAFETY CLAMP
        #
        # Prompt rule 6b already tells the LLM to keep the
        # score low when it found nothing correct, but LLM
        # self-scoring is exactly where leniency creeps in
        # (a wordy-but-wrong answer scoring high). This is a
        # code-level backstop: if the model's OWN analysis
        # says it found no correct points, the score cannot
        # exceed 2, no matter what number it self-reported.
        # (Genuine non-answers like "not sure" never reach
        # this code at all -- see the NON-ANSWER check above,
        # which returns 0 directly.)
        # =================================================

        if (
            not correct_points
            and score > 2
        ):

            score = 2


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