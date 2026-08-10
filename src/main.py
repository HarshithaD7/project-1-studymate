import os

import streamlit as st
from dotenv import load_dotenv

from chatbot_utility import get_chapter_list
from rag_service import answer_question

from question_generator import (
    generate_evaluation_question
)

from answer_evaluator import evaluate_answer

from pyq_mapper import (
    get_questions_for_chapter,
    group_questions_by_marks
)

from progress_tracker import (
    save_progress,
    get_progress
)


# =========================================================
# ENVIRONMENT
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ENV_PATH = os.path.join(
    PROJECT_DIR,
    "src",
    ".env"
)

load_dotenv(
    dotenv_path=ENV_PATH
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BioAssist AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# UI STYLING
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1220px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    .app-title {
        font-size: 2.15rem;
        font-weight: 760;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }

    .app-subtitle {
        opacity: 0.70;
        font-size: 0.94rem;
        margin-bottom: 1.6rem;
    }

    .chapter-box {
        padding: 0.9rem 1rem;
        margin-bottom: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.28);
        background: rgba(128,128,128,0.07);
    }

    section[data-testid="stSidebar"] {
        min-width: 260px;
        max-width: 260px;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 2.5rem;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FIXED INTERNAL STUDENT ID
# =========================================================

student = "default_student"


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "evaluation_question_data": None,
    "evaluation_level": "3 Mark",
    "evaluation_result": None,
    "evaluation_question_id": 0,
    "last_evaluation_context": None
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="app-title">
        🧬 BioAssist AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-subtitle">
        NCERT Biology Learning & Exam Preparation Assistant
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "📖 Study Setup"
    )

    selected_class = st.selectbox(
        "Select Class",
        [
            "Class 11",
            "Class 12"
        ]
    )

    chapters = get_chapter_list(
        selected_class,
        "Biology"
    )

    if not chapters:

        st.error(
            "No Biology chapters were found."
        )

        st.stop()

    chapter = st.selectbox(
        "Select Chapter",
        chapters
    )

    st.divider()

    st.markdown(
        "### Navigation"
    )

    mode = st.radio(
        "Mode",
        [
            "📘 Learn",
            "🚀 Practice / PYQs",
            "✅ Evaluate My Answer",
            "📊 My Progress"
        ],
        label_visibility="collapsed"
    )


# =========================================================
# RESET EVALUATION ON CLASS / CHAPTER CHANGE
# =========================================================

current_context = (
    selected_class,
    chapter
)

if (
    st.session_state.last_evaluation_context
    != current_context
):

    st.session_state.evaluation_question_data = None
    st.session_state.evaluation_result = None
    st.session_state.evaluation_question_id += 1

    st.session_state.last_evaluation_context = (
        current_context
    )


# =========================================================
# PROGRESS HELPER
# =========================================================

def save_progress_safely(
    activity,
    topic,
    score=None
):

    try:

        if score is None:

            save_progress(
                student,
                selected_class,
                chapter,
                activity,
                topic
            )

        else:

            save_progress(
                student,
                selected_class,
                chapter,
                activity,
                topic,
                score
            )

    except Exception as error:

        print(
            "Progress error:",
            error
        )


# =========================================================
# LEARN
# =========================================================

if mode == "📘 Learn":

    st.header(
        "📘 Learn"
    )

    st.info(
        "Ask a question from the selected NCERT Biology chapter."
    )

    question = st.text_input(
        "Ask a Biology question",
        placeholder="Example: Explain fertilisation."
    )

    if st.button(
        "Get Answer",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching NCERT..."
            ):

                try:

                    response = answer_question(
                        question,
                        selected_class,
                        chapter
                    )

                    if isinstance(
                        response,
                        tuple
                    ):

                        answer = response[0]

                        docs = (
                            response[1]
                            if len(response) > 1
                            else []
                        )

                    else:

                        answer = response
                        docs = []

                except Exception as error:

                    st.error(
                        f"Unable to generate answer: {error}"
                    )

                    answer = None
                    docs = []

            if answer:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "Answer"
                    )

                    st.write(
                        answer
                    )

                if docs:

                    with st.expander(
                        "📚 Retrieved NCERT Evidence"
                    ):

                        for index, doc in enumerate(
                            docs,
                            start=1
                        ):

                            metadata = getattr(
                                doc,
                                "metadata",
                                {}
                            )

                            source = metadata.get(
                                "source_file",
                                "NCERT Biology"
                            )

                            page = metadata.get(
                                "page",
                                ""
                            )

                            caption = source

                            if isinstance(
                                page,
                                int
                            ):

                                caption += (
                                    f" • Page {page + 1}"
                                )

                            st.caption(
                                caption
                            )

                            st.write(
                                getattr(
                                    doc,
                                    "page_content",
                                    ""
                                )
                            )

                            if index != len(
                                docs
                            ):

                                st.divider()

                save_progress_safely(
                    "Learn",
                    question
                )


# =========================================================
# PRACTICE / PYQs
# =========================================================

elif mode == "🚀 Practice / PYQs":

    st.header(
        "🚀 Practice / Previous Year Questions"
    )

    st.caption(
        "All available previous-year questions for the "
        "selected chapter, grouped by marks."
    )

    if selected_class != "Class 12":

        st.info(
            "Previous-year CBSE board questions are "
            "currently available for Class 12 Biology."
        )

    else:

        st.markdown(
            f"""
            <div class="chapter-box">
                <b>Selected Chapter:</b> {chapter}
                &nbsp;&nbsp; • &nbsp;&nbsp;
                Class 12 • Biology
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            questions = get_questions_for_chapter(
                selected_class,
                chapter
            )

        except Exception as error:

            st.error(
                f"Unable to load PYQs: {error}"
            )

            questions = []

        if not questions:

            st.warning(
                "No previous-year questions were found "
                "for this chapter."
            )

        else:

            grouped = group_questions_by_marks(
                questions
            )

            columns = st.columns(
                6
            )

            metrics = [
                (
                    "Total Questions",
                    len(questions)
                ),
                (
                    "MCQ / 1 Mark",
                    len(grouped["1"])
                ),
                (
                    "2 Mark",
                    len(grouped["2"])
                ),
                (
                    "3 Mark",
                    len(grouped["3"])
                ),
                (
                    "4 Mark",
                    len(grouped["4"])
                ),
                (
                    "5 Mark",
                    len(grouped["5"])
                )
            ]

            for column, (
                label,
                value
            ) in zip(
                columns,
                metrics
            ):

                with column:

                    st.metric(
                        label,
                        value
                    )

            st.write("")


            # =================================================
            # PYQ DISPLAY FUNCTION
            # =================================================

            def show_question_group(
                title,
                items,
                expanded=False
            ):

                with st.expander(
                    f"{title} ({len(items)})",
                    expanded=expanded
                ):

                    if not items:

                        st.info(
                            "No questions available "
                            "in this category."
                        )

                        return

                    for index, item in enumerate(
                        items,
                        start=1
                    ):

                        # A single malformed PYQ record
                        # (e.g. from bad PDF extraction)
                        # should not crash the whole page.
                        try:

                            year = item.get(
                                "year",
                                ""
                            )

                            number = item.get(
                                "question_number",
                                ""
                            )

                            qtype = item.get(
                                "question_type",
                                ""
                            )

                            question_text = item.get(
                                "question",
                                ""
                            )

                            options = item.get(
                                "options",
                                []
                            )

                            with st.container(
                                border=True
                            ):

                                meta = []

                                if year:

                                    meta.append(
                                        str(year)
                                    )

                                if number:

                                    meta.append(
                                        f"Q{number}"
                                    )

                                if qtype:

                                    meta.append(
                                        qtype
                                    )

                                if meta:

                                    st.caption(
                                        " • ".join(
                                            meta
                                        )
                                    )

                                st.markdown(
                                    f"**{index}. {question_text}**"
                                )

                                if isinstance(
                                    options,
                                    list
                                ):

                                    for option_index, option in enumerate(
                                        options
                                    ):

                                        option_text = str(
                                            option
                                        ).strip()

                                        if not option_text:
                                            continue

                                        letter = chr(
                                            65 + option_index
                                        )

                                        st.write(
                                            f"{letter}) {option_text}"
                                        )

                        except Exception as error:

                            st.warning(
                                f"Skipped a question record "
                                f"that could not be displayed: "
                                f"{error}"
                            )

                            continue

            show_question_group(
                "✅ MCQ / 1 Mark Questions",
                grouped["1"],
                True
            )

            show_question_group(
                "✏️ 2 Mark Questions",
                grouped["2"]
            )

            show_question_group(
                "🔖 3 Mark Questions",
                grouped["3"]
            )

            show_question_group(
                "⭐ 4 Mark Questions",
                grouped["4"]
            )

            show_question_group(
                "🏆 5 Mark Questions",
                grouped["5"]
            )

            if grouped.get(
                "Unknown"
            ):

                show_question_group(
                    "Other / Marks Not Available",
                    grouped[
                        "Unknown"
                    ]
                )


# =========================================================
# EVALUATE MY ANSWER
# =========================================================

elif mode == "✅ Evaluate My Answer":

    st.header(
        "✅ Evaluate My Answer"
    )

    st.caption(
        "BioAssist generates an NCERT-grounded question. "
        "Answer it and receive personalized formative feedback."
    )

    st.markdown(
        f"""
        <div class="chapter-box">
            <b>Selected Chapter:</b> {chapter}
            &nbsp;&nbsp; • &nbsp;&nbsp;
            {selected_class} • Biology
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # QUESTION LEVEL
    # =====================================================

    question_level = st.selectbox(
        "Choose Question Level",
        [
            "MCQ / 1 Mark",
            "2 Mark",
            "3 Mark",
            "4 Mark",
            "5 Mark"
        ],
        index=2
    )


    # =====================================================
    # RESET IF LEVEL CHANGES
    # =====================================================

    if (
        st.session_state.evaluation_level
        != question_level
    ):

        st.session_state.evaluation_level = (
            question_level
        )

        st.session_state.evaluation_question_data = None
        st.session_state.evaluation_result = None
        st.session_state.evaluation_question_id += 1


    # =====================================================
    # GENERATE QUESTION
    # =====================================================

    generate_label = (
        "Generate Question"
        if not st.session_state.evaluation_question_data
        else "Generate Another Question"
    )

    if st.button(
        generate_label,
        type="primary",
        key="generate_evaluation_question"
    ):

        previous_question = ""

        if st.session_state.evaluation_question_data:

            previous_question = (
                st.session_state
                .evaluation_question_data
                .get(
                    "question",
                    ""
                )
            )

        with st.spinner(
            "Generating NCERT-grounded question..."
        ):

            try:

                generated = (
                    generate_evaluation_question(
                        selected_class,
                        chapter,
                        question_level,
                        previous_question=previous_question
                    )
                )

            except Exception as error:

                generated = {
                    "question": "",
                    "options": [],
                    "answer": "",
                    "error":
                        str(error)
                }

        if generated.get(
            "error"
        ):

            st.error(
                generated["error"]
            )

        elif generated.get(
            "question"
        ):

            st.session_state.evaluation_question_id += 1

            st.session_state.evaluation_question_data = (
                generated
            )

            st.session_state.evaluation_result = None


    # =====================================================
    # DISPLAY QUESTION
    # =====================================================

    question_data = (
        st.session_state.evaluation_question_data
    )

    if question_data:

        generated_question = question_data.get(
            "question",
            ""
        )

        options = question_data.get(
            "options",
            []
        )

        expected_answer = question_data.get(
            "answer",
            ""
        )

        question_id = (
            st.session_state.evaluation_question_id
        )

        st.divider()

        st.subheader(
            f"📝 {question_level} Question"
        )

        with st.container(
            border=True
        ):

            st.markdown(
                f"**{generated_question}**"
            )


        # =================================================
        # MCQ INPUT
        # =================================================

        if (
            question_level == "MCQ / 1 Mark"
            and isinstance(
                options,
                list
            )
            and options
        ):

            radio_options = []

            for index, option in enumerate(
                options
            ):

                option_text = str(
                    option
                ).strip()

                if not option_text:
                    continue

                letter = chr(
                    65 + index
                )

                radio_options.append(
                    f"{letter}) {option_text}"
                )

            selected_option = st.radio(
                "Choose Your Answer",
                options=radio_options,
                index=None,
                key=f"evaluation_mcq_{question_id}"
            )

            if st.button(
                "Evaluate My Answer",
                type="primary",
                key=f"evaluate_mcq_{question_id}"
            ):

                if not selected_option:

                    st.warning(
                        "Please select an option."
                    )

                else:

                    with st.spinner(
                        "Evaluating your answer..."
                    ):

                        try:

                            evaluation = evaluate_answer(
                                generated_question,
                                selected_option,
                                selected_class,
                                chapter,
                                expected_answer=expected_answer,
                                question_level=question_level
                            )

                        except Exception as error:

                            st.error(
                                f"Evaluation failed: {error}"
                            )

                            evaluation = None

                    st.session_state.evaluation_result = (
                        evaluation
                    )

                    if evaluation:

                        save_progress_safely(
                            "Answer Evaluation",
                            generated_question,
                            evaluation.get(
                                "score",
                                0
                            )
                        )


        # =================================================
        # 2 / 3 / 4 / 5 MARK ANSWERS
        # =================================================

        else:

            with st.form(
                key=f"evaluation_form_{question_id}",
                clear_on_submit=False
            ):

                current_answer = st.text_area(
                    "Your Answer",
                    height=190,
                    placeholder=(
                        "Type your answer here. "
                        "You can edit it and submit again."
                    ),
                    key=f"evaluation_answer_{question_id}"
                )

                submitted = st.form_submit_button(
                    "Evaluate My Answer",
                    type="primary"
                )

            if submitted:

                if not current_answer.strip():

                    st.warning(
                        "Please enter your answer."
                    )

                else:

                    with st.spinner(
                        "Evaluating your answer..."
                    ):

                        try:

                            evaluation = evaluate_answer(
                                generated_question,
                                current_answer,
                                selected_class,
                                chapter,
                                expected_answer=expected_answer,
                                question_level=question_level
                            )

                        except Exception as error:

                            st.error(
                                f"Evaluation failed: {error}"
                            )

                            evaluation = None

                    st.session_state.evaluation_result = (
                        evaluation
                    )

                    if evaluation:

                        save_progress_safely(
                            "Answer Evaluation",
                            generated_question,
                            evaluation.get(
                                "score",
                                0
                            )
                        )


        # =================================================
        # RESULT
        # =================================================

        evaluation = (
            st.session_state.evaluation_result
        )

        if evaluation:

            score = evaluation.get(
                "score",
                0
            )

            exam_marks = evaluation.get(
                "exam_marks",
                0
            )

            max_marks = evaluation.get(
                "max_marks",
                1
            )

            st.divider()

            st.subheader(
                "📊 Your Result"
            )

            col1, col2, col3 = st.columns(
                3
            )

            with col1:

                st.metric(
                    "Performance Score",
                    f"{score}/10"
                )

            with col2:

                st.metric(
                    "Estimated Exam Marks",
                    f"{exam_marks}/{max_marks}"
                )

            with col3:

                if score >= 9:

                    st.success(
                        "Excellent"
                    )

                elif score >= 7:

                    st.success(
                        "Good"
                    )

                elif score >= 5:

                    st.warning(
                        "Needs Improvement"
                    )

                else:

                    st.error(
                        "Review Required"
                    )


            # =================================================
            # FEEDBACK
            # =================================================

            st.subheader(
                "🎯 Personalized Feedback"
            )

            correct_points = evaluation.get(
                "correct_points",
                []
            )

            missing_points = evaluation.get(
                "missing_points",
                []
            )

            missing_keywords = evaluation.get(
                "missing_keywords",
                []
            )

            improvement = evaluation.get(
                "improvement",
                ""
            )

            model_answer = evaluation.get(
                "model_answer",
                ""
            )


            # =================================================
            # CORRECT POINTS
            # =================================================

            with st.container(
                border=True
            ):

                st.markdown(
                    "#### ✅ What You Did Well"
                )

                if correct_points:

                    for point in correct_points:

                        st.write(
                            f"• {point}"
                        )

                else:

                    st.write(
                        "No major correct points "
                        "were identified."
                    )


            # =================================================
            # MISSING POINTS
            # =================================================

            with st.container(
                border=True
            ):

                st.markdown(
                    "#### ⚠️ What You Missed"
                )

                if missing_points:

                    for point in missing_points:

                        st.write(
                            f"• {point}"
                        )

                else:

                    st.success(
                        "No major NCERT points are missing."
                    )


            # =================================================
            # KEYWORDS
            # =================================================

            with st.container(
                border=True
            ):

                st.markdown(
                    "#### 🔑 Important Keywords"
                )

                if missing_keywords:

                    if isinstance(
                        missing_keywords,
                        list
                    ):

                        st.write(
                            ", ".join(
                                str(keyword)
                                for keyword
                                in missing_keywords
                            )
                        )

                    else:

                        st.write(
                            str(
                                missing_keywords
                            )
                        )

                else:

                    st.success(
                        "Important terminology is covered."
                    )


            # =================================================
            # IMPROVEMENT
            # =================================================

            with st.container(
                border=True
            ):

                st.markdown(
                    "#### 💡 How You Can Improve"
                )

                if improvement:

                    st.write(
                        improvement
                    )

                else:

                    st.write(
                        "Your answer already covers "
                        "the important concepts well."
                    )


            # =================================================
            # MODEL ANSWER
            # =================================================

            with st.expander(
                "📝 View Suggested NCERT-Aligned Answer"
            ):

                if model_answer:

                    st.write(
                        model_answer
                    )

                else:

                    st.write(
                        expected_answer
                    )


# =========================================================
# MY PROGRESS
# =========================================================

elif mode == "📊 My Progress":

    st.header(
        "📊 My Progress"
    )

    st.caption(
        "Track your BioAssist learning and "
        "answer-evaluation performance."
    )

    try:

        progress = get_progress(
            student
        )

    except Exception as error:

        st.error(
            f"Unable to load progress: {error}"
        )

        progress = []


    if not progress:

        st.info(
            "No learning activity has been recorded yet."
        )

    else:

        # =================================================
        # SUMMARY
        # =================================================

        evaluation_scores = []

        for row in progress:

            try:

                if isinstance(
                    row,
                    dict
                ):

                    activity = row.get(
                        "Activity",
                        row.get(
                            "activity",
                            ""
                        )
                    )

                    score = row.get(
                        "Score",
                        row.get(
                            "score"
                        )
                    )

                else:

                    activity = row[2]
                    score = row[4]

                if (
                    activity == "Answer Evaluation"
                    and score is not None
                ):

                    evaluation_scores.append(
                        float(
                            score
                        )
                    )

            except Exception:

                continue


        questions_attempted = len(
            evaluation_scores
        )

        average_score = (
            sum(
                evaluation_scores
            ) /
            questions_attempted
            if questions_attempted
            else 0
        )

        best_score = (
            max(
                evaluation_scores
            )
            if evaluation_scores
            else 0
        )


        summary1, summary2, summary3 = st.columns(
            3
        )


        with summary1:

            st.metric(
                "Questions Attempted",
                questions_attempted
            )


        with summary2:

            st.metric(
                "Average Score",
                (
                    f"{average_score:.1f}/10"
                    if questions_attempted
                    else "-"
                )
            )


        with summary3:

            st.metric(
                "Best Score",
                (
                    f"{best_score:g}/10"
                    if questions_attempted
                    else "-"
                )
            )


        st.divider()


        # =================================================
        # DETAILED HISTORY
        # =================================================

        st.subheader(
            "Recent Activity"
        )


        if (
            isinstance(
                progress,
                list
            )
            and progress
            and isinstance(
                progress[0],
                dict
            )
        ):

            st.dataframe(
                progress,
                use_container_width=True,
                hide_index=True
            )

        else:

            rows = []

            for row in progress:

                try:

                    rows.append(
                        {
                            "Class":
                                row[0],

                            "Chapter":
                                row[1],

                            "Activity":
                                row[2],

                            "Topic / Question":
                                row[3],

                            "Score":
                                row[4],

                            "Date":
                                row[5]
                        }
                    )

                except Exception:

                    continue

            if rows:

                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No progress records are available."
                )