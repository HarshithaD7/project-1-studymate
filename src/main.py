import os
import re
import random
import html

import pandas as pd
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
    group_questions_by_marks,
    get_question_counts
)

from progress_tracker import (
    save_progress,
    get_progress,
    suggest_explanation_level
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

    :root {
        /* Forest-green primary, gold/amber for real-exam content,
           blue for informational sections -- matches the approved
           mockup palette. */
        --accent: #22c55e;
        --accent-soft: rgba(34, 197, 94, 0.14);
        --accent-border: rgba(34, 197, 94, 0.40);
        --success: #22c55e;
        --success-soft: rgba(34, 197, 94, 0.14);
        --success-border: rgba(34, 197, 94, 0.38);
        --warning: #f0b429;
        --warning-soft: rgba(240, 180, 41, 0.14);
        --warning-border: rgba(240, 180, 41, 0.38);
        --purple: #3b82f6;
        --purple-soft: rgba(59, 130, 246, 0.14);
        --purple-border: rgba(59, 130, 246, 0.36);
        --gold: #eab308;
        --gold-soft: rgba(234, 179, 8, 0.14);
        --gold-border: rgba(234, 179, 8, 0.45);
        --line: rgba(255, 255, 255, 0.12);
        --surface: rgba(34, 197, 94, 0.04);
        --shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
        --bg: #0a0f0c;
        --bg-card: #0f1712;
    }

    .stApp {
        background: var(--bg);
    }

    /* Streamlit's own header/toolbar is a fixed, blurred bar (it
       ships with backdrop-filter + a translucent background) that
       was only being re-colored before -- recoloring alone left
       the blur active, which "ghosted" a blurred duplicate of the
       page title underneath it whenever the page scrolled. This
       app doesn't need Streamlit's chrome at all, so remove it
       outright instead of trying to blend it. */

    header[data-testid="stHeader"] {
        display: none;
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--line);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    /* ===================================================
       FORCE THE GREEN THEME ONTO NATIVE WIDGETS
       -------------------------------------------------
       Streamlit/BaseWeb draws its own radio dots, checkboxes
       and primary buttons in JS using whatever primaryColor
       the server resolved at startup (.streamlit/config.toml
       only loads on a full process restart, so it can lag
       behind this file). accent-color alone doesn't repaint
       BaseWeb's custom-drawn controls, so every rule below is
       !important and targets the actual painted elements --
       this keeps colors correct on every script rerun, no
       server restart required.
       =================================================== */

    input[type="radio"],
    input[type="checkbox"] {
        accent-color: var(--accent) !important;
    }

    /* BaseWeb's own radio mark is two nested plain <div>s inside
       the label (outer ring, inner dot) with no stable class name
       to hook -- and its color is repainted by React from
       primaryColor, not real CSS, so trying to override those two
       divs directly is fighting a moving target. Instead: hide
       BaseWeb's mark entirely and draw one clean dot ourselves via
       ::before, driven purely by :has(input:checked) on the label
       (already proven reliable -- it's what colors the card
       background/border on selection). */

    div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
        visibility: hidden;
        width: 1rem;
        position: relative;
    }

    div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child::before {
        content: "";
        visibility: visible;
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 1rem;
        height: 1rem;
        box-sizing: border-box;
        border-radius: 50%;
        border: 2px solid rgba(255, 255, 255, 0.35);
        background: transparent;
    }

    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) > div:first-child::before {
        border-color: var(--accent);
        background: var(--accent);
        box-shadow: inset 0 0 0 3px var(--bg-card);
    }

    button[kind="primary"],
    button[kind="primaryFormSubmit"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #062712 !important;
    }

    button[kind="primary"]:hover,
    button[kind="primaryFormSubmit"]:hover {
        background-color: #16a34a !important;
        border-color: #16a34a !important;
        color: #062712 !important;
    }

    button[kind="secondary"] {
        border-color: var(--line) !important;
    }

    button[kind="secondary"]:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextArea textarea {
        border-color: var(--line) !important;
        background: var(--bg-card) !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    .app-header {
        padding-bottom: 0.9rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid var(--accent-border);
    }

    .app-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: normal;
        line-height: 1.3;
        margin-bottom: 0.3rem;
    }

    .app-subtitle {
        opacity: 0.70;
        font-size: 0.94rem;
        line-height: 1.4;
    }

    .chapter-box {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.85rem 1rem;
        margin-bottom: 1.2rem;
        border-radius: 10px;
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        background: var(--surface);
        box-shadow: var(--shadow);
        font-size: 0.92rem;
    }

    section[data-testid="stSidebar"] {
        min-width: 265px;
        max-width: 265px;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
        box-shadow: var(--shadow);
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 2.5rem;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.7rem 0.9rem 0.6rem 0.9rem;
        box-shadow: var(--shadow);
    }

    footer {
        visibility: hidden;
    }

    .source-badge {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        margin-bottom: 0.6rem;
    }

    .source-badge.pyq {
        background: var(--success-soft);
        color: var(--success);
    }

    .source-badge.ai {
        background: var(--warning-soft);
        color: var(--warning);
    }

    .level-ladder {
        margin: 0.6rem 0 1.1rem 0;
    }

    .level-step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.32rem 0.85rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.4rem;
        border: 1px solid var(--line);
        opacity: 0.45;
    }

    .level-step.active {
        opacity: 1;
        background: var(--accent-soft);
        border-color: var(--accent-border);
        color: var(--accent);
    }

    .level-step.done {
        opacity: 0.85;
        background: var(--success-soft);
        border-color: var(--success-border);
        color: var(--success);
    }

    .suggested-level-box {
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        background: var(--accent-soft);
        border: 1px solid var(--accent-border);
        box-shadow: var(--shadow);
        font-size: 0.82rem;
        margin-top: 0.8rem;
    }

    .sidebar-caption {
        opacity: 0.6;
        font-size: 0.78rem;
        margin-top: -0.6rem;
        margin-bottom: 0.8rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.25rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.4rem 0.6rem;
        border-radius: 8px;
        transition: background 0.15s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: var(--surface);
    }

    .feedback-card {
        border-radius: 10px;
        border: 1px solid var(--line);
        background: var(--surface);
        box-shadow: var(--shadow);
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
    }

    .feedback-card.positive { border-left: 4px solid var(--success); }
    .feedback-card.negative { border-left: 4px solid var(--warning); }
    .feedback-card.keywords { border-left: 4px solid var(--purple); }
    .feedback-card.improve  { border-left: 4px solid var(--accent); }

    .feedback-card .fc-title {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .feedback-card ul {
        margin: 0;
        padding-left: 1.15rem;
    }

    .feedback-card li {
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        line-height: 1.45;
    }

    .feedback-card p {
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }

    .empty-state {
        text-align: center;
        padding: 2.2rem 1rem;
        border-radius: 10px;
        border: 1px dashed var(--line);
        opacity: 0.75;
    }

    /* ===================================================
       SIDEBAR BRAND BLOCK
       =================================================== */

    .brand-block {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.1rem;
    }

    .brand-icon {
        width: 2.3rem;
        height: 2.3rem;
        border-radius: 50%;
        background: var(--accent-soft);
        border: 1px solid var(--accent-border);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        flex-shrink: 0;
    }

    .brand-name {
        font-weight: 800;
        font-size: 1.15rem;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.72rem;
        opacity: 0.6;
        line-height: 1.2;
    }

    .sidebar-section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        opacity: 0.55;
        text-transform: uppercase;
        margin: 1.1rem 0 0.4rem 0;
    }

    /* Restyle the Mode radio into full-width card buttons */

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.3rem;
        background: transparent;
        width: 100%;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: var(--accent-soft);
        border-color: var(--accent-border);
    }

    .session-card {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        margin-top: 1.2rem;
        background: rgba(255, 255, 255, 0.02);
    }

    /* Purely decorative botanical line-art, pinned to the
       bottom of the sidebar to match the approved mockup. */

    .sidebar-plant {
        margin-top: 2rem;
        opacity: 0.45;
        pointer-events: none;
        line-height: 0;
    }

    .sidebar-plant svg {
        width: 90px;
        height: auto;
        display: block;
    }

    .session-card .sc-title {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        opacity: 0.55;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .session-card .sc-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        padding: 0.15rem 0;
    }

    .session-card .sc-row span:first-child {
        opacity: 0.6;
    }

    /* ===================================================
       PAGE HEADER (per-tab, e.g. "Practice")
       =================================================== */

    .page-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.1rem;
    }

    .page-header-left {
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }

    .page-header-icon {
        width: 2.4rem;
        height: 2.4rem;
        border-radius: 10px;
        background: var(--surface);
        border: 1px solid var(--line);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }

    .page-header-title {
        font-size: 1.3rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .page-header-subtitle {
        font-size: 0.82rem;
        opacity: 0.6;
    }

    /* ===================================================
       CHAPTER INFO BAR
       =================================================== */

    .chapter-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: var(--bg-card);
        box-shadow: var(--shadow);
    }

    .chapter-bar-title {
        font-weight: 700;
        font-size: 1.02rem;
    }

    .chapter-bar-tags {
        font-size: 0.78rem;
        opacity: 0.6;
        margin-top: 0.15rem;
    }

    .ncert-badge {
        text-align: right;
        font-size: 0.78rem;
        color: var(--accent);
        font-weight: 700;
    }

    .ncert-badge .nb-label {
        display: block;
        opacity: 0.55;
        font-weight: 500;
        font-size: 0.7rem;
        color: var(--warning);
    }

    /* ===================================================
       LEVEL PILL ROW (main content, mirrors sidebar choice)
       =================================================== */

    .level-pill-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }

    .level-pill {
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        border: 1px solid var(--line);
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.65;
    }

    .level-pill.active {
        opacity: 1;
        background: var(--accent);
        color: #062712;
        border-color: var(--accent);
    }

    /* ===================================================
       QUESTION CARD
       =================================================== */

    .question-card {
        border: 1px solid var(--gold-border);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, rgba(234,179,8,0.06), transparent 60%);
    }

    .question-card.ai {
        border-color: var(--warning-border);
        background: linear-gradient(180deg, rgba(240,180,41,0.06), transparent 60%);
    }

    .question-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.7rem;
    }

    .question-card-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.3px;
        background: var(--gold-soft);
        color: var(--gold);
        border: 1px solid var(--gold-border);
    }

    .question-card.ai .question-card-badge {
        background: var(--warning-soft);
        color: var(--warning);
        border-color: var(--warning-border);
    }

    .question-card-meta {
        display: flex;
        gap: 0.4rem;
    }

    .meta-chip {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        border: 1px solid var(--line);
        opacity: 0.85;
    }

    .question-card-text {
        font-size: 1.02rem;
        font-weight: 600;
        line-height: 1.5;
    }

    /* ===================================================
       RESULT: SCORE RING + STAT CARDS
       =================================================== */

    .result-panel {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        background: var(--bg-card);
        margin-bottom: 0.8rem;
    }

    .score-ring {
        width: 5.2rem;
        height: 5.2rem;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background: conic-gradient(var(--accent) calc(var(--pct) * 1%), rgba(255,255,255,0.08) 0);
    }

    .score-ring-inner {
        width: 4.1rem;
        height: 4.1rem;
        border-radius: 50%;
        background: var(--bg-card);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.05rem;
        line-height: 1.1;
    }

    .score-ring-inner span {
        font-size: 0.62rem;
        font-weight: 600;
        opacity: 0.6;
    }

    .result-headline {
        font-size: 1.05rem;
        font-weight: 800;
    }

    .result-subtext {
        font-size: 0.85rem;
        opacity: 0.7;
        margin: 0.15rem 0 0.4rem 0;
    }

    .mini-stat-row {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }

    .mini-stat {
        flex: 1;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        background: var(--bg-card);
    }

    .mini-stat .ms-value {
        font-size: 1.1rem;
        font-weight: 800;
    }

    .mini-stat.positive .ms-value { color: var(--success); }
    .mini-stat.negative .ms-value { color: var(--warning); }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }

    .chip {
        display: inline-block;
        padding: 0.22rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background: var(--purple-soft);
        color: var(--purple);
        border: 1px solid var(--purple-border);
    }

    .model-answer-card {
        border: 1px solid var(--purple-border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        background: rgba(59, 130, 246, 0.05);
        margin-bottom: 0.8rem;
    }

    .model-answer-card .ma-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--purple);
        margin-bottom: 0.5rem;
    }

    .model-answer-card p {
        font-size: 0.88rem;
        line-height: 1.55;
        opacity: 0.9;
    }

    /* ===================================================
       CLARITY / RE-TEACH BAR
       =================================================== */

    .clarity-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.8rem;
        border: 1px solid var(--gold-border);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        background: rgba(234, 179, 8, 0.05);
        margin-top: 1rem;
    }

    .clarity-bar-left {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 0.85rem;
    }

    .clarity-bar-left b {
        display: block;
        font-size: 0.95rem;
    }

    .ladder-steps {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .ladder-step {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        font-size: 0.78rem;
        border: 1px solid var(--line);
        opacity: 0.5;
    }

    .ladder-step .ls-num {
        width: 1.3rem;
        height: 1.3rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 800;
        background: rgba(255,255,255,0.08);
    }

    .ladder-step.done {
        opacity: 0.9;
        border-color: var(--success-border);
    }

    .ladder-step.done .ls-num {
        background: var(--success);
        color: #062712;
    }

    .ladder-step.active {
        opacity: 1;
        border-color: var(--gold-border);
        background: var(--gold-soft);
    }

    .ladder-step.active .ls-num {
        background: var(--gold);
        color: #2b2103;
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
    "practice_level": "3 Mark",
    "practice_question_data": None,
    "practice_question_id": 0,
    "practice_result": None,
    "practice_stage": "attempt",
    "practice_explanation_level": "Class 10",
    "practice_explanation_cache_key": None,
    "practice_explanation_text": "",
    "practice_explanation_docs": [],
    "practice_last_pyq_key": None,
    "last_practice_context": None,
    "questions_attempted_session": 0
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR
#
# Branding moved here (icon + name + tagline) to match the
# approved UI redesign -- the main content area now opens
# directly with a compact per-page header instead of a large
# top banner.
# =========================================================

def _on_level_change():

    # Widget-bound key ("practice_level") already updated by
    # Streamlit at this point -- just clear out the question
    # that belonged to the previous level so a stale question
    # never shows under the new tab.

    st.session_state.practice_question_data = None
    st.session_state.practice_result = None
    st.session_state.practice_question_id += 1
    st.session_state.practice_stage = "attempt"


with st.sidebar:

    st.markdown(
        """
        <div class="brand-block">
            <div class="brand-icon">🌿</div>
            <div>
                <div class="brand-name">BioAssist AI</div>
                <div class="brand-subtitle">NCERT-grounded Biology Practice</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sidebar-section-label'>Chapter</div>",
        unsafe_allow_html=True
    )

    # BioAssist is currently scoped to Class 12 Biology only.
    selected_class = "Class 12"

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
        chapters,
        label_visibility="collapsed"
    )

    st.markdown(
        f"<div class='sidebar-caption'>{len(chapters)} chapters available</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sidebar-section-label'>Mode</div>",
        unsafe_allow_html=True
    )

    mode = st.radio(
        "Mode",
        [
            "✏️ Practice",
            "📊 My Progress"
        ],
        label_visibility="collapsed"
    )

    if mode == "✏️ Practice":

        st.markdown(
            "<div class='sidebar-section-label'>Question Level</div>",
            unsafe_allow_html=True
        )

        st.radio(
            "Question Level",
            [
                "MCQ",
                "1 Mark",
                "2 Mark",
                "3 Mark",
                "4 Mark",
                "5 Mark"
            ],
            key="practice_level",
            on_change=_on_level_change,
            label_visibility="collapsed"
        )

    # Chapter files/labels are prefixed with their chapter number
    # (e.g. "1. Sexual Reproduction in Flowering Plants") -- pull
    # just that number for the compact session-card row, matching
    # the approved mockup. Falls back to the full name if a chapter
    # is ever stored without a numeric prefix.
    chapter_number_match = re.match(
        r"\s*(\d+)\s*[\.\)]",
        chapter
    )

    chapter_number = (
        chapter_number_match.group(1)
        if chapter_number_match
        else chapter
    )

    st.markdown(
        f"""
        <div class="session-card">
            <div class="sc-title">Current Session</div>
            <div class="sc-row"><span>Class</span><span>12</span></div>
            <div class="sc-row"><span>Subject</span><span>Biology</span></div>
            <div class="sc-row"><span>Chapter</span><span>{html.escape(chapter_number)}</span></div>
            <div class="sc-row"><span>Questions Attempted</span><span>{st.session_state.questions_attempted_session}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-plant" aria-hidden="true">
            <svg viewBox="0 0 100 160" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round">
                <path d="M35 155 C35 110 35 70 40 20"/>
                <path d="M40 55 C55 50 65 38 68 22"/>
                <path d="M38 85 C22 82 12 70 10 55"/>
                <path d="M38 115 C55 111 63 98 65 82"/>
                <ellipse cx="68" cy="20" rx="11" ry="6" fill="#22c55e" stroke="none" transform="rotate(-25 68 20)"/>
                <ellipse cx="10" cy="53" rx="11" ry="6" fill="#22c55e" stroke="none" transform="rotate(25 10 53)"/>
                <ellipse cx="65" cy="80" rx="11" ry="6" fill="#22c55e" stroke="none" transform="rotate(-25 65 80)"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# RESET PRACTICE STATE ON CHAPTER CHANGE
# =========================================================

current_context = (
    selected_class,
    chapter
)

if (
    st.session_state.last_practice_context
    != current_context
):

    st.session_state.practice_question_data = None
    st.session_state.practice_result = None
    st.session_state.practice_question_id += 1
    st.session_state.practice_stage = "attempt"
    st.session_state.practice_last_pyq_key = None

    st.session_state.last_practice_context = (
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
# PRACTICE
#
# Core loop: attempt a REAL previous-year question at full
# Class 12 difficulty. If the student struggles, BioAssist
# explains the underlying concept at a simpler level (Class
# 10, then Class 8 if still needed), then lets the student
# retry at Class 12 level. Real PYQs are always clearly
# distinguished from AI-generated fallback questions.
# =========================================================

if mode == "✏️ Practice":

    st.markdown(
        """
        <div class="page-header">
            <div class="page-header-left">
                <div class="page-header-icon">🚀</div>
                <div>
                    <div class="page-header-title">Practice</div>
                    <div class="page-header-subtitle">
                        Real previous-year questions, re-taught simply if you're struggling
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:

        chapter_counts = get_question_counts(
            selected_class,
            chapter
        )

        if chapter_counts.get("total"):

            tags_line = (
                f"{chapter_counts['total']} previous-year questions available &nbsp;•&nbsp; "
                f"MCQ {chapter_counts['1']} · "
                f"2M {chapter_counts['2']} · "
                f"3M {chapter_counts['3']} · "
                f"4M {chapter_counts['4']} · "
                f"5M {chapter_counts['5']}"
            )

        else:

            tags_line = (
                "No previous-year questions indexed yet — "
                "AI-generated practice questions will be used, clearly labelled as such."
            )

    except Exception:

        tags_line = "Class 12 • Biology"

    st.markdown(
        f"""
        <div class="chapter-bar">
            <div>
                <div class="chapter-bar-title">📘 {html.escape(chapter)}</div>
                <div class="chapter-bar-tags">{tags_line}</div>
            </div>
            <div class="ncert-badge">
                <span class="nb-label">Grounded in</span>
                NCERT Class 12
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # QUESTION LEVEL
    #
    # The actual widget now lives in the sidebar (see
    # "Question Level" radio, bound to session_state directly
    # via key="practice_level"). Read its committed value here.
    #
    # internal_level exists because "MCQ / 1 Mark" is a single
    # load-bearing string used throughout question_generator.py
    # and answer_evaluator.py to trigger deterministic MCQ
    # handling. Splitting the UI into separate "MCQ" and
    # "1 Mark" tabs must not disturb that -- so whenever the
    # student is on the MCQ tab, we still pass the exact legacy
    # string internally. The new "1 Mark" tab passes through as
    # its own value and is handled by dedicated "1 Mark" rules
    # added to get_level_instruction()/get_depth_instruction().
    # =====================================================

    practice_level = st.session_state.practice_level

    internal_level = (
        "MCQ / 1 Mark"
        if practice_level == "MCQ"
        else practice_level
    )

    # Visual-only tab row mirroring the sidebar's Question Level
    # selection -- the sidebar radio (key="practice_level") is
    # still the real widget/source of truth, so this is a plain
    # <span> readout, not another Streamlit widget, to avoid any
    # duplicate-key or widget-state conflicts.

    pills_html = "".join(
        f'<span class="level-pill{" active" if lvl == practice_level else ""}">{lvl}</span>'
        for lvl in ["MCQ", "1 Mark", "2 Mark", "3 Mark", "4 Mark", "5 Mark"]
    )

    st.markdown(
        f'<div class="level-pill-row">{pills_html}</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # FETCH A QUESTION: REAL PYQ FIRST, AI FALLBACK SECOND
    # =====================================================

    LEVEL_KEY_MAP = {
        "MCQ": "1",
        "1 Mark": "1",
        "2 Mark": "2",
        "3 Mark": "3",
        "4 Mark": "4",
        "5 Mark": "5"
    }

    # Diagram-based PYQs (e.g. "study the diagram given below")
    # reference an image that the current text-only extraction
    # pipeline does not capture, and evaluation runs on a
    # text-only LLM that cannot judge a labelled diagram anyway.
    # They stay in pyq_questions.json (so PYQ coverage counts
    # for the chapter remain accurate) but are excluded from
    # the pool a student can actually be served to attempt.

    def fetch_practice_question(level):

        try:

            all_questions = get_questions_for_chapter(
                selected_class,
                chapter
            )

            attemptable_questions = [
                item
                for item in all_questions
                if str(
                    item.get("question_type", "")
                ).strip()
                != "Diagram Based"
            ]

            grouped = group_questions_by_marks(
                attemptable_questions
            )

        except Exception:

            grouped = {}

        pool = (
            grouped.get(
                LEVEL_KEY_MAP.get(level, ""),
                []
            )
            if grouped
            else []
        )

        # The "1" marks bucket contains both real MCQs and
        # plain 1-mark short-answer questions mixed together.
        # Now that MCQ and 1 Mark are separate tabs, split the
        # bucket by the record's actual question_type so each
        # tab only ever gets the kind of question it promises.

        if level == "MCQ":

            pool = [
                item
                for item in pool
                if str(item.get("question_type", "")).strip()
                == "MCQ"
            ]

        elif level == "1 Mark":

            pool = [
                item
                for item in pool
                if str(item.get("question_type", "")).strip()
                != "MCQ"
            ]

        if pool:

            last_key = (
                st.session_state.practice_last_pyq_key
            )

            candidates = [
                item
                for item in pool
                if (
                    item.get("source_file", ""),
                    item.get("question_number", "")
                )
                != last_key
            ] or pool

            picked = random.choice(
                candidates
            )

            st.session_state.practice_last_pyq_key = (
                picked.get("source_file", ""),
                picked.get("question_number", "")
            )

            return {
                "source": "pyq",
                "question": picked.get("question", ""),
                "options": picked.get("options", []) or [],
                # A handful of PYQ source papers explicitly
                # state their answer key ("Correct Answer:
                # (X) ..."), captured at build time into
                # answer_key. When present, this is a real,
                # verified answer -- so it's used exactly
                # like an AI-generated question's answer,
                # routing evaluation through the deterministic
                # letter/text-match path instead of the
                # unverified-MCQ LLM fallback. Most PYQs still
                # won't have this and correctly fall back.
                "expected_answer":
                    picked.get("answer_key", "") or "",
                "meta": {
                    "year": picked.get("year", ""),
                    "question_number":
                        picked.get("question_number", "")
                }
            }

        # No real PYQ available at this level for this
        # chapter. Fall back to a clearly labelled
        # AI-generated practice question.

        previous_question = ""

        if st.session_state.practice_question_data:

            previous_question = (
                st.session_state
                .practice_question_data
                .get("question", "")
            )

        # Same MCQ / 1 Mark translation as internal_level above,
        # applied to whatever level this function was called
        # with (mirrors the outer practice_level in every real
        # call site, but computed locally so this function is
        # correct on its own terms).
        generated = generate_evaluation_question(
            selected_class,
            chapter,
            "MCQ / 1 Mark" if level == "MCQ" else level,
            previous_question=previous_question
        )

        if generated.get("error"):

            return {
                "error": generated["error"]
            }

        return {
            "source": "ai_generated",
            "question": generated.get("question", ""),
            "options": generated.get("options", []) or [],
            "expected_answer": generated.get("answer", ""),
            "meta": {}
        }


    def load_new_practice_question(level):
        """
        Fetches a question and updates session state in one
        step. Returns True on success so callers can decide
        whether to force an immediate rerun.
        """

        with st.spinner(
            "Finding a question..."
        ):

            try:

                result = fetch_practice_question(
                    level
                )

            except Exception as error:

                result = {
                    "error": str(error)
                }

        if result.get("error"):

            st.error(
                result["error"]
            )

            return False

        if result.get("question"):

            st.session_state.practice_question_id += 1
            st.session_state.practice_question_data = result
            st.session_state.practice_result = None
            st.session_state.practice_stage = "attempt"

            return True

        return False


    generate_label = (
        "Get Practice Question"
        if not st.session_state.practice_question_data
        else "Get Another Question"
    )

    if st.button(
        generate_label,
        type="primary",
        key="fetch_practice_question"
    ):

        load_new_practice_question(
            practice_level
        )


    # =====================================================
    # DISPLAY QUESTION + ATTEMPT
    # =====================================================

    question_data = (
        st.session_state.practice_question_data
    )

    if question_data:

        current_question = question_data.get(
            "question", ""
        )

        options = question_data.get(
            "options", []
        )

        expected_answer = question_data.get(
            "expected_answer", ""
        )

        source = question_data.get(
            "source", "pyq"
        )

        meta = question_data.get(
            "meta", {}
        )

        question_id = (
            st.session_state.practice_question_id
        )

        if source == "pyq":

            meta_bits = [
                practice_level
            ]

            if meta.get("year"):
                meta_bits.append(str(meta["year"]))

            if meta.get("question_number"):
                meta_bits.append(f"Q{meta['question_number']}")

            meta_chips_html = "".join(
                f'<span class="meta-chip">{html.escape(bit)}</span>'
                for bit in meta_bits
            )

            st.markdown(
                f"""
                <div class="question-card">
                    <div class="question-card-top">
                        <span class="question-card-badge">📄 PREVIOUS YEAR QUESTION</span>
                        <div class="question-card-meta">{meta_chips_html}</div>
                    </div>
                    <div class="question-card-text">{html.escape(current_question)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="question-card ai">
                    <div class="question-card-top">
                        <span class="question-card-badge">🤖 AI-GENERATED PRACTICE</span>
                        <div class="question-card-meta"><span class="meta-chip">{html.escape(practice_level)}</span></div>
                    </div>
                    <div class="question-card-text">{html.escape(current_question)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "No previous-year question was available at this level "
                "for this chapter, so this was generated by BioAssist. "
                "It is NOT an actual board exam question."
            )


        # =================================================
        # MCQ INPUT
        # =================================================

        if (
            practice_level == "MCQ"
            and isinstance(options, list)
            and options
        ):

            radio_options = []

            for index, option in enumerate(options):

                option_text = str(option).strip()

                if not option_text:
                    continue

                letter = chr(65 + index)

                radio_options.append(
                    f"{letter}) {option_text}"
                )

            selected_option = st.radio(
                "Choose Your Answer",
                options=radio_options,
                index=None,
                key=f"practice_mcq_{question_id}"
            )

            if st.button(
                "Evaluate My Answer",
                type="primary",
                key=f"practice_evaluate_mcq_{question_id}"
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
                                current_question,
                                selected_option,
                                selected_class,
                                chapter,
                                expected_answer=expected_answer,
                                question_level=internal_level,
                                options=options
                            )

                        except Exception as error:

                            st.error(
                                f"Evaluation failed: {error}"
                            )

                            evaluation = None

                    st.session_state.practice_result = evaluation
                    st.session_state.practice_stage = "attempt"

                    if evaluation:

                        label = (
                            f"[PYQ] {current_question}"
                            if source == "pyq"
                            else f"[AI Practice] {current_question}"
                        )

                        save_progress_safely(
                            "Answer Evaluation",
                            label,
                            evaluation.get("score", 0)
                        )


        # =================================================
        # DESCRIPTIVE INPUT (2 / 3 / 4 / 5 MARK)
        # =================================================

        else:

            with st.form(
                key=f"practice_form_{question_id}",
                clear_on_submit=False
            ):

                current_answer = st.text_area(
                    "Your Answer",
                    height=190,
                    placeholder=(
                        "Type your answer here. "
                        "You can edit it and submit again."
                    ),
                    key=f"practice_answer_{question_id}"
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
                                current_question,
                                current_answer,
                                selected_class,
                                chapter,
                                expected_answer=expected_answer,
                                question_level=internal_level,
                                options=options
                            )

                        except Exception as error:

                            st.error(
                                f"Evaluation failed: {error}"
                            )

                            evaluation = None

                    st.session_state.practice_result = evaluation
                    st.session_state.practice_stage = "attempt"

                    if evaluation:

                        label = (
                            f"[PYQ] {current_question}"
                            if source == "pyq"
                            else f"[AI Practice] {current_question}"
                        )

                        save_progress_safely(
                            "Answer Evaluation",
                            label,
                            evaluation.get("score", 0)
                        )


        # =================================================
        # RESULT
        # =================================================

        evaluation = (
            st.session_state.practice_result
        )

        if evaluation:

            st.session_state.questions_attempted_session += 1

            score = evaluation.get("score", 0)
            exam_marks = evaluation.get("exam_marks", 0)
            max_marks = evaluation.get("max_marks", 1)
            pct = round(min(max(score, 0), 10) * 10)

            if score >= 9:
                headline = "Excellent"
            elif score >= 7:
                headline = "Good"
            elif score >= 5:
                headline = "Needs Improvement"
            else:
                headline = "Review Required"

            st.markdown(
                f"""
                <div class="result-panel" style="--pct:{pct}">
                    <div class="score-ring">
                        <div class="score-ring-inner">
                            {score}/10
                            <span>SCORE</span>
                        </div>
                    </div>
                    <div>
                        <div class="result-headline">{headline}</div>
                        <div class="result-subtext">
                            Estimated Exam Marks: {exam_marks}/{max_marks}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            # =============================================
            # FEEDBACK
            # =============================================

            correct_points = evaluation.get("correct_points", [])
            missing_points = evaluation.get("missing_points", [])
            missing_keywords = evaluation.get("missing_keywords", [])
            improvement = evaluation.get("improvement", "")
            model_answer = evaluation.get("model_answer", "")

            st.markdown(
                f"""
                <div class="mini-stat-row">
                    <div class="mini-stat positive">
                        <span>✅ Correct Points</span>
                        <span class="ms-value">{len(correct_points) if isinstance(correct_points, list) else 0}</span>
                    </div>
                    <div class="mini-stat negative">
                        <span>⚠️ Missing Points</span>
                        <span class="ms-value">{len(missing_points) if isinstance(missing_points, list) else 0}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            def render_feedback_card(
                title,
                css_class,
                items=None,
                text=None,
                empty_message=""
            ):

                if items:

                    list_html = "".join(
                        f"<li>{html.escape(str(point))}</li>"
                        for point in items
                    )

                    body_html = f"<ul>{list_html}</ul>"

                elif text:

                    body_html = f"<p>{html.escape(str(text))}</p>"

                else:

                    body_html = f"<p>{html.escape(empty_message)}</p>"

                st.markdown(
                    f"""
                    <div class="feedback-card {css_class}">
                        <div class="fc-title">{title}</div>
                        {body_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            render_feedback_card(
                "✅ What You Did Well",
                "positive",
                items=correct_points,
                empty_message="No major correct points were identified."
            )

            render_feedback_card(
                "⚠️ What You Missed",
                "negative",
                items=missing_points,
                empty_message="No major NCERT points are missing — nice work."
            )

            if missing_keywords:

                keyword_list = (
                    missing_keywords
                    if isinstance(missing_keywords, list)
                    else [missing_keywords]
                )

                chips_html = "".join(
                    f'<span class="chip">{html.escape(str(keyword))}</span>'
                    for keyword in keyword_list
                    if str(keyword).strip()
                )

                st.markdown(
                    f"""
                    <div class="feedback-card keywords">
                        <div class="fc-title">🔑 Important Keywords</div>
                        <div class="chip-row">{chips_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                render_feedback_card(
                    "🔑 Important Keywords",
                    "keywords",
                    empty_message="Important terminology is covered."
                )

            render_feedback_card(
                "💡 How You Can Improve",
                "improve",
                text=improvement,
                empty_message=(
                    "Your answer already covers the "
                    "important concepts well."
                )
            )

            st.markdown(
                f"""
                <div class="model-answer-card">
                    <div class="ma-title">📝 Suggested NCERT-Aligned Answer</div>
                    <p>{html.escape(model_answer or expected_answer or "Not available.")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


            # =============================================
            # PROGRESSIVE EXPLANATION
            #
            # Score >= 7: student is solid at Class 12 level
            # on this question -- move on.
            #
            # Score < 7: step DOWN to a simpler explanation
            # (Class 10, or Class 8 if really struggling),
            # then let the student climb back up to a fresh
            # Class 12 attempt once ready.
            # =============================================

            if score >= 7:

                st.markdown(
                    """
                    <div class="clarity-bar">
                        <div class="clarity-bar-left">
                            <span>✅</span>
                            <span>
                                <b>Solid at Class 12 level</b>
                                Ready for another question?
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button(
                    "Try Another Question",
                    key=f"practice_next_{question_id}"
                ):

                    load_new_practice_question(
                        practice_level
                    )

                    st.rerun()

            else:

                if st.session_state.practice_stage != "explain":

                    st.session_state.practice_stage = "explain"

                    st.session_state.practice_explanation_level = (
                        "Class 8"
                        if score < 4
                        else "Class 10"
                    )

                explanation_level = (
                    st.session_state.practice_explanation_level
                )

                def ladder_step_class(step_level):

                    if step_level == explanation_level:
                        return "ladder-step active"

                    step_order = [
                        "Class 8",
                        "Class 10",
                        "Class 12"
                    ]

                    if (
                        step_order.index(step_level)
                        < step_order.index(explanation_level)
                    ):
                        return "ladder-step done"

                    return "ladder-step"

                def ladder_step_num(step_level, index):

                    step_order = [
                        "Class 8",
                        "Class 10",
                        "Class 12"
                    ]

                    if step_order.index(step_level) < step_order.index(explanation_level):
                        return "✓"

                    return str(index)

                st.markdown(
                    f"""
                    <div class="clarity-bar">
                        <div class="clarity-bar-left">
                            <span>📘</span>
                            <span>
                                <b>Let's break this down</b>
                                Re-teaching at a simpler level, then back to Class 12
                            </span>
                        </div>
                        <div class="ladder-steps">
                            <span class="{ladder_step_class('Class 8')}"><span class="ls-num">{ladder_step_num('Class 8', 1)}</span>Class 8</span>
                            <span class="{ladder_step_class('Class 10')}"><span class="ls-num">{ladder_step_num('Class 10', 2)}</span>Class 10</span>
                            <span class="{ladder_step_class('Class 12')}"><span class="ls-num">3</span>Class 12 (Goal)</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader(
                    f"📘 {explanation_level} level"
                )

                explanation_cache_key = (
                    question_id,
                    explanation_level
                )

                if (
                    st.session_state.practice_explanation_cache_key
                    != explanation_cache_key
                ):

                    with st.spinner(
                        "Preparing a simpler explanation..."
                    ):

                        try:

                            response = answer_question(
                                current_question,
                                selected_class,
                                chapter,
                                explanation_level=explanation_level
                            )

                            if isinstance(response, tuple):
                                explanation_text = response[0]
                                evidence_docs = (
                                    response[1]
                                    if len(response) > 1
                                    else []
                                )
                            else:
                                explanation_text = response
                                evidence_docs = []

                        except Exception as error:

                            explanation_text = (
                                f"Unable to generate explanation: {error}"
                            )

                            evidence_docs = []

                    st.session_state.practice_explanation_cache_key = (
                        explanation_cache_key
                    )

                    st.session_state.practice_explanation_text = (
                        explanation_text
                    )

                    st.session_state.practice_explanation_docs = (
                        evidence_docs
                    )

                explanation_text = (
                    st.session_state.practice_explanation_text
                )

                evidence_docs = (
                    st.session_state.practice_explanation_docs
                )

                with st.container(border=True):

                    st.write(explanation_text)

                if evidence_docs:

                    with st.expander(
                        "📚 Retrieved NCERT Evidence"
                    ):

                        for index, doc in enumerate(
                            evidence_docs,
                            start=1
                        ):

                            metadata = getattr(
                                doc, "metadata", {}
                            )

                            doc_source = metadata.get(
                                "source_file", "NCERT Biology"
                            )

                            page = metadata.get("page", "")

                            caption = doc_source

                            if isinstance(page, int):
                                caption += f" • Page {page + 1}"

                            st.caption(caption)

                            st.write(
                                getattr(doc, "page_content", "")
                            )

                            if index != len(evidence_docs):
                                st.divider()

                col_a, col_b = st.columns(2)

                with col_a:

                    if explanation_level == "Class 8":

                        if st.button(
                            "I understand this now — show Class 10 level",
                            key=f"practice_step_up_10_{question_id}"
                        ):

                            st.session_state.practice_explanation_level = (
                                "Class 10"
                            )

                            st.rerun()

                    elif explanation_level == "Class 10":

                        if st.button(
                            "Ready to retry at Class 12 level",
                            key=f"practice_retry_12_{question_id}"
                        ):

                            load_new_practice_question(
                                practice_level
                            )

                            st.rerun()

                with col_b:

                    if st.button(
                        "Skip — try a new question instead",
                        key=f"practice_skip_{question_id}"
                    ):

                        load_new_practice_question(
                            practice_level
                        )

                        st.rerun()


# =========================================================
# MY PROGRESS
# =========================================================

elif mode == "📊 My Progress":

    st.header(
        "📊 My Progress"
    )

    st.caption(
        "Track your BioAssist practice and evaluation performance."
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

        st.markdown(
            """
            <div class="empty-state">
                🧪 No learning activity has been recorded yet.<br>
                Head to <b>Practice</b> and attempt a question to get started.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        # =================================================
        # SUMMARY
        # =================================================

        evaluation_scores = []

        for row in progress:

            try:

                if isinstance(row, dict):

                    activity = row.get(
                        "Activity",
                        row.get("activity", "")
                    )

                    score = row.get(
                        "Score",
                        row.get("score")
                    )

                else:

                    activity = row[2]
                    score = row[4]

                if (
                    activity == "Answer Evaluation"
                    and score is not None
                ):

                    evaluation_scores.append(float(score))

            except Exception:

                continue


        questions_attempted = len(evaluation_scores)

        average_score = (
            sum(evaluation_scores) / questions_attempted
            if questions_attempted
            else 0
        )

        best_score = (
            max(evaluation_scores)
            if evaluation_scores
            else 0
        )

        suggested_level = suggest_explanation_level(
            student
        )


        summary1, summary2, summary3, summary4 = st.columns(4)

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

        with summary4:

            st.metric(
                "Current Level",
                suggested_level
            )


        # =================================================
        # SCORE TREND
        # =================================================

        trend_rows = []

        for row in progress:

            try:

                if isinstance(row, dict):

                    activity = row.get(
                        "Activity",
                        row.get("activity", "")
                    )

                    score = row.get(
                        "Score",
                        row.get("score")
                    )

                    date = row.get(
                        "Date",
                        row.get("created_at", "")
                    )

                else:

                    activity = row[2]
                    score = row[4]
                    date = row[5]

                if (
                    activity == "Answer Evaluation"
                    and score is not None
                    and date
                ):

                    trend_rows.append(
                        {
                            "Date": date,
                            "Score": float(score)
                        }
                    )

            except Exception:

                continue

        if len(trend_rows) >= 2:

            trend_df = pd.DataFrame(
                trend_rows
            )

            trend_df["Date"] = pd.to_datetime(
                trend_df["Date"],
                errors="coerce"
            )

            trend_df = (
                trend_df
                .dropna(subset=["Date"])
                .sort_values("Date")
                .set_index("Date")
            )

            if not trend_df.empty:

                st.caption(
                    "Score trend across attempted questions"
                )

                st.line_chart(
                    trend_df["Score"],
                    height=220
                )


        st.divider()


        # =================================================
        # DETAILED HISTORY
        # =================================================

        st.subheader(
            "Recent Activity"
        )


        if (
            isinstance(progress, list)
            and progress
            and isinstance(progress[0], dict)
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
                            "Class": row[0],
                            "Chapter": row[1],
                            "Activity": row[2],
                            "Topic / Question": row[3],
                            "Score": row[4],
                            "Date": row[5]
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
