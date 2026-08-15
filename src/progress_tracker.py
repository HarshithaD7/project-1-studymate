import os, sqlite3
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_DIR, 'student_progress.db')

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS progress(
        id INTEGER PRIMARY KEY AUTOINCREMENT, student TEXT, class_name TEXT,
        chapter TEXT, activity TEXT, topic TEXT, score REAL, created_at TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT, student TEXT, rating INTEGER,
        pre_score REAL, post_score REAL, comments TEXT, created_at TEXT)''')

    # Migration: the "progress" table pre-dates the Capstone 2
    # Critical Thinking level, so existing databases won't have
    # this column yet. ADD COLUMN is safe on an existing table --
    # old rows just get NULL here, which every reader below
    # already treats as "recall/unknown level". Wrapped in
    # try/except because SQLite has no "ADD COLUMN IF NOT EXISTS",
    # and re-running this on every connect would otherwise error
    # once the column already exists.
    try:
        conn.execute('ALTER TABLE progress ADD COLUMN question_level TEXT')
    except sqlite3.OperationalError:
        pass

    conn.commit()
    return conn

def save_progress(student, class_name, chapter, activity, topic='', score=None, question_level=None):
    conn = _connect()
    conn.execute('INSERT INTO progress(student,class_name,chapter,activity,topic,score,created_at,question_level) VALUES(?,?,?,?,?,?,?,?)',
                 (student,class_name,chapter,activity,topic,score,datetime.now().isoformat(timespec='seconds'),question_level))
    conn.commit(); conn.close()

def get_progress(student):
    conn = _connect()
    rows = conn.execute('SELECT class_name,chapter,activity,topic,score,created_at FROM progress WHERE student=? ORDER BY id DESC',(student,)).fetchall()
    conn.close(); return rows

def save_feedback(student, rating, pre_score, post_score, comments):
    conn = _connect()
    conn.execute('INSERT INTO feedback(student,rating,pre_score,post_score,comments,created_at) VALUES(?,?,?,?,?,?)',
                 (student,rating,pre_score,post_score,comments,datetime.now().isoformat(timespec='seconds')))
    conn.commit(); conn.close()


def get_average_score(student):
    """
    Average 'Answer Evaluation' score (0-10) for the student.
    Returns None if there is no evaluation history yet.
    """
    rows = get_progress(student)
    scores = []
    for row in rows:
        try:
            activity = row[2]
            score = row[4]
            if activity == 'Answer Evaluation' and score is not None:
                scores.append(float(score))
        except Exception:
            continue
    if not scores:
        return None
    return sum(scores) / len(scores)


def _evaluation_rows_with_level(student):
    """
    Internal helper: every 'Answer Evaluation' row for this
    student with a usable score, including question_level.
    Kept separate from get_progress() so that function's
    existing return shape (and every place that already
    unpacks it by position) stays untouched.
    """
    conn = _connect()
    rows = conn.execute(
        'SELECT chapter, score, question_level FROM progress '
        'WHERE student=? AND activity=? AND score IS NOT NULL',
        (student, 'Answer Evaluation')
    ).fetchall()
    conn.close()
    return rows


def get_skill_breakdown(student):
    """
    Splits evaluation history into two cognitive skills:
    'recall' (MCQ / 1-5 Mark -- and anything saved before this
    column existed, which is treated as recall) vs
    'critical_thinking' (the Capstone 2 applied-reasoning level).

    Returns None for a skill with no attempts yet, so the UI can
    say "not attempted yet" instead of a misleading 0.
    """
    rows = _evaluation_rows_with_level(student)

    recall_scores = []
    critical_thinking_scores = []

    for _chapter, score, question_level in rows:
        try:
            score = float(score)
        except Exception:
            continue

        if question_level == 'Critical Thinking':
            critical_thinking_scores.append(score)
        else:
            recall_scores.append(score)

    def _avg(values):
        return round(sum(values) / len(values), 1) if values else None

    return {
        'recall_average': _avg(recall_scores),
        'recall_attempts': len(recall_scores),
        'critical_thinking_average': _avg(critical_thinking_scores),
        'critical_thinking_attempts': len(critical_thinking_scores),
    }


def mastery_label(recall_avg, critical_thinking_avg):
    """
    One-line, rule-based (no LLM needed) diagnostic phrase --
    this is the actual pedagogical payoff of tracking the two
    skills separately: showing that knowing facts and applying
    them are different things, and which one needs more work.
    """
    if recall_avg is None and critical_thinking_avg is None:
        return 'Not attempted yet'

    if critical_thinking_avg is None:
        return 'Only recall practice so far -- try Critical Thinking'

    if recall_avg is None:
        return 'Only Critical Thinking practice so far'

    gap = recall_avg - critical_thinking_avg

    if gap >= 3:
        return 'Strong recall, needs practice applying concepts'

    if gap <= -3:
        return 'Strong reasoning -- revisit core facts for speed'

    if recall_avg >= 7 and critical_thinking_avg >= 7:
        return 'Solid on both recall and application'

    if recall_avg < 4 and critical_thinking_avg < 4:
        return 'Needs foundational review'

    return 'Balanced -- keep practicing both'


def get_chapter_mastery(student):
    """
    Per-chapter knowledge view: average score, attempt count,
    and the recall-vs-critical-thinking split for every chapter
    the student has actually attempted. This is the "how much
    does the student know on each topic, collectively" view --
    grouped by chapter instead of one flat overall average.

    Returns a list of dicts sorted by most-attempted chapter
    first (a reasonable proxy for "most relevant to the
    student right now" without needing any extra bookkeeping).
    """
    rows = _evaluation_rows_with_level(student)

    by_chapter = {}

    for chapter, score, question_level in rows:
        chapter = (chapter or 'Unknown Chapter').strip()

        try:
            score = float(score)
        except Exception:
            continue

        entry = by_chapter.setdefault(
            chapter,
            {'all_scores': [], 'recall_scores': [], 'critical_thinking_scores': []}
        )

        entry['all_scores'].append(score)

        if question_level == 'Critical Thinking':
            entry['critical_thinking_scores'].append(score)
        else:
            entry['recall_scores'].append(score)

    def _avg(values):
        return round(sum(values) / len(values), 1) if values else None

    results = []

    for chapter, data in by_chapter.items():
        recall_avg = _avg(data['recall_scores'])
        ct_avg = _avg(data['critical_thinking_scores'])

        results.append({
            'chapter': chapter,
            'attempts': len(data['all_scores']),
            'overall_average': _avg(data['all_scores']),
            'recall_average': recall_avg,
            'critical_thinking_average': ct_avg,
            'label': mastery_label(recall_avg, ct_avg),
        })

    results.sort(key=lambda item: item['attempts'], reverse=True)

    return results


def suggest_explanation_level(student):
    """
    Suggests a Learn explanation depth (Class 8 / Class 10 / Class 12)
    based on the student's average Evaluate My Answer performance.

    Defaults to Class 12 until there is enough evaluation history to
    judge, so a new student always starts at full NCERT depth rather
    than being assumed weak.
    """
    average_score = get_average_score(student)

    if average_score is None:
        return 'Class 12'

    if average_score >= 7:
        return 'Class 12'

    if average_score >= 4:
        return 'Class 10'

    return 'Class 8'
