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
    conn.commit()
    return conn

def save_progress(student, class_name, chapter, activity, topic='', score=None):
    conn = _connect()
    conn.execute('INSERT INTO progress(student,class_name,chapter,activity,topic,score,created_at) VALUES(?,?,?,?,?,?,?)',
                 (student,class_name,chapter,activity,topic,score,datetime.now().isoformat(timespec='seconds')))
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
