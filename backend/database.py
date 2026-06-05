import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'healthdecode.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            original_text TEXT,
            summary TEXT,
            observations TEXT,
            abnormal_values TEXT,
            terminology TEXT,
            doctor_questions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_report(report_id, filename, file_type, original_text, summary, observations, abnormal_values, terminology, doctor_questions):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reports (id, filename, file_type, original_text, summary, observations, abnormal_values, terminology, doctor_questions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        report_id,
        filename,
        file_type,
        original_text,
        summary,
        json.dumps(observations),
        json.dumps(abnormal_values),
        json.dumps(terminology),
        json.dumps(doctor_questions)
    ))
    conn.commit()
    conn.close()

def get_all_reports():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, file_type, created_at, summary FROM reports ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    reports = []
    for r in rows:
        reports.append({
            'id': r['id'],
            'filename': r['filename'],
            'file_type': r['file_type'],
            'created_at': r['created_at'],
            'summary': r['summary']
        })
    return reports

def get_report(report_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    return {
        'id': row['id'],
        'filename': row['filename'],
        'file_type': row['file_type'],
        'original_text': row['original_text'],
        'summary': row['summary'],
        'observations': json.loads(row['observations']),
        'abnormal_values': json.loads(row['abnormal_values']),
        'terminology': json.loads(row['terminology']),
        'doctor_questions': json.loads(row['doctor_questions']),
        'created_at': row['created_at']
    }

def delete_report(report_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM reports WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()
