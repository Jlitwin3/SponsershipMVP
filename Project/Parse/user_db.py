"""
SQLite database helper for user management
"""
import sqlite3
import os
from datetime import datetime
from typing import List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)

# Admin functions
def is_admin(email: str) -> bool:
    """Check if email is in admin list"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM admins WHERE email = ?', (email.lower(),))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def get_all_admins() -> List[str]:
    """Get all admin emails"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT email FROM admins ORDER BY email')
    admins = [row[0] for row in cursor.fetchall()]
    conn.close()
    return admins

def add_admin(email: str, added_by: str = 'admin') -> bool:
    """Add an email to admin list"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO admins (email, added_date, added_by) VALUES (?, ?, ?)',
            (email.lower(), datetime.now().isoformat(), added_by)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists

def remove_admin(email: str) -> bool:
    """Remove an email from admin list"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM admins WHERE email = ?', (email.lower(),))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

# Whitelisted user functions
def is_whitelisted_user(email: str) -> bool:
    """Check if email is in whitelist"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM whitelisted_users WHERE email = ?', (email.lower(),))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def get_all_whitelisted_users() -> List[str]:
    """Get all whitelisted user emails"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT email FROM whitelisted_users ORDER BY email')
    users = [row[0] for row in cursor.fetchall()]
    conn.close()
    return users

def add_whitelisted_user(email: str, added_by: str = 'admin') -> bool:
    """Add an email to whitelist"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO whitelisted_users (email, added_date, added_by) VALUES (?, ?, ?)',
            (email.lower(), datetime.now().isoformat(), added_by)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists

def remove_whitelisted_user(email: str) -> bool:
    """Remove an email from whitelist"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM whitelisted_users WHERE email = ?', (email.lower(),))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

# Combined authorization check
def is_authorized_for_chatbot(email: str) -> bool:
    """Check if user can access chatbot (admin OR whitelisted user)"""
    return is_admin(email) or is_whitelisted_user(email)
