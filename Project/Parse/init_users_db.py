"""
Initialize SQLite database for user management.
This script creates the database and migrates existing users from .env
"""
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def init_database():
    """Create the database tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create admins table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            added_date TEXT NOT NULL,
            added_by TEXT DEFAULT 'system'
        )
    ''')

    # Create whitelisted_users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whitelisted_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            added_date TEXT NOT NULL,
            added_by TEXT DEFAULT 'system'
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Database created at: {DB_PATH}")

def migrate_from_env():
    """Migrate existing emails from .env file to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    current_time = datetime.now().isoformat()

    # Migrate admin list
    admin_str = os.getenv("ADMIN_LIST", "")
    admin_emails = [e.strip().lower() for e in admin_str.split(",") if e.strip()]

    migrated_admins = 0
    for email in admin_emails:
        try:
            cursor.execute(
                'INSERT OR IGNORE INTO admins (email, added_date, added_by) VALUES (?, ?, ?)',
                (email, current_time, 'env_migration')
            )
            if cursor.rowcount > 0:
                migrated_admins += 1
        except Exception as e:
            print(f"⚠️ Error migrating admin {email}: {e}")

    # Migrate user whitelist
    user_str = os.getenv("USER_WHITELIST", "")
    whitelisted_emails = [e.strip().lower() for e in user_str.split(",") if e.strip()]

    migrated_users = 0
    for email in whitelisted_emails:
        try:
            cursor.execute(
                'INSERT OR IGNORE INTO whitelisted_users (email, added_date, added_by) VALUES (?, ?, ?)',
                (email, current_time, 'env_migration')
            )
            if cursor.rowcount > 0:
                migrated_users += 1
        except Exception as e:
            print(f"⚠️ Error migrating user {email}: {e}")

    conn.commit()
    conn.close()

    print(f"✅ Migrated {migrated_admins} admins from ADMIN_LIST")
    print(f"✅ Migrated {migrated_users} users from USER_WHITELIST")

def display_current_users():
    """Display all users in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n📋 Current Admins:")
    cursor.execute('SELECT email, added_date, added_by FROM admins ORDER BY added_date')
    admins = cursor.fetchall()
    if admins:
        for email, added_date, added_by in admins:
            print(f"  - {email} (added {added_date[:10]} by {added_by})")
    else:
        print("  (none)")

    print("\n📋 Current Whitelisted Users:")
    cursor.execute('SELECT email, added_date, added_by FROM whitelisted_users ORDER BY added_date')
    users = cursor.fetchall()
    if users:
        for email, added_date, added_by in users:
            print(f"  - {email} (added {added_date[:10]} by {added_by})")
    else:
        print("  (none)")

    conn.close()

if __name__ == '__main__':
    print("🚀 Initializing user database...")
    init_database()
    migrate_from_env()
    display_current_users()
    print("\n✅ Database initialization complete!")
