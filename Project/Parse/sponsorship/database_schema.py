"""
Database Schema for Sponsor Management
Creates SQLite database structure for tracking UO sponsors
"""

import sqlite3
import os
from datetime import datetime

# Use persistent path if provided (Render), else local directory
data_dir = os.getenv("SQLITE_DATA_PATH", os.path.dirname(__file__))
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
DB_PATH = os.path.join(data_dir, "sponsors.db")


def init_database():
    """Initialize the sponsor database with tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create sponsors table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sponsors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            is_exclusive BOOLEAN DEFAULT 0,
            start_date TEXT,
            end_date TEXT,
            contract_value DECIMAL,
            sports_focus TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create sponsorship categories table (reference)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create sponsorship history/events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sponsorship_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sponsor_id INTEGER NOT NULL,
            event_date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            details TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sponsor_id) REFERENCES sponsors(id)
        )
    """)

    # Create competing brands reference table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS competing_brands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sponsor_id INTEGER NOT NULL,
            competing_brand_name TEXT NOT NULL,
            notes TEXT,
            FOREIGN KEY (sponsor_id) REFERENCES sponsors(id)
        )
    """)

    # Insert default categories
    default_categories = [
        ("Athletic Apparel", "Uniforms, shoes, athletic wear"),
        ("Beverages", "Sports drinks, soft drinks, water"),
        ("Financial Services", "Banks, credit cards, insurance"),
        ("Technology", "Computers, software, telecommunications"),
        ("Automotive", "Cars, trucks, vehicle services"),
        ("Food & Dining", "Restaurants, food products, snacks"),
        ("Healthcare", "Hospitals, medical services, health products"),
        ("Retail", "Department stores, sporting goods stores"),
        ("Media & Entertainment", "Broadcasting, streaming services"),
        ("Travel & Hospitality", "Airlines, hotels, travel services"),
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO categories (name, description)
        VALUES (?, ?)
    """, default_categories)

    conn.commit()
    conn.close()

    print(f"✅ Database initialized at: {DB_PATH}")
    return DB_PATH


def reset_database():
    """Drop all tables and reinitialize (use with caution!)"""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("🗑️  Old database deleted")
    init_database()
    print("🔄 Database reset complete")


if __name__ == "__main__":
    # Run this file directly to initialize the database
    init_database()

    # Print summary
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM categories")
    category_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM sponsors")
    sponsor_count = cursor.fetchone()[0]

    print(f"\n📊 Database Summary:")
    print(f"   Categories: {category_count}")
    print(f"   Sponsors: {sponsor_count}")
    print(f"\n💡 Next steps:")
    print(f"   1. Run 'python3 populate_sample_data.py' to add sample sponsors")
    print(f"   2. Or use sponsor_manager.py functions to add real data")

    conn.close()
