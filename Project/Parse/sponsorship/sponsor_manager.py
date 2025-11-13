"""
Sponsor Manager - Database operations for sponsor management
Provides functions to check conflicts, retrieve sponsor info, and manage sponsors
"""

import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "sponsors.db")


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def check_sponsor_conflict(sponsor_name: str, category: str = None) -> Dict:
    """
    Check if a sponsor conflicts with existing agreements.

    Args:
        sponsor_name: Name of the potential sponsor
        category: Category to check (e.g., "Athletic Apparel")

    Returns:
        {
            'conflict': bool,
            'existing_sponsor': str or None,
            'details': str,
            'is_exclusive': bool,
            'category': str
        }
    """
    conn = get_connection()
    cursor = conn.cursor()

    # First, check if this exact sponsor already exists as current
    cursor.execute("""
        SELECT name, category, is_exclusive, start_date, notes
        FROM sponsors
        WHERE LOWER(name) = LOWER(?)
        AND relationship_type = 'current'
    """, (sponsor_name,))

    existing = cursor.fetchone()

    if existing:
        name, cat, is_exclusive, start_date, notes = existing
        conn.close()
        return {
            'conflict': True,
            'existing_sponsor': name,
            'details': f"{name} is already a current sponsor in {cat} (since {start_date}). {notes or ''}",
            'is_exclusive': bool(is_exclusive),
            'category': cat,
            'reason': 'ALREADY_SPONSOR'
        }

    # If category provided, check for exclusive sponsors in that category
    if category:
        cursor.execute("""
            SELECT name, category, start_date, notes
            FROM sponsors
            WHERE category = ?
            AND relationship_type = 'current'
            AND is_exclusive = 1
        """, (category,))

        exclusive_sponsor = cursor.fetchone()

        if exclusive_sponsor:
            name, cat, start_date, notes = exclusive_sponsor
            conn.close()
            return {
                'conflict': True,
                'existing_sponsor': name,
                'details': f"{name} has an exclusive agreement in {cat} (since {start_date}). Cannot add competing sponsors. {notes or ''}",
                'is_exclusive': True,
                'category': cat,
                'reason': 'EXCLUSIVE_CONFLICT'
            }

    conn.close()

    # No conflict found
    return {
        'conflict': False,
        'existing_sponsor': None,
        'details': 'No conflicts found. This sponsor can be proposed.',
        'is_exclusive': False,
        'category': category,
        'reason': 'NO_CONFLICT'
    }


def get_sponsor_info(sponsor_name: str) -> Optional[Dict]:
    """
    Get detailed information about a sponsor.

    Returns:
        Dict with sponsor details or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, category, relationship_type, is_exclusive,
               start_date, end_date, contract_value, sports_focus, notes
        FROM sponsors
        WHERE LOWER(name) = LOWER(?)
    """, (sponsor_name,))

    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    sponsor_id, name, category, rel_type, is_exclusive, start_date, end_date, value, sports, notes = row

    # Get history
    cursor.execute("""
        SELECT event_date, event_type, details
        FROM sponsorship_history
        WHERE sponsor_id = ?
        ORDER BY event_date DESC
    """, (sponsor_id,))

    history = [{'date': h[0], 'type': h[1], 'details': h[2]} for h in cursor.fetchall()]

    # Get competing brands
    cursor.execute("""
        SELECT competing_brand_name, notes
        FROM competing_brands
        WHERE sponsor_id = ?
    """, (sponsor_id,))

    competing = [{'name': c[0], 'notes': c[1]} for c in cursor.fetchall()]

    conn.close()

    return {
        'id': sponsor_id,
        'name': name,
        'category': category,
        'relationship_type': rel_type,
        'is_exclusive': bool(is_exclusive),
        'start_date': start_date,
        'end_date': end_date,
        'contract_value': value,
        'sports_focus': sports,
        'notes': notes,
        'history': history,
        'competing_brands': competing
    }


def get_all_current_sponsors() -> List[Dict]:
    """Get all current sponsors."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, category, is_exclusive, start_date, sports_focus, notes
        FROM sponsors
        WHERE relationship_type = 'current'
        ORDER BY category, name
    """)

    sponsors = []
    for row in cursor.fetchall():
        sponsors.append({
            'name': row[0],
            'category': row[1],
            'is_exclusive': bool(row[2]),
            'start_date': row[3],
            'sports_focus': row[4],
            'notes': row[5]
        })

    conn.close()
    return sponsors


def get_sponsors_by_category(category: str) -> List[Dict]:
    """Get all sponsors in a specific category."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, relationship_type, is_exclusive, start_date, end_date
        FROM sponsors
        WHERE category = ?
        ORDER BY relationship_type, name
    """, (category,))

    sponsors = []
    for row in cursor.fetchall():
        sponsors.append({
            'name': row[0],
            'relationship_type': row[1],
            'is_exclusive': bool(row[2]),
            'start_date': row[3],
            'end_date': row[4]
        })

    conn.close()
    return sponsors


def add_sponsor(name: str, category: str, relationship_type: str = "current",
                is_exclusive: bool = False, start_date: str = None,
                end_date: str = None, contract_value: float = None,
                sports_focus: str = None, notes: str = None) -> int:
    """
    Add a new sponsor to the database.

    Returns:
        sponsor_id of the newly created sponsor
    """
    conn = get_connection()
    cursor = conn.cursor()

    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""
        INSERT INTO sponsors (name, category, relationship_type, is_exclusive,
                            start_date, end_date, contract_value, sports_focus, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, category, relationship_type, is_exclusive, start_date,
          end_date, contract_value, sports_focus, notes))

    sponsor_id = cursor.lastrowid

    # Add initial history event
    cursor.execute("""
        INSERT INTO sponsorship_history (sponsor_id, event_date, event_type, details)
        VALUES (?, ?, ?, ?)
    """, (sponsor_id, start_date, "created", f"Sponsor {name} added to database"))

    conn.commit()
    conn.close()

    print(f"✅ Added sponsor: {name} (ID: {sponsor_id})")
    return sponsor_id


def update_sponsor(sponsor_id: int, **kwargs) -> bool:
    """Update sponsor fields."""
    conn = get_connection()
    cursor = conn.cursor()

    # Build dynamic update query
    fields = []
    values = []

    allowed_fields = ['name', 'category', 'relationship_type', 'is_exclusive',
                     'start_date', 'end_date', 'contract_value', 'sports_focus', 'notes']

    for field, value in kwargs.items():
        if field in allowed_fields:
            fields.append(f"{field} = ?")
            values.append(value)

    if not fields:
        conn.close()
        return False

    # Add updated_at
    fields.append("updated_at = ?")
    values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    values.append(sponsor_id)

    query = f"UPDATE sponsors SET {', '.join(fields)} WHERE id = ?"
    cursor.execute(query, values)

    conn.commit()
    success = cursor.rowcount > 0
    conn.close()

    if success:
        print(f"✅ Updated sponsor ID: {sponsor_id}")

    return success


def add_sponsor_history(sponsor_id: int, event_date: str, event_type: str, details: str):
    """Add a history event for a sponsor."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sponsorship_history (sponsor_id, event_date, event_type, details)
        VALUES (?, ?, ?, ?)
    """, (sponsor_id, event_date, event_type, details))

    conn.commit()
    conn.close()

    print(f"✅ Added history event for sponsor ID {sponsor_id}")


def get_all_categories() -> List[str]:
    """Get list of all available categories."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM categories ORDER BY name")
    categories = [row[0] for row in cursor.fetchall()]

    conn.close()
    return categories


# Helper function for formatting sponsor info
def format_sponsor_context(sponsor_info: Dict) -> str:
    """
    Format sponsor information as context for LLM.

    Returns formatted string suitable for injection into AI context.
    """
    if not sponsor_info:
        return ""

    # More natural formatting without technical labels
    context = f"[Current UO Sponsor Information]\n"
    context += f"{sponsor_info['name']} - {sponsor_info['category']}\n"
    context += f"Partnership Status: {sponsor_info['relationship_type'].title()}\n"

    if sponsor_info['is_exclusive']:
        context += f"Exclusive Agreement: Yes (no competing brands allowed in this category)\n"

    context += f"Partnership Since: {sponsor_info['start_date']}\n"

    if sponsor_info['end_date']:
        context += f"Contract End Date: {sponsor_info['end_date']}\n"

    if sponsor_info['sports_focus']:
        context += f"Supports: {sponsor_info['sports_focus']}\n"

    if sponsor_info['notes']:
        context += f"Additional Details: {sponsor_info['notes']}\n"

    if sponsor_info['history']:
        context += f"\nPartnership Timeline:\n"
        for event in sponsor_info['history'][:3]:  # Last 3 events
            context += f"  • {event['date']}: {event['type'].title()} - {event['details']}\n"

    return context


if __name__ == "__main__":
    # Example usage
    print("Sponsor Manager - Example Usage\n")

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print("❌ Database not found. Run 'python3 database_schema.py' first.")
    else:
        print("✅ Database found\n")

        # Show all current sponsors
        sponsors = get_all_current_sponsors()
        print(f"📊 Current Sponsors: {len(sponsors)}")
        for s in sponsors:
            exclusive = " (EXCLUSIVE)" if s['is_exclusive'] else ""
            print(f"  - {s['name']} - {s['category']}{exclusive}")
