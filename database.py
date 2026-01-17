import sqlite3

DB_NAME = "ecotogether.db"

def connect_db():
    return sqlite3.connect(DB_NAME)

def create_table():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            points INTEGER
        )
    """)
    conn.commit()
    conn.close()

def get_user(username):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT points FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0

def update_user(username, add_points):
    current = get_user(username)
    total = current + add_points
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO users (username, points) VALUES (?, ?)",
        (username, total)
    )
    conn.commit()
    conn.close()
    return total


