# This file can be dropped into your codebase and imported instead of your current CSV helpers.
import sqlite3
import threading
from typing import List, Dict, Optional

class EntryStorage:
    def __init__(self, db_path="entries.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute('CREATE VIRTUAL TABLE IF NOT EXISTS entries USING FTS5(text, link, category)')
            c.execute('CREATE TABLE IF NOT EXISTS categories (name TEXT PRIMARY KEY)')
            # Ensure default categories
            c.execute('SELECT COUNT(*) FROM categories')
            if c.fetchone()[0] == 0:
                c.executemany('INSERT INTO categories (name) VALUES (?)',
                              [('General',), ('Documentation',), ('Tutorials',), ('References',)])
            conn.commit()

    def add_entry(self, text: str, link: str, category: str = "General") -> bool:
        with self._lock, self._get_conn() as conn:
            c = conn.cursor()
            c.execute('SELECT 1 FROM entries WHERE text=? AND link=?', (text, link))
            if c.fetchone():
                return False
            c.execute('INSERT INTO entries (text, link, category) VALUES (?, ?, ?)', (text, link, category))
            c.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (category,))
            conn.commit()
        return True

    def insert_entry_at(self, index: int, text: str, link: str, category: str = "General") -> bool:
        # Not natively supported in FTS, so fetch all, insert, and rewrite
        with self._lock, self._get_conn() as conn:
            c = conn.cursor()
            c.execute('SELECT text, link, category FROM entries ORDER BY rowid')
            entries = c.fetchall()
            new_row = (text, link, category)
            if index < 0 or index > len(entries):
                return False
            entries = entries[:index] + [new_row] + entries[index:]
            c.execute('DELETE FROM entries')
            c.executemany('INSERT INTO entries (text, link, category) VALUES (?, ?, ?)', entries)
            c.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (category,))
            conn.commit()
        return True

    def get_entries(self, category: Optional[str] = None) -> List[Dict]:
        with self._get_conn() as conn:
            c = conn.cursor()
            if category:
                c.execute('SELECT rowid, text, link, category FROM entries WHERE category=? ORDER BY rowid', (category,))
            else:
                c.execute('SELECT rowid, text, link, category FROM entries ORDER BY rowid')
            return [{"id": row[0], "text": row[1], "link": row[2], "category": row[3]} for row in c.fetchall()]

    def get_entry_by_index(self, index: int) -> Optional[Dict]:
        entries = self.get_entries()
        if 0 <= index < len(entries):
            return entries[index]
        return None

    def delete_entry_by_index(self, index: int) -> bool:
        entries = self.get_entries()
        if 0 <= index < len(entries):
            entry_id = entries[index]['id']
            with self._lock, self._get_conn() as conn:
                c = conn.cursor()
                c.execute('DELETE FROM entries WHERE rowid=?', (entry_id,))
                conn.commit()
                return c.rowcount > 0
        return False

    def get_categories(self) -> List[str]:
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute('SELECT name FROM categories ORDER BY name')
            return [row[0] for row in c.fetchall()]

    def add_category(self, category: str) -> bool:
        with self._lock, self._get_conn() as conn:
            c = conn.cursor()
            try:
                c.execute('INSERT INTO categories (name) VALUES (?)', (category,))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
