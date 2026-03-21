#import sqlite3
import aiosqlite
from datetime import datetime


class DbClient:
    def __init__(self, db_path: str = "./bot_history.db"):
        self.db_path = db_path

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    chat_id   INTEGER,
                    user_id   INTEGER,
                    username  TEXT,
                    first_name TEXT,
                    message_text TEXT,
                    role      TEXT DEFAULT 'user',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            await db.commit()

    async def save_message(self, chat_id: int, user_id: int, username: str,
                           first_name: str, text: str, role: str = "user"):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO messages 
                (chat_id, user_id, username, first_name, message_text, role, timestamp) 
                VALUES (?,?,?,?,?,?,?)""",
                (chat_id, user_id, username, first_name, text, role, datetime.now().isoformat())
            )            
            await db.commit()

    async def get_user_history(self, chat_id: int, user_id: int, limit: int = 20) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                '''SELECT role, message_text, username, timestamp
                   FROM messages
                   WHERE chat_id = ? AND user_id = ?
                   AND timestamp >= date('now', 'start of day')-1
                   AND message_text IS NOT NULL
                   ORDER BY timestamp DESC
                   LIMIT ?''',
                (chat_id, user_id, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [{"role": row["role"], "content": row["message_text"]} for row in reversed(rows)]  # type: ignore

    async def get_user_messages(self, chat_id: int, username: str) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                '''SELECT username, first_name, message_text, timestamp
                   FROM messages
                   WHERE chat_id = ? AND username = ?
                   AND timestamp >= date('now', 'start of day')
                   AND message_text IS NOT NULL
                   ORDER BY timestamp ASC
                   ''',
                (chat_id, username)
            ) as cursor:
                rows = await cursor.fetchall()
                return [f"[{r[3]}] {r[0] or r[1] or 'Unknown'}: {r[2]}" for r in rows]                

    def build_ai_context(self, history: list[dict]) -> list[dict]:
        """Перетворює історію в формат messages для AI"""
        return [
            {"role": r["role"], "content": r["message_text"]}
            for r in history
        ]
    