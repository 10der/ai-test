import sqlite3
import json
from typing import Any
import os
import numpy as np
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import re


def clean_for_fts(text):
    # Видаляємо емодзі та неалфавітні символи на початку слів
    return re.sub(r'[^\w\s]', ' ', text).strip()


class MyRAG:
    def __init__(self, db_path: str = "./rag.db", model_name: str = '../MiniLM'):
        self.db_path = db_path
        self._init_db(self.db_path, "./create_db.sql")
        self.model = SentenceTransformer(model_name)
        # Кеш для векторів, щоб не читати диск при кожному запиті
        self._matrix_cache: dict[str, np.ndarray] = {}
        self._metadata_cache: dict[str, list] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _init_db(self, db_path: str, sql_file_path: str):
        # Перевіряємо, чи існує файл зі схемою
        if not os.path.exists(sql_file_path):
            print(f"Помилка: Файл {sql_file_path} не знайдено!")
            return

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Читаємо весь SQL файл
                with open(sql_file_path, 'r', encoding='utf-8') as f:
                    sql_script = f.read()

                # Виконуємо всі команди одним махом
                cursor.executescript(sql_script)
                conn.commit()
                print("Базу даних успішно ініціалізовано.")

        except sqlite3.Error as e:
            print(f"Помилка SQLite: {e}")

    def _load_cache(self, source: str | None = None):
        """Завантажує всі вектори в пам'ять один раз"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if source:
                cursor.execute("""
                    SELECT c.text, c.embedding, d.date, d.source 
                    FROM rag_chunks c 
                    JOIN raw_docs d ON c.doc_id = d.id
                    WHERE d.source = ?
                """, (source,))
            else:
                cursor.execute("""
                    SELECT c.text, c.embedding, d.date, d.source 
                    FROM rag_chunks c 
                    JOIN raw_docs d ON c.doc_id = d.id
                """)
            rows = cursor.fetchall()

        if rows:
            key = source or "__all__"
            self._matrix_cache[key] = np.array(
                [json.loads(r[1]) for r in rows])
            self._metadata_cache[key] = rows

    def cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        # Оптимізована версія без зайвих копіювань
        dot_product = np.dot(matrix, query_vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        return dot_product / (matrix_norms * query_norm + 1e-10)

    def _keyword_search(self, query: str, top_k: int) -> list:
        keywords = [kw for kw in query.split() if len(kw) > 2]
        if not keywords:
            return []

        conditions = " OR ".join(["c.text LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT c.text, d.source, d.date
                FROM rag_chunks c
                JOIN raw_docs d ON c.doc_id = d.id
                WHERE {conditions}
                LIMIT ?
            """, params + [top_k])
            rows = cursor.fetchall()

        return [{"static_data": r[0], "subjects": r[1].split(","),
                "date": r[2], "similarity": 1.0} for r in rows]

    def _fts_search(self, query: str, top_k: int, recency_weight: float = 0.0, source: str | None = None) -> list:

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = clean_for_fts(query)
            params: list[Any] = [query]
            if recency_weight > 0:
                order_clause = "(bm25(rag_chunks_fts) * (1.0 / (1 + (julianday('now') - julianday(d.date)) * ?)))"
                params = [query, recency_weight]
            else:
                order_clause = "bm25(rag_chunks_fts)"
                params = [query]

            if source:
                params.append(source)

            params.append(top_k)

            cursor.execute(f"""
                SELECT c.text, d.source, d.date,
                    bm25(rag_chunks_fts) as score
                FROM rag_chunks_fts
                JOIN rag_chunks c ON rag_chunks_fts.rowid = c.id
                JOIN raw_docs d ON c.doc_id = d.id
                WHERE rag_chunks_fts MATCH ?
                {"AND d.source = ?" if source else ""}
                ORDER BY {order_clause}
                LIMIT ?
            """, params)

            rows = cursor.fetchall()

        if not rows:
            return []

        # Нормалізуємо BM25 до 0-1
        scores = [abs(r[3]) for r in rows]
        max_score = max(scores)

        return [{"static_data": r[0], "subjects": r[1].split(","),
                "date": r[2], "similarity": abs(r[3]) / max_score} for r in rows]

    def _merge_results(self, semantic: list, keyword: list, top_k: int) -> list:
        seen = set()
        merged = []
        for r in keyword:
            if r["static_data"] not in seen:
                seen.add(r["static_data"])
                merged.append(r)
        for r in semantic:
            if r["static_data"] not in seen:
                seen.add(r["static_data"])
                merged.append(r)
        return merged[:top_k]

    async def search(self, user_query: str, recency_weight: float = 0.0, top_k: int = 3, threshold: float = 0.45, source: str | None = None):
        semantic = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._sync_search, user_query, top_k, threshold, source
        )

        keyword = self._fts_search(user_query, top_k, recency_weight, source)

        return self._merge_results(semantic, keyword, top_k)

    def _sync_search(self, user_query: str, top_k: int, threshold: float, source: str | None = None):
        key = source or "__all__"

        if key not in self._matrix_cache:
            self._load_cache(source)

        if key not in self._matrix_cache:
            return []

        matrix = self._matrix_cache[key]
        metadata = self._metadata_cache[key]

        # Енкодимо запит
        query_vector = np.array(self.model.encode(
            user_query, convert_to_numpy=True))

        # Рахуємо схожість одразу для всієї матриці
        similarities = self.cosine_similarity(query_vector, matrix)

        # Знаходимо індекси найкращих результатів
        best_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in best_indices:
            score = similarities[idx]
            if score < threshold:
                continue

            text, _, doc_date, doc_source = metadata[idx]

            if source and doc_source != source:
                continue

            results.append({
                "static_data": text,
                "subjects": doc_source.split(","),
                "date": doc_date,
                "similarity": float(score)
            })

        return results

    def chunk_text(self, text: str, size: int = 150, overlap: int = 25) -> list[str]:
        """
        Розбиває текст на чанки по словах із перекриттям.
        size: кількість слів у чанку.
        overlap: кількість слів, що повторюються.
        """
        words = text.split()

        # Якщо текст коротший за розмір чанка, повертаємо його цілим
        if len(words) <= size:
            return [text]

        chunks = []
        # Крок (stride) — це скільки НОВИХ слів ми беремо в кожен наступний чанк
        stride = size - overlap

        for i in range(0, len(words), stride):
            # Беремо "вікно" слів
            chunk_slice = words[i: i + size]
            chunk = " ".join(chunk_slice)
            chunks.append(chunk)

            # Якщо ми вже охопили кінець тексту, виходимо
            if i + size >= len(words):
                break

        return chunks

    def remove_documents_by_date(self, source: str, target_date: datetime):
        # 1. Форматуємо дату ТАК САМО, як у вашому робочому CLI запиті
        date_str = target_date.strftime('%Y-%m-%d')

        with sqlite3.connect(self.db_path) as conn:
            # 2. Вмикаємо прагми ПЕРЕД виконанням запиту
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA recursive_triggers = ON;")

            cursor = conn.cursor()

            # 3. Виконуємо видалення
            cursor.execute("""
                DELETE FROM raw_docs 
                WHERE source = ? AND date(date) = ?
            """, (source, date_str))

            # 4. Перевіряємо, чи взагалі щось було видалено
            deleted_count = cursor.rowcount
            conn.commit()

        # 5. Скидаємо кеш
        self._matrix_cache.pop(source, None)
        self._matrix_cache.pop("__all__", None)

        print(f"Видалено записів: {deleted_count} для {source} за {date_str}")
        return deleted_count

    def add_document(self, text: str, source: str, date_str: datetime, metadata: dict = {}):
        """Метод для додавання нового документа в базу знань"""

        # 1. Розбиваємо на чанки (використовуємо твій метод chunk_text)
        text = clean_for_fts(text)
        chunks = self.chunk_text(text)

        # 2. Генеруємо ембеддінги для всіх чанків одним махом (це швидше)
        embeddings = self.model.encode(chunks, convert_to_numpy=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Зберігаємо основний документ
            cursor.execute(
                "INSERT INTO raw_docs (text, source, date, metadata) VALUES (?, ?, ?, ?)",
                (text, source, date_str, json.dumps(
                    metadata) if metadata else None)
            )

            doc_id = cursor.lastrowid

            # Зберігаємо чанки та їхні вектори
            for chunk, emb in zip(chunks, embeddings):
                cursor.execute(
                    "INSERT INTO rag_chunks (doc_id, text, embedding) VALUES (?, ?, ?)",
                    (doc_id, chunk, json.dumps(emb.tolist()))
                )
            conn.commit()

        # 3. ВАЖЛИВО: Після додавання знань треба скинути кеш матриці,
        # щоб наступний пошук побачив нові дані
        self._matrix_cache.pop(source, None)
        self._matrix_cache.pop("__all__", None)
        print(f"Додано документ: {source}, база оновлена.")
