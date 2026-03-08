-- Таблиця для оригінальних документів
CREATE TABLE IF NOT EXISTS raw_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT DEFAULT 'manual',
    lang TEXT DEFAULT 'uk',
    date TEXT DEFAULT (datetime('now')),
    text TEXT NOT NULL,
    hash TEXT UNIQUE, -- Захист від дублів
    is_indexed INTEGER DEFAULT 0,
    metadata TEXT -- Для JSON-метаданих (теги, лінки)
);

-- Таблиця для чанків
CREATE TABLE IF NOT EXISTS rag_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    chunk_order INTEGER, -- Порядок у документі
    text TEXT NOT NULL,
    embedding BLOB, -- Зберігати як BLOB швидше, ніж як JSON-текст
    FOREIGN KEY(doc_id) REFERENCES raw_docs(id) ON DELETE CASCADE
);

-- Створюємо індекс для швидкого пошуку/видалення
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON rag_chunks(doc_id);


CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts 
USING fts5(
    text, 
    content='rag_chunks', 
    content_rowid='id',
    tokenize = "unicode61 separators '«»-/•·'"
);

CREATE TRIGGER IF NOT EXISTS rag_chunks_ai AFTER INSERT ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS rag_chunks_ad AFTER DELETE ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rag_chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

-- INSERT INTO rag_chunks_fts(rowid, text) SELECT id, text FROM rag_chunks;
