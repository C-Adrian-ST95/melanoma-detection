import hashlib
from contextlib import contextmanager
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .ingestion import chunks

class LocalEncoder:
    def __init__(self, directory):
        from sentence_transformers import SentenceTransformer
        path = Path(directory).resolve()
        if not (path / "modules.json").exists():
            raise FileNotFoundError("Falta el modelo de embeddings. Ejecute python -m scripts_dermatoai.download_embeddings.")
        h = hashlib.sha256()
        for file in sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts):
            h.update(str(file.relative_to(path)).encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
        self.model_id = "sentence-transformer-sha256:" + h.hexdigest()
        self.model = SentenceTransformer(str(path), device="cpu", local_files_only=True, trust_remote_code=False, tokenizer_kwargs={"fix_mistral_regex": True})
    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

class Index:
    def __init__(self, path, encoder):
        self.path, self.encoder = Path(path), encoder
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    title TEXT NOT NULL, origin TEXT NOT NULL, sha256 TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS fragments(id TEXT PRIMARY KEY, source_id TEXT NOT NULL
                    REFERENCES sources(id) ON DELETE CASCADE, page INTEGER, text TEXT NOT NULL, vector BLOB NOT NULL);
            """)
            row = db.execute("SELECT value FROM settings WHERE key='encoder'").fetchone()
            if row and row[0] != encoder.model_id:
                raise ValueError("El modelo de embeddings cambió: utilice otro índice o reconstruya el actual.")
            db.execute("INSERT OR IGNORE INTO settings VALUES('encoder', ?)", (encoder.model_id,))
    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()
    def list_sources(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT s.*, COUNT(f.id) AS fragments FROM sources s LEFT JOIN fragments f ON s.id=f.source_id GROUP BY s.id ORDER BY s.created")]
    def add(self, kind, title, origin, pages, sha256):
        if kind not in {"pdf", "organization"}:
            raise ValueError("Tipo de fuente inválido.")
        if not title.strip() or not origin.strip():
            raise ValueError("La fuente necesita título y procedencia.")
        source_id = hashlib.sha256((kind + origin + sha256).encode()).hexdigest()[:24]
        parts = [(page, text) for page, content in pages for text in chunks(content)]
        if not parts or len(parts) > 5000:
            raise ValueError("Documento vacío o demasiado grande para este índice local.")
        vectors = np.asarray(self.encoder.encode([t for _, t in parts]), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(parts) or not np.isfinite(vectors).all():
            raise ValueError("Embeddings inválidos.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding nulo.")
        vectors /= norms
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key='dimension'").fetchone()
            if row and int(row[0]) != vectors.shape[1]:
                raise ValueError("Dimensión incompatible.")
            db.execute("INSERT OR IGNORE INTO settings VALUES('dimension', ?)", (str(vectors.shape[1]),))
            db.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?)", (source_id, kind, title, origin, sha256, datetime.now(timezone.utc).isoformat()))
            db.execute("DELETE FROM fragments WHERE source_id=?", (source_id,))
            db.executemany("INSERT INTO fragments VALUES(?,?,?,?,?)", [(f"{source_id}:{i}", source_id, page, text, vec.tobytes()) for i, ((page, text), vec) in enumerate(zip(parts, vectors))])
        return source_id
    def delete(self, source_id):
        with self.connect() as db:
            db.execute("DELETE FROM sources WHERE id=?", (source_id,))
    def retrieve(self, query, kinds, source_ids=None, k=4, min_score=0.25):
        if not kinds or source_ids == []:
            return []
        if not set(kinds) <= {"pdf", "organization"} or not 1 <= k <= 20:
            raise ValueError("Filtro o top-k inválido.")
        where = "s.kind IN (" + ",".join("?" for _ in kinds) + ")"
        params = list(kinds)
        if source_ids is not None:
            where += " AND s.id IN (" + ",".join("?" for _ in source_ids) + ")"
            params += list(source_ids)
        with self.connect() as db:
            rows = db.execute("SELECT f.*,s.kind,s.title,s.origin FROM fragments f JOIN sources s ON f.source_id=s.id WHERE " + where, params).fetchall()
        if not rows:
            return []
        q = np.asarray(self.encoder.encode([query])[0], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        vectors = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        if vectors.shape[1] != q.shape[0]:
            raise ValueError("Dimensión de consulta incompatible con el índice.")
        scores = vectors @ q
        selected = []
        for i in np.argsort(-scores, kind="stable"):
            if float(scores[i]) < min_score:
                break
            r = rows[int(i)]
            # Evitar fragmentos vecinos casi idénticos en un top-k muy pequeño.
            if any(x["source_id"] == r["source_id"] and x["text"] == r["text"] for x in selected):
                continue
            selected.append({key: r[key] for key in ("id", "source_id", "page", "text", "kind", "title", "origin")})
            selected[-1].update(score=float(scores[i]), citation_id=f"S{len(selected)}")
            if len(selected) == k:
                break
        return selected
