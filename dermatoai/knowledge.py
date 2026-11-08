"""Base documental v2: SQLite persistente y BM25 local sin descarga obligatoria."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata
from filelock import FileLock
from .ingestion import pdf_pages, clean
from .preparation import prepare, PIPELINE_VERSION, CLASSES

STOP = set("a al con de del el en es esta este la las lo los para por que se su un una y o según sobre como qué cuál cuáles documento documentos pdf explica información".split())

def tokens(text):
    value = unicodedata.normalize("NFKD", text.lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", value) if len(w) > 1 and w not in STOP]

def split_text(text, size=1000, overlap=160):
    if not 0 <= overlap < size:
        raise ValueError("Solapamiento inválido.")
    text = clean(text)
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start + size // 2, end), text.rfind(" ", start + size // 2, end))
            if boundary > start:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            yield part
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

def read_document(path):
    body = path.read_bytes()
    if len(body) > 25 * 1024 * 1024:
        raise ValueError("Máximo 25 MiB por documento.")
    if path.suffix.lower() == ".pdf":
        return pdf_pages(body)
    if path.suffix.lower() in {".txt", ".md"}:
        text = clean(body.decode("utf-8-sig"))
        if len(text) < 30 or len(text) > 2_000_000:
            raise ValueError("El texto debe tener entre 30 y 2.000.000 de caracteres.")
        return [(None, text)]
    raise ValueError("Use PDF con texto, TXT o Markdown UTF-8.")

class KnowledgeBase:
    def __init__(self, database, documents):
        self.path, self.documents = Path(database), Path(documents)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.documents.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, filename TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL, organization TEXT NOT NULL, url TEXT NOT NULL, license TEXT NOT NULL,
                sha256 TEXT NOT NULL, updated TEXT NOT NULL, metadata TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS fragments(id TEXT PRIMARY KEY, source_id TEXT NOT NULL
                REFERENCES sources(id) ON DELETE CASCADE, page INTEGER, ordinal INTEGER,
                text TEXT NOT NULL, terms TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS embeddings(fragment_id TEXT PRIMARY KEY
                REFERENCES fragments(id) ON DELETE CASCADE, model_id TEXT NOT NULL, vector BLOB NOT NULL);
              CREATE TABLE IF NOT EXISTS document_pages(source_id TEXT NOT NULL
                REFERENCES sources(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, page INTEGER,
                raw_text TEXT NOT NULL, PRIMARY KEY(source_id,ordinal));
            ''')
        with FileLock(str(self.path) + '.migration.lock', timeout=20):
            with self.connect() as db:
                existing = {r['name'] for r in db.execute('PRAGMA table_info(fragments)')}
                if 'section' not in existing:
                    backup = self.path.with_name(self.path.name + '.pre_21.bak')
                    if not backup.exists():
                        original = sqlite3.connect(self.path)
                        target = sqlite3.connect(backup)
                        try: original.backup(target)
                        finally: original.close(); target.close()
                    for name,sqltype,default in [('section','TEXT',"'Sin preparar'"),('categories','TEXT',"'[]'"),
                          ('language','TEXT',"'und'"),('quality','TEXT',"'pending'"),('reasons','TEXT',"'[\"version_anterior\"]'"),('approved','INTEGER','0')]:
                        db.execute(f'ALTER TABLE fragments ADD COLUMN {name} {sqltype} NOT NULL DEFAULT {default}')
                source_columns = {r['name'] for r in db.execute('PRAGMA table_info(sources)')}
                for name in ('pipeline','preparation_report'):
                    if name not in source_columns:
                        db.execute(f"ALTER TABLE sources ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
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
            return [dict(r) for r in db.execute("SELECT s.*,COUNT(f.id) fragments, SUM(CASE WHEN f.quality='ready' OR f.approved=1 THEN 1 ELSE 0 END) ready_fragments FROM sources s LEFT JOIN fragments f ON f.source_id=s.id GROUP BY s.id ORDER BY s.filename")]
    def catalog(self):
        file = self.documents / "catalog.json"
        result = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
        if not isinstance(result, dict):
            raise ValueError("catalog.json debe ser un objeto por nombre de archivo.")
        return result
    def ingest(self, path, meta=None):
        path = Path(path).resolve()
        if path.parent != self.documents.resolve():
            raise ValueError("El archivo debe estar dentro de la carpeta de documentos.")
        meta = meta or {}
        if not isinstance(meta, dict):
            raise ValueError("Metadatos inválidos.")
        text_meta = {k: str(meta.get(k, ''))[:2000] for k in ('title','organization','url','license')}
        categories = meta.get('categories', [])
        if not isinstance(categories,list) or any(c not in CLASSES for c in categories):
            raise ValueError('categories debe ser una lista de ACK, BCC, MEL, NEV, SCC o SEK.')
        language = meta.get('language','auto')
        modality = meta.get('modality','not_specified')
        if language not in ('auto','es','en','und') or modality not in ('not_specified','clinical','dermoscopy','mixed'):
            raise ValueError('Idioma o modalidad documental inválidos.')
        meta = dict(text_meta,categories=sorted(set(categories)),language=language,modality=modality)
        meta["title"] = meta["title"].strip() or path.stem
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        ident = hashlib.sha256(path.name.encode()).hexdigest()[:24]
        meta_json = json.dumps(meta, sort_keys=True, ensure_ascii=False)
        with self.connect() as db:
            old = db.execute("SELECT * FROM sources WHERE id=? OR filename=?", (ident,path.name)).fetchone()
        if old and old["sha256"] == sha and old["metadata"] == meta_json and old["pipeline"] == PIPELINE_VERSION:
            return "unchanged"
        pages,parts,report = prepare(path,meta)
        with self.connect() as db:
            db.execute("DELETE FROM sources WHERE id=? OR filename=?", (ident,path.name))
            db.execute("INSERT INTO sources(id,filename,title,organization,url,license,sha256,updated,metadata,pipeline,preparation_report) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
               (ident,path.name,meta['title'],meta['organization'],meta['url'],meta['license'],sha,
                datetime.now(timezone.utc).isoformat(),meta_json,PIPELINE_VERSION,json.dumps(report,ensure_ascii=False)))
            db.executemany("INSERT INTO document_pages VALUES(?,?,?,?)",[(ident,i,page,text) for i,(page,text) in enumerate(pages)])
            db.executemany("INSERT INTO fragments(id,source_id,page,ordinal,text,terms,section,categories,language,quality,reasons,approved) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
               [(f"{ident}:{i}",ident,part['page'],i,part['text'],json.dumps(tokens(part['section']+' '+part['text']+' '+' '.join(part['categories']))),
                 part['section'],json.dumps(part['categories']),part['language'],part['quality'],json.dumps(part['reasons']),0) for i,part in enumerate(parts)])
        return "added" if old is None else "updated"
    def sync(self):
        report = {"added": [], "updated": [], "unchanged": [], "errors": []}
        with FileLock(str(self.path) + ".lock", timeout=20):
            catalog = self.catalog()
            for path in sorted(self.documents.iterdir()):
                if path.suffix.lower() not in {".pdf", ".txt", ".md"} or path.name == "LEEME.md":
                    continue
                try:
                    report[self.ingest(path, catalog.get(path.name, {}))].append(path.name)
                except Exception as exc:
                    report["errors"].append({"file": path.name, "error": str(exc)[:260]})
        return report
    def save_upload(self, filename, body, meta):
        name = Path(filename.replace("\\", "/")).name
        if not re.fullmatch(r"[\w .()-]+\.(pdf|txt|md)", name, flags=re.I):
            raise ValueError("Nombre inválido. Use letras, números, espacios, guiones y extensión PDF/TXT/MD.")
        if len(body) > 25 * 1024 * 1024:
            raise ValueError("Máximo 25 MiB.")
        with FileLock(str(self.path) + ".lock", timeout=20):
            catalog = self.catalog()
            target = self.documents / name
            if target.is_symlink():
                raise ValueError("No se admiten enlaces simbólicos.")
            previous = target.read_bytes() if target.exists() else None
            target.write_bytes(body)
            try:
                status = self.ingest(target, meta)
            except Exception:
                if previous is None:
                    target.unlink()
                else:
                    target.write_bytes(previous)
                raise
            catalog[name] = meta
            temp = self.documents / "catalog.json.tmp"
            temp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.documents / "catalog.json")
        return status
    def delete(self, ident):
        with FileLock(str(self.path) + ".lock", timeout=20):
            with self.connect() as db:
                row = db.execute("SELECT filename FROM sources WHERE id=?", (ident,)).fetchone()
                if row:
                    (self.documents / row[0]).unlink(missing_ok=True)
                    catalog = self.catalog()
                    catalog.pop(row[0], None)
                    temp = self.documents / "catalog.json.tmp"
                    temp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
                    temp.replace(self.documents / "catalog.json")
                db.execute("DELETE FROM sources WHERE id=?", (ident,))
    def rows(self, source_ids=None, *, include_review=False, categories=None, languages=None):
        if source_ids == []:
            return []
        where, params = "", []
        if source_ids is not None:
            where = " WHERE s.id IN (" + ",".join("?" for _ in source_ids) + ")"
            params = list(source_ids)
        with self.connect() as db:
            found = [dict(r) for r in db.execute("SELECT f.*,s.title,s.organization,s.url,s.filename,s.sha256,s.metadata FROM fragments f JOIN sources s ON s.id=f.source_id" + where + " ORDER BY s.filename,f.ordinal", params)]
        result = []
        for row in found:
            if not include_review and row['quality'] != 'ready' and not row['approved']:
                continue
            row['categories'] = json.loads(row['categories'])
            row['reasons'] = json.loads(row['reasons'])
            row['modality'] = json.loads(row['metadata']).get('modality','not_specified')
            if categories and not set(categories).intersection(row['categories']):
                continue
            if languages and row['language'] not in languages:
                continue
            result.append(row)
        return result
    def raw_pages(self, source_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM document_pages WHERE source_id=? ORDER BY ordinal',(source_id,))]
    def approve(self, fragment_id, approved):
        with self.connect() as db:
            db.execute('UPDATE fragments SET approved=? WHERE id=?',(int(bool(approved)),fragment_id))
    def revision(self):
        with self.connect() as db:
            return hashlib.sha256(json.dumps([tuple(r) for r in db.execute('SELECT id,quality,approved FROM fragments ORDER BY id')]).encode()).hexdigest()

    def build_vectors(self, encoder):
        import numpy as np
        rows = self.rows(include_review=True)
        if not rows:
            raise ValueError("Primero incorpore documentos.")
        vectors = np.asarray(encoder.encode([r["text"] for r in rows]), dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if len(vectors) != len(rows) or not np.isfinite(vectors).all() or (norms == 0).any():
            raise ValueError("Embeddings inválidos.")
        vectors /= norms
        with self.connect() as db:
            db.execute("DELETE FROM embeddings")
            db.executemany("INSERT INTO embeddings VALUES(?,?,?)", [(r["id"], encoder.model_id, v.tobytes()) for r, v in zip(rows, vectors)])
        return len(rows)
    def retrieve(self, query, kinds=("pdf",), source_ids=None, k=6, min_score=0.0, method="lexical", encoder=None, *, include_review=False, categories=None, languages=None):
        if not kinds or source_ids == []:
            return []
        if not 1 <= k <= 24 or method not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("Método o número de fragmentos inválido.")
        rows = self.rows(source_ids,include_review=include_review,categories=categories,languages=languages)
        if not rows:
            return []
        words = tokens(query)
        if not words:
            return []
        bags = [Counter(json.loads(r["terms"])) for r in rows]
        avg = sum(sum(b.values()) for b in bags) / max(1, len(bags))
        frequencies = {w: sum(w in b for b in bags) for w in set(words)}
        lex = []
        for bag, row in zip(bags, rows):
            score = 0.0
            for word in set(words):
                df = frequencies[word]
                idf = math.log(1 + (len(rows) - df + .5) / (df + .5))
                tf = bag[word]
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * sum(bag.values()) / max(avg, 1)))
            lex.append(score)
        semantic = [None] * len(rows)
        if method != "lexical":
            if encoder is None:
                raise ValueError("Cargue el encoder y calcule los embeddings antes de usar búsqueda semántica.")
            import numpy as np
            with self.connect() as db:
                vectors = {r["fragment_id"]: r["vector"] for r in db.execute("SELECT * FROM embeddings WHERE model_id=?", (encoder.model_id,))}
            if any(r["id"] not in vectors for r in rows):
                raise ValueError("Faltan embeddings o cambió el encoder. Recalcule los vectores en Documentos.")
            q = np.asarray(encoder.encode([query])[0], dtype="float32")
            q /= max(float(np.linalg.norm(q)), 1e-12)
            for i, row in enumerate(rows):
                v = np.frombuffer(vectors[row["id"]], dtype="float32")
                if v.shape != q.shape:
                    raise ValueError("Dimensión de embedding incompatible.")
                semantic[i] = float(v @ q)
        if method == "lexical":
            scores = lex
        elif method == "semantic":
            scores = semantic
        else:
            scores = [0.0] * len(rows)
            for ranking in (lex, semantic):
                for rank, i in enumerate(sorted(range(len(rows)), key=lambda i: -ranking[i]), 1):
                    if ranking[i] > (0 if ranking is lex else min_score):
                        scores[i] += 1 / (60 + rank)
        hits = []
        for i in sorted(range(len(rows)), key=lambda i: -scores[i]):
            if scores[i] <= 0 or (method == "semantic" and scores[i] < min_score):
                continue
            r = rows[i]
            if any(h["text"] == r["text"] for h in hits):
                continue
            hits.append({key: r[key] for key in ("id", "source_id", "page", "text", "title", "organization", "url", "filename", "sha256", "section", "categories", "language", "quality", "reasons", "modality")})
            hits[-1].update(kind="pdf", origin=r["url"] or r["filename"], score=float(scores[i]), citation_id=f"S{len(hits)}")
            if len(hits) == k:
                break
        return hits
    def overview(self, source_ids, limit=12, *, include_review=False, categories=None, languages=None):
        """Panorama de pasajes aptos, priorizando resumen/definición y diversidad de secciones."""
        rows = self.rows(source_ids,include_review=include_review,categories=categories,languages=languages)
        rows.sort(key=lambda r:(0 if re.search(r'resumen|definici|introducci|generalidades',r['section'],re.I) else 1,r['ordinal']))
        result,seen,documents = [],set(),list(dict.fromkeys(r['source_id'] for r in rows))
        while len(result)<limit:
            advanced=False
            for source in documents:
                candidates=[r for r in rows if r['source_id']==source and r['id'] not in seen]
                chosen=next((r for r in candidates if (source,r['section']) not in {(v['source_id'],v['section']) for v in result}),candidates[0] if candidates else None)
                if chosen:
                    seen.add(chosen['id']);result.append(chosen);advanced=True
                    if len(result)==limit:break
            if not advanced:break
        return [dict(row,citation_id=f'S{i}',kind='pdf',origin=row['url'] or row['filename'],score=0.0) for i,row in enumerate(result,1)]
