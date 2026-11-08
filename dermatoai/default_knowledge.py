"""Carga incremental de PDF elegidos por el usuario en SQLite."""
import hashlib
import json
from pathlib import Path
from filelock import FileLock
from .ingestion import pdf_pages

def load_defaults(index, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "catalog.json"
    catalog = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    if not isinstance(catalog, dict):
        raise ValueError("catalog.json debe ser un objeto: nombre.pdf -> metadatos.")
    result = {"added": [], "unchanged": [], "errors": []}
    with FileLock(str(index.path) + ".defaults.lock", timeout=15):
        existing = index.list_sources()
        for file in sorted(directory.glob("*.pdf")):
            try:
                if not file.resolve().is_relative_to(directory):
                    raise ValueError("PDF fuera de la carpeta autorizada.")
                meta = catalog.get(file.name, {})
                if not isinstance(meta, dict):
                    raise ValueError("Metadatos inválidos.")
                origin = "predeterminados/" + file.name
                if meta.get("url"):
                    origin += " | " + str(meta["url"])
                body = file.read_bytes()
                sha = hashlib.sha256(body).hexdigest()
                title = " — ".join(str(v) for v in [meta.get("organization"), meta.get("title", file.stem)] if v)
                prefix = "predeterminados/" + file.name
                old = [s for s in existing if s["kind"] == "pdf" and (s["origin"] == prefix or s["origin"].startswith(prefix + " | "))]
                if any(s["sha256"] == sha and s["title"] == title and s["origin"] == origin for s in old):
                    result["unchanged"].append(file.name)
                    continue
                source_id = index.add("pdf", title, origin, pdf_pages(body), sha)
                for source in old:
                    if source["id"] != source_id:
                        index.delete(source["id"])
                result["added"].append(file.name)
            except Exception as exc:
                result["errors"].append({"file": file.name, "error": str(exc)[:240]})
    return result
