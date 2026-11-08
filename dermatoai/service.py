"""Consultas documentales independientes de la inferencia y de la lectura visual."""
from datetime import datetime, timezone
import json
import re
import time
from .agent import SYSTEM, validate_claims, escape_text, base_report
from .llm import complete
from .preparation import categories_for, normalize
from .settings import NAMES, NOTICE

def query_for(question, result=None, history=()):
    question = str(question).strip()[:4000]
    query = question
    # Solo la pregunta previa resuelve referencias; no reutilizar toda la charla como evidencia.
    if history and re.search(r"\b(eso|esa|ese|anterior|ampl[ií]a|detalla|expl[ií]calo|m[aá]s detalle)\b", question, re.I):
        last = history[-1]
        query += " " + (last.get("question", "") if isinstance(last, dict) else str(last))[:800]
    if result and re.search(r"\b(resultado|predicci[oó]n|categor[ií]a|puntuaci[oó]n|modelo|informe|esta lesi[oó]n)\b", question, re.I):
        query += " " + NAMES[result["top_class"]] + " " + result["top_class"]
    return query

def comparison_context(query):
    """Ayuda de búsqueda para comparaciones explícitas; no interpreta la imagen."""
    topics = categories_for(query)
    comparison = re.search(r"\b(compar\w*|diferencias?|disting\w*|versus|vs)\b", normalize(query))
    return topics if comparison and len(topics) > 1 else []

def comparison_hits(hits, topics, query, k):
    # Evitar que coincidencias semánticas de otra categoría o de tratamiento llenen una comparación.
    treatment_requested = re.search(r"\b(trat\w*|terapi\w*|farmac\w*|medicamento\w*)\b", normalize(query))
    candidates = []
    for rank, hit in enumerate(hits):
        direct = set(categories_for(hit.get("section", "") + " " + hit["text"]))
        if direct and not direct.intersection(topics):
            continue
        section = normalize(hit.get("section", ""))
        if not treatment_requested and re.search(r"\b(tratamiento\w*|terapi\w*)\b", section):
            continue
        relevant_section = bool(re.search(r"definici|caracter|diagnost|diferenc|criterio", section))
        candidates.append((-len(direct.intersection(topics)), -int(relevant_section), rank, hit))
    chosen = [item[-1] for item in sorted(candidates, key=lambda v: v[:3])[:k]]
    return [dict(hit, citation_id=f"S{i}") for i, hit in enumerate(chosen, 1)]

def respond(question, config, index, source_ids, *, result=None, history=(), detail="Detallada", k=6,
            method="lexical", encoder=None, overview=False, use_rag=True, categories=None, languages=None, include_review=False):
    if not str(question).strip():
        raise ValueError("Escriba una pregunta.")
    started = time.perf_counter()
    query = query_for(question, result, history)
    topics = comparison_context(query) if not overview else []
    retrieval_categories = categories or topics or None
    hits, warning = [], ""
    total = len(index.rows(source_ids,categories=categories,languages=languages,include_review=include_review)) if use_rag else 0
    if use_rag:
        hits = index.overview(source_ids, limit=12,categories=categories,languages=languages,include_review=include_review) if overview else index.retrieve(query, source_ids=source_ids, k=24 if topics else k, method=method, encoder=encoder, min_score=.25,categories=retrieval_categories,languages=languages,include_review=include_review)
        if topics:
            hits = comparison_hits(hits, topics, query, k)
    retrieval_ms = 1000 * (time.perf_counter() - started)
    claims = None
    if hits and config.provider != "Informe local (sin LLM)":
        payload = {"question": str(question)[:4000], "detail": detail,
                   "model_result": result, "evidence": hits,
                   "history": [{"question": str(h.get("question", ""))[:600],
                                "answer": str(h.get("markdown", ""))[:1200]} for h in history[-3:] if isinstance(h, dict)]}
        try:
            claims = validate_claims(complete(config, SYSTEM, payload), hits)
        except (ValueError, RuntimeError, PermissionError, KeyError, TypeError) as exc:
            warning = "No se obtuvo una respuesta generada verificable: " + str(exc)[:220] + " Se muestran los fragmentos."
    sections = []
    if overview:
        sections.append(f"Cobertura documental: {len(hits)} de {total} fragmentos. " +
                        ("Se utiliza todo el texto indexado seleccionado." if len(hits) == total else "Es una selección de pasajes aptos por documento y sección; no un resumen integral de todos los PDF."))
    if not use_rag:
        sections.append("RAG desactivado: no se consultaron documentos. Active Usar documentos para obtener una explicación con fuentes.")
    elif source_ids == []:
        sections.append("No hay documentos seleccionados. Incorpore y seleccione fuentes en Documentos.")
    elif not hits:
        sections.append("No se encontró evidencia pertinente para esta pregunta en los documentos seleccionados. Revise el texto extraído, los filtros de categoría/idioma y los fragmentos apartados en Documentos.")
    elif claims is not None:
        sections.extend(escape_text(c["text"]) + " " + " ".join(f"[{i}]" for i in c["citations"]) for c in claims["scientific"])
        if not claims["scientific"]:
            sections.append("Las fuentes recuperadas no aportaron soporte suficiente para responder la consulta.")
        if claims["relation"] and result:
            sections.append("### Relación con el resultado experimental")
            sections.extend(escape_text(c["text"]) + " " + " ".join(f"[{i}]" for i in c["citations"]) for c in claims["relation"])
    else:
        sections.append("### Evidencia documental extraída")
        sections.append("Modo extractivo: se muestran pasajes originales; no se ha generado una explicación con un LLM.")
        sections.extend(escape_text(h["text"][:420]) + ("…" if len(h["text"])>420 else "") + f" [{h['citation_id']}]" for h in hits[:3])
        next_step = ("El LLM seleccionado no pudo completar una respuesta válida; revise el aviso y pulse Probar generación."
                     if warning else "Para una explicación redactada, conecte y seleccione un LLM.")
        sections.append("Se muestran hasta tres pasajes abreviados. Abra las referencias para leer el texto completo. " + next_step)
    if hits:
        sections.append("### Referencias de esta respuesta")
        for h in hits:
            loc = f"página {h['page']}" if h["page"] else "texto sin paginación"
            sections.append(f"[{h['citation_id']}] {escape_text(h.get('organization') or 'Institución no registrada')} · {escape_text(h['title'])}; {escape_text(h.get('section','sección no registrada'))}; {loc}; {escape_text(h['origin'])}.")
    return {"question": str(question)[:4000], "markdown": "\n\n".join(sections), "hits": hits,
            "warning": warning, "generated": claims is not None, "provider": config.provider, "model": config.model,
            "query": query, "method": method, "detail": detail, "retrieval_ms": round(retrieval_ms, 2),
            "elapsed_ms": round(1000 * (time.perf_counter() - started), 2),
            "created": datetime.now(timezone.utc).isoformat(), "claims": claims,
            "filters": {"categories": categories, "comparison_topics": topics, "languages": languages, "include_review": include_review},
            "coverage": {"selected_fragments": total, "used_fragments": len(hits), "overview": overview}}

def make_report(response, result=None, visual=None):
    sections = [base_report(result)] if result else ["# Consulta documental DermatoAI"]
    if visual:
        sections += ["### Descripción visual experimental", visual["markdown"]]
    sections += ["### Información de los documentos seleccionados", response["markdown"], "### Limitaciones", NOTICE,
                 "Las citas se comprueban por identificador y coincidencia literal. La fidelidad de las paráfrasis requiere revisión humana. El contexto documental no modifica las puntuaciones del clasificador."]
    return "\n\n".join(sections)
