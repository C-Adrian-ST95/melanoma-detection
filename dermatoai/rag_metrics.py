"""Evaluación de recuperación; no sustituye evaluación humana de respuestas."""
import math
import hashlib
import json
from datetime import datetime, timezone
import time

def evaluate_cases(kb, cases, k=6, method="lexical", encoder=None, *, include_review=False, categories=None, languages=None):
    if not cases or len({c.get('id') for c in cases}) != len(cases):
        raise ValueError("Use casos con identificadores únicos.")
    source_map = {s['filename']:s['id'] for s in kb.list_sources()}
    details = []
    for case in cases:
        if not case.get('id') or not str(case.get('question', '')).strip():
            raise ValueError("Cada caso requiere id y question.")
        unanswerable = case.get('unanswerable', False)
        expected = case.get('relevant', [])
        if not unanswerable and not expected:
            raise ValueError("Anote al menos un archivo y página pertinente, o marque unanswerable.")
        for item in expected:
            if item.get('filename') not in source_map or 'page' not in item:
                raise ValueError("La anotación relevante requiere filename indexado y page (null para TXT).")
        selected = case.get('allowed_files')
        if selected is not None and any(f not in source_map for f in selected):
            raise ValueError("allowed_files contiene un documento no indexado.")
        ids = [source_map[f] for f in selected] if selected is not None else None
        start = time.perf_counter()
        hits = [] if method == 'none' else kb.retrieve(case['question'], source_ids=ids, k=k, method=method, encoder=encoder,include_review=include_review,categories=categories,languages=languages)
        latency = (time.perf_counter()-start)*1000
        relevant = {(i['filename'], i['page']) for i in expected}
        ranked = []
        # Unidad de evaluación: documento-página. No contar varias veces fragmentos solapados.
        for hit in hits:
            unit = (hit['filename'], hit['page'])
            if unit not in ranked:
                ranked.append(unit)
        flags = [unit in relevant for unit in ranked]
        found = sum(flags)
        dcg = sum(1/math.log2(i+2) for i,v in enumerate(flags) if v)
        ideal = sum(1/math.log2(i+2) for i in range(min(len(relevant), k)))
        details.append(dict(id=case['id'], question=case['question'], unanswerable=unanswerable,
             recall=found/len(relevant) if relevant else None,
             precision=found/len(ranked) if ranked and relevant else (0.0 if relevant else None),
             reciprocal_rank=next((1/(i+1) for i,v in enumerate(flags) if v),0.0) if relevant else None,
             ndcg=dcg/ideal if ideal else (0.0 if relevant else None),
             correctly_empty=not bool(hits) if unanswerable else None, latency_ms=latency,
             hits=[{key:h[key] for key in ('filename','page','score','id')} for h in hits]))
    result = {'method':method,'k':k,'n':len(cases),'unit':'document-page, deduplicated after top-k chunks',
              'note':'Métricas de recuperación; no prueban corrección clínica ni fidelidad semántica de las respuestas.', 'cases':details}
    result['corpus'] = [{k:s[k] for k in ('filename','sha256','metadata')} for s in kb.list_sources()]
    result['cases_sha256'] = hashlib.sha256(json.dumps(cases,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    result['filters'] = {'include_review':include_review,'categories':categories,'languages':languages}
    result['encoder_id'] = getattr(encoder,'model_id',None)
    result['created'] = datetime.now(timezone.utc).isoformat()
    for metric in ('recall','precision','reciprocal_rank','ndcg','correctly_empty','latency_ms'):
        values = [r[metric] for r in details if r[metric] is not None]
        result[metric] = sum(values)/len(values) if values else None
    return result
