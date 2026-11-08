"""Casos artificiales que comprueban estructura y exclusión; no evaluación clínica."""
import io
import json
from pathlib import Path
import sqlite3
import pytest
from reportlab.pdfgen.canvas import Canvas
from dermatoai.knowledge import KnowledgeBase
from dermatoai.preparation import prepare, sentence_chunks, clean_lines, categories_for
from dermatoai.service import respond
from dermatoai.llm import LLMConfig

@pytest.fixture
def kb(tmp_path):return KnowledgeBase(tmp_path/'rag.sqlite3',tmp_path/'docs')

def upload(kb,text,name='guia.md',**meta):
    kb.save_upload(name,text.encode(),dict(title='Documento artificial',organization='Prueba',**meta))
    return kb.list_sources()[0]['id']


def test_dense_statistics_stay_saved_but_do_not_answer_normal_query(kb):
    ident=upload(kb,'# Definición\n\nMelanoma es aquí una palabra de prueba para comprobar recuperación documental.\n\n# Estadísticas\n\n' + 'Melanoma AL 30,270 AZ 42,670 AR 19,100 CA 193,880 '*18)
    rows=kb.rows([ident],include_review=True)
    assert any(r['quality']=='review' and 'tabla_o_estadisticas_densas' in r['reasons'] for r in rows)
    normal=kb.retrieve('melanoma',source_ids=[ident])
    assert normal and all('193,880' not in h['text'] for h in normal)
    excluded=next(r for r in rows if r['quality']=='review')
    assert kb.raw_pages(ident) and '193,880' in kb.raw_pages(ident)[0]['raw_text']
    kb.approve(excluded['id'],True)
    assert any(r['id']==excluded['id'] for r in kb.rows([ident]))
    kb.approve(excluded['id'],False)
    assert all(r['id']!=excluded['id'] for r in kb.rows([ident]))


def test_sections_never_merge_clinical_text_with_bibliography(kb):
    ident=upload(kb,'# Diagnóstico\n\nTexto artificial sobre nevos para probar conservación de la sección y la referencia.\n\n# Bibliografía\n\nAutor de prueba. Nevos en artículos inexistentes: una cita artificial para pruebas de software.')
    rows=kb.rows([ident],include_review=True)
    assert {r['section'] for r in rows}=={'Diagnóstico','Bibliografía'}
    assert all(r['quality']=='review' for r in rows if r['section']=='Bibliografía')
    assert all('inexistentes' not in h['text'] for h in kb.retrieve('nevos'))


def test_language_and_category_filters_before_ranking(kb):
    upload(kb,'# Definición\n\nEl melanoma y la piel se mencionan en este pasaje artificial de prueba.\n\n# Abstract\n\nThe melanoma of the skin is described in this artificial document for software testing.',categories=['MEL'])
    kb.save_upload('bcc.txt','El carcinoma basocelular de la piel se menciona en esta prueba artificial de extracción.'.encode(),{'categories':['BCC']})
    hits=kb.retrieve('melanoma',categories=['MEL'],languages=['es'])
    assert hits and all(h['language']=='es' and 'MEL' in h['categories'] for h in hits)
    assert not kb.retrieve('melanoma',categories=['SCC'])
    assert not kb.retrieve('melanoma',languages=['en'],categories=['BCC'])


def test_explicit_spanish_document_still_detects_english_abstract(kb):
    ident=upload(kb,'# Abstract\n\nThe melanoma of the skin is discussed in this artificial document for software testing.',language='es')
    assert kb.rows([ident])[0]['language']=='en'
    report=json.loads(kb.list_sources()[0]['preparation_report'])
    assert any('idiomas' in s for s in report['warnings'])


def test_category_aliases_and_unclassified_fragments():
    assert set(categories_for('Carcinoma epidermoide y queratosis seborreica'))=={'SCC','SEK'}
    assert categories_for('La biblioteca abre los lunes.')==[]


def test_repeated_headers_removed_and_original_kept():
    pages=[(1,'REVISTA ARTIFICIAL\nDiagnóstico\nEl texto de la primera página es una prueba.\nPágina 1'),
           (2,'REVISTA ARTIFICIAL\nEl texto de la segunda página es otra prueba.\nPágina 2')]
    cleaned=clean_lines(pages)
    assert 'REVISTA ARTIFICIAL' not in cleaned[0][1] and 'Diagnóstico' in cleaned[0][1]
    assert 'REVISTA ARTIFICIAL' in pages[0][1]


def test_no_crossing_of_sentence_or_section_boundaries(kb):
    upload(kb,'# Definición\n\n' + 'Una frase artificial completa para probar el contexto conservado. '*70 + '\n\n# Limitaciones\n\nOtra sección de prueba que debe conservar su título y no mezclarse.')
    rows=kb.rows()
    assert len(rows)>2 and all(len(r['text'])<=1500 for r in rows)
    assert all(r['text'].endswith('.') for r in rows)
    assert not any('Otra sección' in r['text'] for r in rows if r['section']=='Definición')


def test_extract_fallback_is_short_and_full_evidence_is_available(kb):
    upload(kb,'# Resumen\n\n'+'El melanoma se menciona como una palabra artificial en esta prueba documental. '*80)
    response=respond('melanoma',LLMConfig(),kb,None,k=6)
    assert response['hits'] and not response['generated']
    assert 'hasta tres pasajes abreviados' in response['markdown']
    assert len(response['markdown'])<4000
    assert any(len(h['text'])>420 for h in response['hits'])


def test_overview_avoids_unreviewed_tables_and_diversifies(kb):
    upload(kb,'# Resumen\n\nUn resumen artificial de melanoma para las pruebas.\n\n# Discusión\n\nUna discusión artificial de la evidencia de prueba.\n\n# Referencias\n\nUna referencia artificial que no debe aparecer como contenido de respuesta.')
    rows=kb.overview(None)
    assert rows and not any(r['section']=='Referencias' for r in rows)


def test_legacy_database_is_backed_up_and_reprocessed(tmp_path):
    path=tmp_path/'old.sqlite3';docs=tmp_path/'docs';docs.mkdir()
    db=sqlite3.connect(path)
    db.executescript('''CREATE TABLE sources(id TEXT PRIMARY KEY,filename TEXT UNIQUE NOT NULL,title TEXT NOT NULL,organization TEXT NOT NULL,url TEXT NOT NULL,license TEXT NOT NULL,sha256 TEXT NOT NULL,updated TEXT NOT NULL,metadata TEXT NOT NULL);
    CREATE TABLE fragments(id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,page INTEGER,ordinal INTEGER,text TEXT NOT NULL,terms TEXT NOT NULL);
    INSERT INTO sources VALUES('old','old.txt','Test','','','','oldsha','oldtime','{}');
    INSERT INTO fragments VALUES('old:0','old',1,0,'Texto antiguo sin preparar de una prueba de software.','[]');''')
    db.commit();db.close()
    kb=KnowledgeBase(path,docs)
    assert path.with_name(path.name+'.pre_21.bak').exists()
    assert kb.rows()==[] and kb.rows(include_review=True)[0]['quality']=='pending'
    (docs/'old.txt').write_text('Texto nuevo de una prueba artificial de melanoma para verificar reindexado.')
    # El ID legado puede diferir, la aplicación siempre usó SHA del nombre; usar el mismo contrato.
    kb.sync()
    assert kb.retrieve('melanoma')


def test_two_column_pdf_preserves_reading_order_and_pages(kb):
    b=io.BytesIO();c=Canvas(b,pagesize=(600,800))
    for i in range(12):
        c.drawString(40,720-i*16,f'Izquierda linea {i} contiene texto artificial de prueba.')
        c.drawString(330,720-i*16,f'Derecha linea {i} contiene otras palabras de prueba.')
    c.save()
    kb.save_upload('columns.pdf',b.getvalue(),{})
    source=kb.list_sources()[0]
    raw=kb.raw_pages(source['id'])[0]
    assert raw['page']==1
    assert raw['raw_text'].find('Izquierda linea 11')<raw['raw_text'].find('Derecha linea 0')


def test_corrupt_update_keeps_page_text_and_quality(kb):
    ident=upload(kb,'Una prueba de melanoma con texto suficientemente largo para conservar la versión válida.')
    before=kb.raw_pages(ident)
    with pytest.raises(ValueError):kb.save_upload('guia.md',b'bad',{})
    assert kb.raw_pages(ident)==before and kb.retrieve('melanoma')
