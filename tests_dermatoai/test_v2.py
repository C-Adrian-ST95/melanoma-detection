"""Pruebas de software con documentos artificiales; no evaluación médica."""
import io
import json
import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from dermatoai.knowledge import KnowledgeBase
from dermatoai.service import respond, query_for, make_report
from dermatoai.llm import LLMConfig, complete
from dermatoai.vision import describe
from dermatoai.rag_metrics import evaluate_cases
from tests_dermatoai.test_rag_agent import TestEncoder, prediction

@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(tmp_path/'db.sqlite3',tmp_path/'docs')

def seed(kb, name='one.txt', text='Melanoma evidencia artificial para verificar la recuperacion documental en pruebas.'):
    kb.save_upload(name,text.encode(),{'title':name,'organization':'Institucion de prueba','license':'synthetic'})
    return next(s['id'] for s in kb.list_sources() if s['filename']==name)

def test_documents_work_without_encoder_and_reopen(kb):
    seed(kb)
    reopened=KnowledgeBase(kb.path,kb.documents)
    hits=reopened.retrieve('melanoma')
    assert hits and hits[0]['filename']=='one.txt'
    assert reopened.sync()['unchanged']==['one.txt']

def test_pdf_pages_inspectable(kb):
    output=io.BytesIO();c=Canvas(output)
    c.drawString(20,700,'Primera pagina de un documento sintetico para pruebas de software.');c.showPage()
    c.drawString(20,700,'Melanoma evidencia artificial en la segunda pagina del documento.');c.save()
    kb.save_upload('two.pdf',output.getvalue(),{'title':'Test'})
    assert kb.retrieve('melanoma')[0]['page']==2

def test_filters_applied_before_ranking_and_empty(kb):
    one=seed(kb);two=seed(kb,'two.txt','Nevo es una palabra artificial para verificar aislamiento de las fuentes.')
    assert kb.retrieve('melanoma',source_ids=[two])==[]
    assert kb.retrieve('melanoma',source_ids=[])==[]
    assert {h['source_id'] for h in kb.retrieve('nevo',source_ids=[two])}=={two}

def test_update_and_delete_persist_without_resurrection(kb):
    one=seed(kb)
    seed(kb,text='Nevo documento artificial de reemplazo para probar eliminacion de fragmentos viejos.')
    assert kb.retrieve('melanoma')==[] and kb.retrieve('nevo')
    assert len(kb.list_sources())==1
    kb.delete(one);assert kb.sync()['added']==[] and not kb.list_sources()

def test_invalid_upload_keeps_original_file_and_chunks(kb):
    seed(kb)
    old=(kb.documents/'one.txt').read_bytes()
    with pytest.raises(ValueError):kb.save_upload('one.txt',b'x',{})
    assert (kb.documents/'one.txt').read_bytes()==old and kb.retrieve('melanoma')

@pytest.mark.parametrize('name',['../secret.py','bad.exe','a?.pdf'])
def test_invalid_names(kb,name):
    with pytest.raises(ValueError):kb.save_upload(name,b'x',{})

def test_sync_reports_invalid_pdf_without_erasing_previous(kb):
    seed(kb)
    (kb.documents/'bad.pdf').write_bytes(b'bad')
    assert kb.sync()['errors'] and kb.retrieve('melanoma')

def test_missing_vectors_explicit_and_hybrid_isolated(kb):
    one=seed(kb);seed(kb,'other.txt','Nevo evidencia de prueba para la segunda fuente documental artificial.')
    with pytest.raises(ValueError,match='encoder'):kb.retrieve('melanoma',method='semantic')
    encoder=TestEncoder();kb.build_vectors(encoder)
    hits=kb.retrieve('melanoma',source_ids=[one],method='hybrid',encoder=encoder)
    assert hits and {h['source_id'] for h in hits}=={one}
    seed(kb,text='Melanoma documento artificial modificado que invalida sus embeddings anteriores.')
    with pytest.raises(ValueError,match='embeddings'):kb.retrieve('melanoma',method='semantic',encoder=encoder)

def test_visual_input_not_in_documental_payload(kb,monkeypatch):
    ident=seed(kb);captured={}
    def fake(cfg,sys,payload):
        captured.update(payload);hit=payload['evidence'][0]
        claim={'text':'Parafrasis artificial','citations':['S1'],'support':[{'id':'S1','quote':hit['text'][:30]}]}
        return json.dumps({'scientific':[claim],'relation':[]})
    monkeypatch.setattr('dermatoai.service.complete',fake)
    response=respond('melanoma',LLMConfig('Ollama local','test'),kb,[ident])
    assert response['generated'] and response['claims']['scientific'][0]['support']
    assert 'image' not in captured and captured['model_result'] is None

def test_no_evidence_no_network_and_no_forced_class_query(kb,monkeypatch):
    ident=seed(kb)
    monkeypatch.setattr('dermatoai.service.complete',lambda *a,**kw:pytest.fail('No debe invocar LLM'))
    out=respond('astronautas',LLMConfig('OpenAI API','test'),kb,[ident])
    assert not out['hits'] and not out['generated']
    assert 'Melanoma' not in query_for('queratosis',prediction())
    assert 'Melanoma' in query_for('explica el resultado',prediction())

def test_invalid_generated_citation_falls_back_to_extract(kb,monkeypatch):
    ident=seed(kb)
    monkeypatch.setattr('dermatoai.service.complete',lambda *a:json.dumps({'scientific':[{'text':'bad','citations':['S99'],'support':[]}],'relation':[]}))
    out=respond('melanoma',LLMConfig('Ollama local','test'),kb,[ident])
    assert not out['generated'] and out['warning'] and 'Evidencia documental' in out['markdown']

def test_overview_reports_actual_coverage(kb):
    seed(kb,text=('Palabra evidencia artificial para pruebas de muchos fragmentos. '*600))
    out=respond('resumen',LLMConfig(),kb,None,overview=True)
    assert out['coverage']['used_fragments']==12
    assert out['coverage']['selected_fragments']>24 and 'selección de pasajes aptos' in out['markdown']

def test_history_resolves_followup_without_becoming_evidence():
    q=query_for('Explicalo con mas detalle',history=[{'question':'melanoma','markdown':'unrelated'}])
    assert 'melanoma' in q and 'unrelated' not in q

def test_vision_requires_explicit_opt_in_before_network(monkeypatch):
    monkeypatch.setattr('dermatoai.vision.complete',lambda *a,**kw:pytest.fail('No red'))
    with pytest.raises(PermissionError):describe(Image.new('RGB',(10,10)),LLMConfig('OpenAI API','test'))

def test_visual_receives_actual_jpeg_not_prediction(monkeypatch):
    captured={}
    def fake(cfg,system,payload,**kw):
        captured.update(payload=payload,**kw)
        return json.dumps({'observations':['Color aparente de prueba.'],'limitations':['No validado.']})
    monkeypatch.setattr('dermatoai.vision.complete',fake)
    out=describe(Image.new('RGB',(2000,1000)),LLMConfig('Ollama local','vision'),consent=True)
    import base64
    with Image.open(io.BytesIO(base64.b64decode(captured['image_b64']))) as img:assert img.size==(1024,512)
    assert 'model_result' not in captured['payload'] and out['observations']

def test_openai_image_transport_and_text_consent(monkeypatch):
    capture={}
    def fake(url,payload=None,headers=None,timeout=None):
        capture.update(payload)
        return {'output':[{'type':'message','content':[{'type':'output_text','text':'{}'}]}]}
    monkeypatch.setattr('dermatoai.llm.request_json',fake)
    complete(LLMConfig('OpenAI API','vision',api_key='not-real',consent=True),'rules',{},image_b64='TEST')
    assert capture['input'][0]['content'][1]['type']=='input_image'
    with pytest.raises(PermissionError):complete(LLMConfig('OpenAI API','vision'),'rules',{},image_b64='TEST')

def test_oauth_uses_image_payload_without_assuming_model_support():
    class Session:
        def complete(self,model,system,payload):
            assert payload[0]['content'][1]['type']=='input_image';return '{}'
    assert complete(LLMConfig('ChatGPT OAuth','vision',consent=True,oauth_session=Session()),'rules',{},image_b64='TEST')=='{}'

def test_evaluation_metrics_and_unknown_annotations(kb):
    seed(kb)
    cases=[{'id':'a','question':'melanoma','relevant':[{'filename':'one.txt','page':None}]},
           {'id':'b','question':'astronautas','relevant':[],'unanswerable':True}]
    report=evaluate_cases(kb,cases)
    assert report['recall']==1 and report['reciprocal_rank']==1 and report['correctly_empty']==1
    cases[0]['relevant'][0]['filename']='missing.pdf'
    with pytest.raises(ValueError):evaluate_cases(kb,cases)

def test_ui_can_query_without_checkpoint_or_image(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from dermatoai.settings import ROOT
    app_path=ROOT/'app_dermatoai.py'
    monkeypatch.setattr('dermatoai.settings.ROOT',tmp_path)
    kb=KnowledgeBase(tmp_path/'knowledge_dermatoai/rag.sqlite3',tmp_path/'knowledge_dermatoai/documents')
    seed(kb)
    app=AppTest.from_file(str(app_path),default_timeout=30).run()
    assert not app.exception
    next(t for t in app.text_area if t.label=='Escribe tu pregunta').input('melanoma')
    next(b for b in app.button if b.label=='Responder mi pregunta').click().run()
    assert not app.exception
    assert app.session_state['chat'][0]['hits']
    assert not app.session_state['chat'][0]['generated']

@pytest.mark.parametrize('provider',['Informe local (sin LLM)','Ollama local','OpenAI API','ChatGPT OAuth'])
def test_connection_status_does_not_reference_image_payload(provider,monkeypatch):
    from dermatoai.llm import connection_status
    monkeypatch.setattr('dermatoai.llm.request_json',lambda *args,**kw:{'models':[]})
    text=connection_status(LLMConfig(provider,'test',api_key='not-real'))
    assert isinstance(text,str) and text
