import json
import socket
import numpy as np
import pytest
from dermatoai.rag import Index
from dermatoai.agent import answer, validate_claims, escape_text
from dermatoai.llm import LLMConfig, complete
from dermatoai.ingestion import validated_url, chunks, pdf_pages

class TestEncoder:
    """Vectorizador controlado SOLO para pruebas; no es el encoder de producción."""
    model_id = "test-controlled-vocabulary"
    def encode(self, texts):
        vocab = ["melanoma", "nevo", "queratosis", "evidencia"]
        return np.array([[t.lower().count(w) + 0.001 for w in vocab] for t in texts], dtype=np.float32)

@pytest.fixture
def index(tmp_path):
    return Index(tmp_path/"index.sqlite3", TestEncoder())

@pytest.fixture
def populated(index):
    pdf = index.add("pdf", "Guía local", "guia.pdf", [(5,"Melanoma evidencia documental de prueba, sin valor clínico.")], "abc")
    web = index.add("organization", "Organización", "https://www.aad.org/public", [(None,"Melanoma evidencia de la organización para prueba controlada.")], "def")
    return index, pdf, web

def prediction():
    return dict(top_class="MEL",top_score=0.4,scores={"ACK":0.1,"BCC":0.2,"MEL":0.4,"NEV":0.1,"SCC":0.1,"SEK":0.1},
                architecture="fixture",checkpoint="fixture.pt",epoch=0,sha256="TEST_ONLY")

@pytest.mark.parametrize("kind", ["pdf","organization"])
def test_filter_before_ranking(populated, kind):
    index, _, _ = populated
    hits=index.retrieve("melanoma", [kind], k=5)
    assert hits and all(h["kind"]==kind for h in hits)

def test_source_scope_and_empty_selection(populated):
    index,pdf,web=populated
    assert {h["source_id"] for h in index.retrieve("melanoma",["pdf","organization"],[pdf])}=={pdf}
    assert index.retrieve("melanoma",["pdf","organization"],[])==[]
    assert index.retrieve("melanoma",[])==[]
    assert index.retrieve("melanoma",["pdf"],[web])==[]

def test_delete_cascades(populated):
    index,pdf,web=populated;index.delete(pdf)
    assert index.retrieve("melanoma",["pdf"])==[]
    assert len(index.list_sources())==1

def test_duplicate_ingestion_replaces(populated):
    index,pdf,_=populated
    index.add("pdf","Guía local","guia.pdf",[(5,"Melanoma evidencia documental de prueba, sin valor clínico.")],"abc")
    assert len(index.list_sources())==2
    assert len(index.retrieve("melanoma",["pdf"]))==1

def test_persistence_and_model_guard(populated):
    index,pdf,_=populated
    assert Index(index.path,TestEncoder()).retrieve("melanoma",["pdf"])[0]["page"]==5
    other=TestEncoder();other.model_id="different"
    with pytest.raises(ValueError,match="embeddings cambió"):
        Index(index.path,other)

def test_no_evidence_and_no_rag(populated, monkeypatch):
    index,_,_=populated
    monkeypatch.setattr("dermatoai.agent.complete",lambda *args:pytest.fail("No debe consultar LLM sin evidencia"))
    cfg=LLMConfig(provider="OpenAI API",model="test")
    r=answer(prediction(),"pregunta",cfg,index,())
    assert not r["hits"] and "Sin RAG" in r["markdown"]
    r=answer(prediction(),"pregunta",cfg,index,("pdf",),[],min_score=1)
    assert "No se encontró evidencia" in r["markdown"]

def test_citation_validation_rejects_invented(populated):
    index,_,_=populated;hits=index.retrieve("melanoma",["pdf"])
    raw={"scientific":[{"text":"Ejemplo","citations":["S9"],"support":[]}],"relation":[]}
    with pytest.raises(ValueError, match="cita inexistente"):
        validate_claims(json.dumps(raw),hits)

def test_citation_validation_support(populated):
    index,_,_=populated;hits=index.retrieve("melanoma",["pdf"])
    claim={"text":"Texto de prueba","citations":["S1"],"support":[{"id":"S1","quote":hits[0]["text"][:30]}]}
    raw=json.dumps({"scientific":[claim],"relation":[]})
    assert validate_claims(raw,hits)["scientific"]
    claim["support"][0]["quote"]="Texto que nunca estuvo en la fuente"
    with pytest.raises(ValueError,match="no coincide"):
        validate_claims(json.dumps({"scientific":[claim],"relation":[]}),hits)

def test_llm_receives_only_current_sources(populated,monkeypatch):
    index,pdf,web=populated; captured={}
    def fake(config,system,payload):
        captured.update(payload)
        hit=payload["evidence"][0]
        return json.dumps({"scientific":[{"text":"Paráfrasis de prueba","citations":[hit["citation_id"]],
                           "support":[{"id":hit["citation_id"],"quote":hit["text"][:30]}]}],"relation":[]})
    monkeypatch.setattr("dermatoai.agent.complete",fake)
    out=answer(prediction(),"Consulta",LLMConfig("Ollama local","fixture"),index,("pdf",),[pdf])
    assert out["generated"] and "Paráfrasis" in out["markdown"]
    assert all(h["source_id"]==pdf for h in captured["evidence"])
    assert "image" not in captured
    assert "support" not in out["markdown"]

def test_consent_before_external_request(monkeypatch):
    monkeypatch.setattr("dermatoai.llm.request_json",lambda *a,**kw:pytest.fail("No debe haber red"))
    with pytest.raises(PermissionError):
        complete(LLMConfig("OpenAI API","test",api_key="secret",consent=False),"system",{})
    assert "secret" not in repr(LLMConfig(api_key="secret"))

def test_document_markup_escaped():
    out=escape_text("![x](https://example.com) <script> [a]")
    assert "\\!" in out and "\\<" in out and "\\[" in out

@pytest.mark.parametrize("url",["http://www.aad.org/x","https://evil-aad.org/x","https://www.aad.org.evil.com/x",
                               "https://user:pass@www.aad.org/x","https://www.aad.org:8443/x","https://127.0.0.1/x"])
def test_insecure_urls(url):
    with pytest.raises(ValueError):
        validated_url(url,resolver=lambda *a,**k:[])

def test_private_dns_and_public_host():
    resolver=lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,"",("127.0.0.1",443))]
    with pytest.raises(ValueError):
        validated_url("https://www.aad.org/x",resolver)
    resolver=lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,"",("8.8.8.8",443))]
    u,addresses=validated_url("https://www.aad.org/x",resolver)
    assert u.hostname=="www.aad.org" and addresses==["8.8.8.8"]

def test_chunk_overlap():
    parts=list(chunks("a"*1000,size=300,overlap=50))
    assert len(parts)==4 and all(len(p)<=300 for p in parts)
    assert "".join(p if i==0 else p[50:] for i,p in enumerate(parts))=="a"*1000
    with pytest.raises(ValueError):
        list(chunks("abc",size=10,overlap=10))

def test_pdf_real_pages(tmp_path):
    # PDF mínimo real producido por reportlab, solo en las pruebas de desarrollo.
    reportlab=pytest.importorskip("reportlab.pdfgen.canvas")
    file=tmp_path/"test.pdf"
    c=reportlab.Canvas(str(file));c.drawString(30,700,"Melanoma evidence: this is a synthetic test document.");c.showPage()
    c.drawString(30,700,"Second page evidence for index citation validation.");c.save()
    pages=pdf_pages(file.read_bytes())
    assert [p for p,t in pages]==[1,2] and "Melanoma" in pages[0][1]

def test_openai_text_transport(monkeypatch):
    captured={}
    def fake(url,payload=None,headers=None,timeout=90):
        captured.update(url=url,payload=payload,headers=headers)
        return {"output":[{"type":"message","content":[{"type":"output_text","text":'{"scientific":[],"relation":[]}'}]}]}
    monkeypatch.setattr("dermatoai.llm.request_json",fake)
    raw=complete(LLMConfig("OpenAI API","test-model",api_key="not-real",consent=True),"rules",{"evidence":[]})
    assert json.loads(raw)["scientific"]==[]
    assert captured["url"]=="https://api.openai.com/v1/responses"
    assert captured["payload"]["store"] is False
    assert "image" not in json.dumps(captured["payload"])
    assert captured["headers"]["Authorization"]=="Bearer not-real"

def test_ollama_transport(monkeypatch):
    captured={}
    def fake(url,payload=None,**kw):
        captured.update(url=url,payload=payload)
        return {"message":{"content":'{"scientific":[],"relation":[]}'}}
    monkeypatch.setattr("dermatoai.llm.request_json",fake)
    assert complete(LLMConfig("Ollama local","test-model"),"rules",{})
    assert captured["url"]=="http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["stream"] is False
    with pytest.raises(ValueError,match="cloud"):
        complete(LLMConfig("Ollama local","test-cloud"),"rules",{})

def test_pdf_password_and_empty_password(tmp_path):
    canvas=pytest.importorskip("reportlab.pdfgen.canvas")
    from pypdf import PdfReader, PdfWriter
    plain=tmp_path/"plain.pdf";c=canvas.Canvas(str(plain))
    c.drawString(30,700,"Synthetic text for a protected PDF extraction test.");c.save()
    for password in ["secret",""]:
        writer=PdfWriter();writer.append(PdfReader(plain));writer.encrypt(password)
        file=tmp_path/("protected.pdf" if password else "public_encrypted.pdf")
        with file.open("wb") as stream:writer.write(stream)
        if password:
            with pytest.raises(ValueError,match="contraseña"):
                pdf_pages(file.read_bytes())
        else:
            assert "Synthetic" in pdf_pages(file.read_bytes())[0][1]
