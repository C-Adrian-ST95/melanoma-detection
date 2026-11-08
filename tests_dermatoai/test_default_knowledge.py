import json
import pytest
from reportlab.pdfgen.canvas import Canvas
from dermatoai.default_knowledge import load_defaults
from dermatoai.rag import Index
from tests_dermatoai.test_rag_agent import TestEncoder, prediction
from dermatoai.agent import base_report

def pdf(path,text):
    c=Canvas(str(path));c.drawString(20,700,text);c.save()

def test_seed_once_and_reuse_persistent_index(tmp_path):
    folder=tmp_path/"pdfs";folder.mkdir()
    pdf(folder/"guide.pdf","Melanoma evidencia documental artificial para una prueba de software.")
    (folder/"catalog.json").write_text(json.dumps({"guide.pdf":{"organization":"Institucion de prueba","title":"Guia de prueba","url":"https://example.invalid/guide"}}))
    index=Index(tmp_path/"index.sqlite3",TestEncoder())
    assert load_defaults(index,folder)["added"]==["guide.pdf"]
    assert load_defaults(index,folder)["unchanged"]==["guide.pdf"]
    reopened=Index(index.path,TestEncoder())
    hits=reopened.retrieve("melanoma",["pdf"])
    assert hits and hits[0]["page"]==1
    assert "Institucion" in hits[0]["title"]
    assert len(reopened.list_sources())==1

def test_updated_pdf_replaces_its_older_chunks(tmp_path):
    folder=tmp_path/"pdfs";folder.mkdir()
    pdf(folder/"guide.pdf","Melanoma evidencia de la primera version artificial de prueba.")
    index=Index(tmp_path/"index.sqlite3",TestEncoder());load_defaults(index,folder)
    before=index.list_sources()[0]["id"]
    pdf(folder/"guide.pdf","Nevo y queratosis: segunda version artificial de prueba documental.")
    assert load_defaults(index,folder)["added"]==["guide.pdf"]
    assert len(index.list_sources())==1 and index.list_sources()[0]["id"]!=before

def test_failed_update_keeps_previous_version(tmp_path):
    folder=tmp_path/"pdfs";folder.mkdir()
    pdf(folder/"guide.pdf","Melanoma evidencia artificial para verificar conservacion del indice.")
    index=Index(tmp_path/"index.sqlite3",TestEncoder());load_defaults(index,folder)
    old=index.list_sources()[0]["id"]
    (folder/"guide.pdf").write_bytes(b"not a pdf")
    assert load_defaults(index,folder)["errors"]
    assert index.list_sources()[0]["id"]==old

def test_metadata_updates_and_empty_folder(tmp_path):
    folder=tmp_path/"pdfs";folder.mkdir();index=Index(tmp_path/"index.sqlite3",TestEncoder())
    assert load_defaults(index,folder)=={"added":[],"unchanged":[],"errors":[]}
    pdf(folder/"guide.pdf","Melanoma evidencia artificial para verificar los metadatos del documento.")
    load_defaults(index,folder)
    (folder/"catalog.json").write_text(json.dumps({"guide.pdf":{"organization":"Nueva institucion","url":"https://example.invalid/new"}}))
    load_defaults(index,folder)
    assert len(index.list_sources())==1
    assert "Nueva institucion" in index.list_sources()[0]["title"]

def test_report_matches_requested_structure():
    report=base_report(prediction())
    assert "Interpretación del modelo" in report and "MEL" in report
    assert "no demuestra" in report
    assert "\n\n" in report and "\\n" not in report
