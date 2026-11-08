"""Regresiones de generación y comparación con datos artificiales; sin credenciales reales."""
import json
import pytest
from dermatoai.knowledge import KnowledgeBase
from dermatoai.llm import LLMConfig, generation_status
from dermatoai.service import respond

@pytest.fixture
def kb(tmp_path):
    kb = KnowledgeBase(tmp_path / "rag.sqlite3", tmp_path / "documents")
    documents = {
        "comparison.md": "# Diagnóstico diferencial\n\nEl nevo y el melanoma son aquí dos palabras artificiales para probar la comparación documental. Esta frase es soporte de prueba, no información clínica.",
        "nev.md": "# Definición\n\nEl nevo es una categoría mencionada en este texto artificial para las pruebas de software.",
        "mel.md": "# Definición\n\nEl melanoma se describe únicamente como una palabra artificial para comprobar la recuperación de fuentes.",
        "therapy.md": "# Tratamiento\n\nEl imiquimod para la queratosis actínica es aquí un ejemplo artificial de pasaje ajeno a la comparación. No constituye orientación médica.",
    }
    for name, text in documents.items():
        # Simular un documento etiquetado con varias categorías: sigue comprobándose el texto del pasaje.
        kb.save_upload(name, text.encode(), {"title": name, "language": "es", "categories": ["NEV", "MEL", "ACK"]})
    return kb

def test_comparison_excludes_unrelated_treatment_and_prioritizes_both_subjects(kb):
    out = respond("¿Qué diferencias existen entre un nevo y un melanoma?", LLMConfig(), kb, None)
    assert out["hits"] and out["hits"][0]["filename"] == "comparison.md"
    assert all(h["filename"] != "therapy.md" for h in out["hits"])
    assert out["filters"]["comparison_topics"] == ["MEL", "NEV"]
    assert [h["citation_id"] for h in out["hits"]] == [f"S{i}" for i in range(1, len(out["hits"]) + 1)]

def test_comparison_preserves_source_selection(kb):
    selected = next(s["id"] for s in kb.list_sources() if s["filename"] == "therapy.md")
    out = respond("Compara un nevo y un melanoma", LLMConfig(), kb, [selected])
    assert not out["hits"]  # Nunca ampliar las fuentes para forzar una respuesta.

def test_treatment_question_can_still_retrieve_its_sources(kb):
    out = respond("imiquimod queratosis actínica tratamiento", LLMConfig(), kb, None)
    assert any(h["filename"] == "therapy.md" for h in out["hits"])

def test_generation_probe_requires_actual_model_response_and_uses_no_documents(monkeypatch):
    calls = []
    def fake(config, system, payload):
        calls.append((config, system, payload))
        return '{"ok": true}'
    monkeypatch.setattr("dermatoai.llm.complete", fake)
    assert "Generación verificada" in generation_status(LLMConfig("Ollama local", "test"))
    assert len(calls) == 1 and set(calls[0][2]) == {"task"}

@pytest.mark.parametrize("raw", ['{"ok": false}', '[]', 'sin JSON'])
def test_invalid_generation_probe_is_not_reported_as_success(raw, monkeypatch):
    monkeypatch.setattr("dermatoai.llm.complete", lambda *args: raw)
    with pytest.raises(ValueError):
        generation_status(LLMConfig("Ollama local", "test"))

def test_selected_llm_failure_has_actionable_fallback(kb, monkeypatch):
    def fail(*args):
        raise RuntimeError("OpenAI respondió HTTP 400 (campo=input). Solicitud incompatible.")
    monkeypatch.setattr("dermatoai.service.complete", fail)
    out = respond("nevo melanoma diferencias", LLMConfig("ChatGPT OAuth", "test"), kb, None)
    assert not out["generated"] and out["warning"]
    assert "Probar generación" in out["markdown"]
    assert "conecte y seleccione un LLM" not in out["markdown"]
    assert out["model"] == "test"

def test_probe_button_calls_model_from_streamlit(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from dermatoai.settings import ROOT
    app_path = ROOT / "app_dermatoai.py"
    monkeypatch.setattr("dermatoai.settings.ROOT", tmp_path)
    calls = []
    def fake(url, payload=None, **kwargs):
        calls.append(payload)
        return {"message": {"content": '{"ok": true}'}}
    monkeypatch.setattr("dermatoai.llm.request_json", fake)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    next(s for s in app.selectbox if s.label == "Motor de explicación").select("Ollama local").run()
    next(t for t in app.text_input if t.label == "Modelo instalado en Ollama").input("test-model").run()
    next(b for b in app.button if b.label == "Probar generación").click().run()
    assert not app.exception
    assert calls and any("Generación verificada" in s.value for s in app.success)
