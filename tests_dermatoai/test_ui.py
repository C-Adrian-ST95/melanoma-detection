from streamlit.testing.v1 import AppTest
from dermatoai.settings import ROOT

def test_app_opens_without_checkpoints():
    app=AppTest.from_file(str(ROOT/"app_dermatoai.py"),default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value=="Análisis de lesiones cutáneas"
    assert any("Faltan los pesos" in w.value for w in app.warning)
    mode=next(s for s in app.checkbox if s.label=="Usar documentos")
    mode.uncheck().run()
    assert not app.exception

def test_oauth_ui_connects_through_callback_and_lists_models(tmp_path,monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import rsa
    import json,jwt
    from tests_dermatoai.test_oauth_chatgpt import Transport,callback
    from dermatoai.oauth_chatgpt import OAuthBroker
    import dermatoai.oauth_ui as controls
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    jwk=json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="test-key",alg="RS256",use="sig")
    http=Transport((key,jwk));broker=OAuthBroker(tmp_path/"auth",http=http)
    monkeypatch.setattr(controls,"local_broker",lambda:broker)
    app=AppTest.from_file(str(ROOT/"app_dermatoai.py"),default_timeout=30).run()
    next(s for s in app.selectbox if s.label=="Motor de explicación").select("ChatGPT OAuth").run()
    assert not app.exception
    next(b for b in app.button if b.label=="Preparar inicio de sesión").click().run()
    assert not app.exception
    attempt=app.session_state["oauth_attempt"];http.nonce=attempt.nonce
    assert callback(attempt)[0]==200
    next(b for b in app.button if b.label=="Finalizar conexión").click().run()
    assert not app.exception
    selector=next(s for s in app.selectbox if s.label=="Modelo disponible en esta cuenta")
    assert selector.value=="allowed-test"
    assert any("permiso de uso del plan" in s.value for s in app.success)
    assert not app.session_state.filtered_state.get("oauth_data_consent")
    assert "No activada en esta entrega" not in str(app)
