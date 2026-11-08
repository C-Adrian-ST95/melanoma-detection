"""Sin cuenta ni credenciales reales: RSA/JWKS de prueba, callback HTTP y proveedores simulados."""
import http.client
import json
import os
import time
from urllib.parse import urlsplit, parse_qs, urlencode
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from dermatoai.oauth_chatgpt import OAuthBroker, OAuthError, ISSUER, AUTHORIZE, TOKEN, DISCOVERY, RESOURCE, SCOPES

@pytest.fixture(scope="module")
def signer():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="test-key", alg="RS256", use="sig")
    return key, jwk

class Response:
    def __init__(self, data=None, status=200, events=None):
        self.data, self.status_code, self.events = data, status, events
        self.content = json.dumps(data).encode() if data is not None else b""
        self.closed = False
    def json(self):
        return self.data
    def close(self):
        self.closed = True
    def iter_lines(self, **kwargs):
        for event in self.events or []:
            yield "data: " + json.dumps(event)
            yield ""

class Transport:
    def __init__(self, signer):
        self.key, self.jwk = signer
        self.calls = []
        self.nonce = ""
        self.subject = "synthetic-subject"
        self.scopes = SCOPES
        self.events = [{"type":"response.output_text.delta","delta":'{"scientific":[],"relation":[]}'},
                       {"type":"response.completed"}]
        self.failed_revoke = False
    def id_token(self, client="oaiapp_test", **changes):
        claims = dict(iss=ISSUER, aud=client, sub=self.subject, exp=int(time.time())+3600,
                      iat=int(time.time()), nonce=self.nonce, email="test@example.invalid")
        claims.update(changes)
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid":"test-key"})
    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if url == DISCOVERY:
            return Response(dict(issuer=ISSUER, authorization_endpoint=AUTHORIZE, token_endpoint=TOKEN,
                                  jwks_uri=ISSUER+"/.well-known/jwks.json",
                                  revocation_endpoint=ISSUER+"/api/accounts/oauth/revoke"))
        if url.endswith("/.well-known/jwks.json"):
            return Response({"keys":[self.jwk]})
        if url == TOKEN:
            data=kw["data"]
            if data["grant_type"]=="refresh_token":
                return Response(dict(access_token="refreshed-test-token",refresh_token="rotated-refresh-test",
                                     token_type="Bearer",expires_in=3600,scope=self.scopes))
            return Response(dict(access_token="test-access-token",refresh_token="test-refresh-token",
                                 id_token=self.id_token(data["client_id"]),token_type="Bearer",
                                 expires_in=3600,scope=self.scopes))
        if url == RESOURCE+"/models":
            return Response({"models":[{"slug":"allowed-test","display_name":"Test model","visibility":"list"},
                                       {"slug":"hidden-test","visibility":"hidden"}]})
        if url == RESOURCE+"/responses":
            body = kw["json"]
            if not isinstance(body.get("input"), list):
                return Response({"error": {"code": "invalid_value", "param": "input"}}, 400)
            if "max_output_tokens" in body:
                return Response({"error": {"code": "unsupported_parameter", "param": "max_output_tokens"}}, 400)
            return Response(events=self.events)
        if url.endswith("/revoke"):
            return Response({"error":"no-secret-details"},500) if self.failed_revoke else Response()
        raise AssertionError("Unexpected URL")

@pytest.fixture
def pair(tmp_path, signer):
    http=Transport(signer)
    return OAuthBroker(tmp_path/"auth",http=http),http

def callback(attempt, **values):
    url=urlsplit(attempt.redirect_uri)
    conn=http.client.HTTPConnection(url.hostname,url.port,timeout=3)
    query={"state":attempt.state,"code":"synthetic-code","client_id":"oaiapp_test",**values}
    conn.request("GET",url.path+"?"+urlencode(query))
    response=conn.getresponse(); body=response.read().decode();conn.close()
    return response.status,body

def connected(pair):
    broker,http=pair
    attempt=broker.start_login();http.nonce=attempt.nonce
    assert callback(attempt)[0]==200
    ident=broker.finish_login(attempt)
    return broker,http,ident

def test_dynamic_registration_pkce_and_host(pair):
    broker,http=pair
    attempt=broker.start_login()
    try:
        q=parse_qs(urlsplit(attempt.url).query)
        assert q["client_id"]==["dynamic_agent_client"]
        assert q["agent_name_hint"]==["DermatoAI"]
        assert q["code_challenge_method"]==["S256"]
        assert q["resource"]==[RESOURCE]
        assert set(q["scope"][0].split())==set(SCOPES.split())
        assert q["ext_agent_host_id"]==[broker.host_id]
        assert urlsplit(attempt.redirect_uri).hostname=="127.0.0.1"
        assert urlsplit(attempt.redirect_uri).path=="/auth/callback"
        assert OAuthBroker(broker.directory,http=http).host_id==broker.host_id
        assert attempt.verifier not in attempt.url
    finally:attempt.cancel()

def test_successful_callback_store_and_replay(pair):
    broker,http,ident=connected(pair)
    assert ident=="oaiapp_test"
    assert broker.accounts()[0]["connected"]
    assert broker.accounts()[0]["plan_enabled"]
    assert "test-access-token" not in repr(broker.accounts())
    assert broker.access_token(ident)=="test-access-token"
    post=next(kw["data"] for method,url,kw in http.calls if url==TOKEN)
    assert post["client_id"]!="dynamic_agent_client" and "client_secret" not in post
    assert post["redirect_uri"].startswith("http://127.0.0.1:")
    if os.name!="nt":
        assert broker.file.stat().st_mode & 0o777 == 0o600
        assert broker.directory.stat().st_mode & 0o777 == 0o700
    assert "test-access-token" not in repr(broker.session(ident))

def test_state_failure_does_not_consume_login(pair):
    broker,http=pair;attempt=broker.start_login();http.nonce=attempt.nonce
    try:
        assert callback(attempt,state="wrong-state")[0]==400
        assert not attempt.done.is_set()
        assert callback(attempt)[0]==200
        assert broker.finish_login(attempt)=="oaiapp_test"
        with pytest.raises(OAuthError,match="utilizada"):
            broker.finish_login(attempt)
        assert len([1 for _,url,_ in http.calls if url==TOKEN])==1
    finally:attempt.cancel()

def test_denied_consent_no_exchange(pair):
    broker,http=pair;attempt=broker.start_login()
    assert callback(attempt,error="access_denied")[0]==200
    with pytest.raises(OAuthError,match="no concedida"):
        broker.finish_login(attempt)
    assert not any(url==TOKEN for _,url,_ in http.calls)
    assert broker.accounts()==[]

@pytest.mark.parametrize("change",[{"nonce":"wrong"},{"iss":"https://attacker.invalid"},{"aud":"wrong-client"},
                                  {"exp":1},{"sub":""}])
def test_bad_id_token_rejected(pair,change):
    broker,http=pair;http.nonce="expected"
    token=http.id_token(**change)
    with pytest.raises(OAuthError):
        broker._validate_id(token,"oaiapp_test","expected")

def test_invalid_signature(pair,signer):
    broker,http=pair
    foreign=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    token=jwt.encode({"sub":"test","iss":ISSUER,"aud":"oaiapp_test","iat":int(time.time()),
                      "exp":int(time.time())+600,"nonce":"expected"},foreign,algorithm="RS256",headers={"kid":"test-key"})
    with pytest.raises(OAuthError):
        broker._validate_id(token,"oaiapp_test","expected")

def test_returning_login_reuses_registration_and_rejects_other_identity(pair):
    broker,http,ident=connected(pair)
    attempt=broker.start_login(ident)
    params=parse_qs(urlsplit(attempt.url).query)
    assert params["client_id"]==[ident] and "agent_name_hint" not in params
    assert params["id_token_hint"] and params["ext_agent_host_id"]==[broker.host_id]
    http.nonce=attempt.nonce;http.subject="another-account"
    assert callback(attempt)[0]==200
    with pytest.raises(OAuthError,match="identidad"):
        broker.finish_login(attempt)
    assert broker._record(ident)["subject"]=="synthetic-subject"

def test_wrong_returning_client_is_rejected(pair):
    broker,http,ident=connected(pair);attempt=broker.start_login(ident)
    assert callback(attempt,client_id="oaiapp_other")[0]==200
    old_count=len([1 for _,url,_ in http.calls if url==TOKEN])
    with pytest.raises(OAuthError,match="registro válido"):
        broker.finish_login(attempt)
    assert len([1 for _,url,_ in http.calls if url==TOKEN])==old_count

def test_identity_only_permission_cannot_infer(pair):
    broker,http=pair;http.scopes="openid profile email offline_access"
    broker,http,ident=connected(pair)
    assert not broker.accounts()[0]["plan_enabled"]
    with pytest.raises(OAuthError,match="no autorizó"):
        broker.models(ident)
    assert not any(url==RESOURCE+"/models" for _,url,_ in http.calls)

def test_refresh_rotation_and_correct_client(pair):
    broker,http,ident=connected(pair)
    with broker.file_lock:
        data=broker._read();data["accounts"][ident]["expires_at"]=0;broker._write(data)
    assert broker.access_token(ident)=="refreshed-test-token"
    assert broker._record(ident)["refresh_token"]=="rotated-refresh-test"
    grant=[kw["data"] for _,url,kw in http.calls if url==TOKEN][-1]
    assert grant["client_id"]==ident and grant["resource"]==RESOURCE
    assert grant["grant_type"]=="refresh_token" and "scope" not in grant

def test_models_and_completed_stream(pair):
    broker,http,ident=connected(pair)
    assert broker.models(ident)==[{"slug":"allowed-test","display_name":"Test model"}]
    assert broker.session(ident).complete("allowed-test","rules","input")=='{"scientific":[],"relation":[]}'
    post=[kw for _,url,kw in http.calls if url==RESOURCE+"/responses"][-1]
    assert post["json"]["stream"] is True and post["json"]["store"] is False
    assert post["json"]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "input"}]}]
    assert set(post["json"]) == {"model", "instructions", "input", "store", "stream"}
    assert post["stream"] is True and post["allow_redirects"] is False
    assert "image" not in json.dumps(post["json"])

def test_multimodal_input_kept_as_array(pair):
    broker, http, ident = connected(pair)
    content = [{"role": "user", "content": [{"type": "input_text", "text": "test"},
               {"type": "input_image", "image_url": "data:image/jpeg;base64,TEST"}]}]
    broker.complete(ident, "allowed-test", "rules", content)
    post = [kw["json"] for _, url, kw in http.calls if url == RESOURCE + "/responses"][-1]
    assert post["input"] == content and "max_output_tokens" not in post

def test_rag_generates_and_checks_citations_through_oauth_transport(pair, tmp_path):
    from dermatoai.knowledge import KnowledgeBase
    from dermatoai.llm import LLMConfig
    from dermatoai.service import respond
    broker, http, ident = connected(pair)
    kb = KnowledgeBase(tmp_path / "rag.sqlite3", tmp_path / "documents")
    evidence = "El nevo y el melanoma se mencionan en esta frase artificial para verificar el flujo de software."
    kb.save_upload("prueba.txt", evidence.encode(), {"title": "Fuente artificial"})
    claim = {"text": "Explicación artificial respaldada por la fuente de prueba.", "citations": ["S1"],
             "support": [{"id": "S1", "quote": evidence[:65]}]}
    http.events = [{"type": "response.output_text.delta", "delta": json.dumps({"scientific": [claim], "relation": []})},
                   {"type": "response.completed"}]
    config = LLMConfig("ChatGPT OAuth", "allowed-test", consent=True, oauth_session=broker.session(ident))
    out = respond("¿Qué diferencias existen entre un nevo y un melanoma?", config, kb, None)
    assert out["generated"] and not out["warning"] and "[S1]" in out["markdown"]
    assert out["claims"]["scientific"][0]["support"][0]["quote"] == evidence[:65]
    post = next(kw["json"] for _, url, kw in http.calls if url == RESOURCE + "/responses")
    payload = json.loads(post["input"][0]["content"][0]["text"])
    assert payload["evidence"][0]["text"] == evidence
    assert "image" not in payload and "max_output_tokens" not in post

@pytest.mark.parametrize("status,code,param,expected", [
    (400, "unsupported_parameter", "max_output_tokens", "campo=max_output_tokens"),
    (400, "invalid_value", "input", "Solicitud incompatible"),
    (401, None, None, "vuelva a conectar"),
    (403, "access_denied", None, "permisos"),
    (429, "rate_limit_exceeded", None, "límite"),
    (503, None, None, "temporal"),
])
def test_http_errors_explain_status_without_leaking_payload(pair, status, code, param, expected):
    broker, http = pair
    response = Response({"error": {"code": code, "param": param,
                         "message": "PRIVATE_DOCUMENT test-access-token"}}, status)
    http.request = lambda *args, **kwargs: response
    with pytest.raises(OAuthError) as exc:
        broker._request("POST", RESOURCE + "/responses")
    message = str(exc.value)
    assert expected in message and f"HTTP {status}" in message
    assert "PRIVATE_DOCUMENT" not in message and "test-access-token" not in message
    if status == 400:
        assert "vuelva a conectar" not in message
    assert response.closed

def test_unknown_error_details_never_echoed(pair):
    broker, http = pair
    http.request = lambda *args, **kwargs: Response({"error": {"code": "PRIVATE_DOCUMENT", "param": "test-access-token"}}, 400)
    with pytest.raises(OAuthError) as exc:
        broker._request("POST", RESOURCE + "/responses")
    assert "PRIVATE_DOCUMENT" not in str(exc.value) and "test-access-token" not in str(exc.value)

@pytest.mark.parametrize("events", [
    [{"type":"response.output_text.delta","delta":"incomplete"}],
    [{"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded"}}}],
    [{"type":"response.incomplete"}],
])
def test_failed_or_incomplete_stream_never_success(pair,events):
    broker,http,ident=connected(pair);http.events=events
    with pytest.raises(OAuthError):
        broker.complete(ident,"test","rules","question")

def test_sign_out_preserves_registration_not_tokens(pair):
    broker,http,ident=connected(pair)
    assert broker.sign_out(ident)
    record=broker._record(ident)
    assert record["subject"] and record["client_id"]==ident
    assert not any(key in record for key in ("access_token","id_token","refresh_token"))
    assert not broker.accounts()[0]["connected"]

def test_uncertain_revocation_is_reported_and_local_tokens_cleared(pair,monkeypatch):
    broker,http,ident=connected(pair);http.failed_revoke=True
    monkeypatch.setattr("dermatoai.oauth_chatgpt.time.sleep",lambda _:None)
    assert not broker.sign_out(ident)
    assert not broker._record(ident).get("access_token")

def test_timeout_and_cancel(pair):
    broker,http=pair
    a=broker.start_login(ttl=0)
    with pytest.raises(OAuthError,match="venció"):
        broker.finish_login(a)
    b=broker.start_login();b.cancel()
    with pytest.raises(OAuthError,match="venció"):
        broker.finish_login(b)
