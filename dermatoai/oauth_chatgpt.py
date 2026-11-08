"""OAuth oficial para una instalación LOCAL de DermatoAI; no importa credenciales ajenas."""
from __future__ import annotations
import base64
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
import jwt
import requests
from filelock import FileLock, Timeout as LockTimeout

ISSUER = "https://auth.openai.com"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
DISCOVERY = ISSUER + "/.well-known/openid-configuration"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
REQUIRED = {"resource.invoke", "chatgpt.tokens.use.direct"}

class OAuthError(RuntimeError):
    """Mensaje público sin URLs de callback, códigos ni tokens."""

def default_directory():
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "DermatoAI/auth"
    return Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "DermatoAI/auth"

def _b64(value):
    return base64.urlsafe_b64encode(value).decode().rstrip("=")

def _atomic_json(path, data):
    descriptor, name = tempfile.mkstemp(prefix=".oauth-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            os.chmod(name, 0o600)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

def _origin_auth(url):
    parsed = urlsplit(url)
    return (parsed.scheme == "https" and parsed.hostname == "auth.openai.com"
            and parsed.port in (None, 443) and not parsed.username and not parsed.password)

@dataclass(repr=False)
class LoginAttempt:
    client_id: str
    previous_subject: str | None
    state: str = field(repr=False)
    nonce: str = field(repr=False)
    verifier: str = field(repr=False)
    deadline: float = 0
    redirect_uri: str = ""
    url: str = field(default="", repr=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    code: str = field(default="", repr=False)
    returned_client: str = ""
    error: str = ""
    consumed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    def cancel(self):
        self.cancelled.set()
    def __repr__(self):
        return f"LoginAttempt(ready={self.done.is_set()}, consumed={self.consumed})"

class OAuthBroker:
    def __init__(self, directory=None, http=None):
        self.directory = Path(directory) if directory else default_directory()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            os.chmod(self.directory, 0o700)
        self.file = self.directory / "accounts.json"
        self.file_lock = FileLock(str(self.directory / "accounts.lock"), timeout=15)
        self.http = http or requests.Session()
        self.metadata = None
        self.keys = None
        with self.locked():
            state = self._read()
            if not state.get("host_id"):
                state["host_id"] = "urn:uuid:" + str(uuid.uuid4())
                self._write(state)
            self.host_id = state["host_id"]

    @contextmanager
    def locked(self):
        try:
            with self.file_lock:
                yield
        except LockTimeout:
            raise OAuthError("Otra operación está actualizando la sesión. Espere y vuelva a intentar.") from None

    def _read(self):
        if not self.file.exists():
            return {"accounts": {}}
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            if not isinstance(data.get("accounts"), dict):
                raise ValueError()
            return data
        except (OSError, ValueError):
            raise OAuthError("No se pudo leer la configuración OAuth local.") from None

    def _write(self, data):
        _atomic_json(self.file, data)

    def accounts(self):
        with self.locked():
            data = self._read()
        # Nunca retorna tokens a la interfaz.
        return [dict(client_id=key, email=value.get("email", "Cuenta ChatGPT"),
                     label=f"{value.get('email', 'Cuenta ChatGPT')} · {key[-8:]}",
                     connected=bool(value.get("access_token")),
                     plan_enabled=REQUIRED <= set(value.get("scopes", [])))
                for key, value in data["accounts"].items()]

    def _record(self, client_id):
        with self.locked():
            record = self._read()["accounts"].get(client_id)
        if not record:
            raise OAuthError("Seleccione una cuenta registrada o conecte una nueva.")
        return record

    def _request(self, method, url, **kwargs):
        try:
            response = self.http.request(method, url, timeout=(8, 25), allow_redirects=False, **kwargs)
        except requests.RequestException:
            raise OAuthError("No se pudo contactar OpenAI. Revise la conexión y vuelva a intentar.") from None
        if response.status_code != 200:
            code, param = "", ""
            try:
                error = response.json().get("error", "")
                code = error.get("code", "") if isinstance(error, dict) else error
                param = error.get("param", "") if isinstance(error, dict) else ""
            except (ValueError, AttributeError):
                pass
            code = code if isinstance(code, str) else ""
            allowed = {"access_denied", "invalid_grant", "invalid_scope", "invalid_client",
                       "subscription_sharing_usage_limit_exceeded", "subscription_sharing_usage_unavailable",
                       "invalid_request_error", "unsupported_parameter", "unsupported_value", "invalid_value",
                       "model_not_found", "insufficient_quota", "rate_limit_exceeded"}
            # No mostrar el mensaje/cuerpo del servidor: puede repetir texto privado de la consulta.
            details = [code] if isinstance(code, str) and code in allowed else []
            if isinstance(param, str) and param in {"input", "model", "instructions", "max_output_tokens",
                                                   "stream", "store", "temperature", "top_p", "text.format"}:
                details.append("campo=" + param)
            label = " (" + "; ".join(details) + ")" if details else ""
            status = response.status_code
            response.close()
            if status == 401 or code in {"invalid_grant", "invalid_client"}:
                reason = "La sesión no es válida; vuelva a conectar la cuenta."
            elif status == 403 or code in {"access_denied", "invalid_scope"}:
                reason = "Revise los permisos de la cuenta y el acceso al modelo."
            elif status == 429 or code == "subscription_sharing_usage_limit_exceeded":
                reason = "Se alcanzó un límite de uso; revise la cuota de la cuenta."
            elif status == 400:
                reason = "Solicitud incompatible; revise el campo indicado y el modelo elegido."
            elif status >= 500:
                reason = "Fallo temporal del proveedor; vuelva a intentar."
            else:
                reason = "Revise el acceso y la disponibilidad del modelo."
            raise OAuthError(f"OpenAI respondió HTTP {status}{label}. {reason}")
        return response

    def _json(self, method, url, **kwargs):
        response = self._request(method, url, **kwargs)
        try:
            if len(response.content) > 2_000_000:
                raise OAuthError("Respuesta de autorización demasiado grande.")
            data = response.json()
            if not isinstance(data, dict):
                raise OAuthError("Formato de autorización no reconocido.")
            return data
        except ValueError:
            raise OAuthError("OpenAI no devolvió JSON válido.") from None
        finally:
            response.close()

    def discovery(self):
        if self.metadata is None:
            data = self._json("GET", DISCOVERY)
            if (data.get("issuer") != ISSUER or data.get("authorization_endpoint") != AUTHORIZE
                    or data.get("token_endpoint") != TOKEN or not _origin_auth(data.get("jwks_uri", ""))
                    or not _origin_auth(data.get("revocation_endpoint", ""))):
                raise OAuthError("La configuración OIDC no coincide con los endpoints oficiales esperados.")
            self.metadata = data
        return self.metadata

    def _validate_id(self, token, client_id, nonce=None):
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
                raise OAuthError("Algoritmo o identificador de firma del ID token inválido.")
            metadata = self.discovery()
            if self.keys is None or not any(key.get("kid") == header["kid"] for key in self.keys):
                self.keys = self._json("GET", metadata["jwks_uri"]).get("keys", [])
            key = next((key for key in self.keys if key.get("kid") == header["kid"]), None)
            if key is None:
                raise OAuthError("No se encontró la clave de firma del ID token.")
            claims = jwt.decode(token, jwt.PyJWK.from_dict(key).key, algorithms=["RS256"],
                                audience=client_id, issuer=ISSUER,
                                options={"require": ["exp", "iat", "sub", "iss", "aud"]}, leeway=10)
            if not isinstance(claims["sub"], str) or not claims["sub"]:
                raise OAuthError("El ID token no contiene identidad válida.")
            if nonce is not None and not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
                raise OAuthError("El nonce del ID token no coincide con este inicio de sesión.")
            return claims
        except (jwt.PyJWTError, ValueError, KeyError, TypeError):
            raise OAuthError("El ID token no superó la validación de firma, identidad o vigencia.") from None

    def start_login(self, client_id=None, ttl=300):
        self.discovery()  # valida los endpoints antes de mostrar el enlace
        previous = self._record(client_id) if client_id else None
        attempt = LoginAttempt(client_id=client_id or "dynamic_agent_client",
                               previous_subject=previous.get("subject") if previous else None,
                               state=secrets.token_urlsafe(32), nonce=secrets.token_urlsafe(32),
                               verifier=secrets.token_urlsafe(64), deadline=time.time() + ttl)
        class Listener(HTTPServer):
            def get_request(self):
                conn, address = super().get_request()
                conn.settimeout(3)
                return conn, address
        class Callback(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # nunca registrar el query OAuth
            def do_GET(self):
                parsed = urlsplit(self.path)
                status, message = 400, "Solicitud de autorización inválida."
                if parsed.path == "/auth/callback" and len(self.path) <= 16384:
                    query = parse_qs(parsed.query, keep_blank_values=True)
                    if all(len(values) == 1 for values in query.values()):
                        supplied = query.get("state", [""])[0]
                        with attempt.lock:
                            if time.time() >= attempt.deadline or attempt.cancelled.is_set() or attempt.consumed:
                                status, message = 410, "La autorización venció o fue cancelada."
                            elif attempt.done.is_set():
                                status, message = 409, "Esta autorización ya fue recibida."
                            elif secrets.compare_digest(supplied.encode(), attempt.state.encode()):
                                error = query.get("error", [""])[0]
                                issued = query.get("client_id", [attempt.client_id if previous else ""])[0]
                                code = query.get("code", [""])[0]
                                if error:
                                    attempt.error = "Autorización no concedida. Puede volver a conectar y aceptar los permisos."
                                elif (not code or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", issued)
                                      or issued == "dynamic_agent_client" or (previous and issued != attempt.client_id)):
                                    attempt.error = "El retorno OAuth no contiene un registro válido para esta cuenta."
                                else:
                                    attempt.code, attempt.returned_client = code, issued
                                attempt.done.set()
                                status, message = 200, "Autorización recibida. Regrese a DermatoAI y pulse Finalizar conexión."
                body = message.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'none'")
                self.end_headers()
                self.wfile.write(body)
        server = Listener(("127.0.0.1", 0), Callback)
        server.timeout = .25
        attempt.redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        params = dict(client_id=attempt.client_id, ext_agent_host_id=self.host_id,
                      response_type="code", redirect_uri=attempt.redirect_uri, scope=SCOPES,
                      resource=RESOURCE, state=attempt.state, nonce=attempt.nonce,
                      code_challenge_method="S256",
                      code_challenge=_b64(hashlib.sha256(attempt.verifier.encode()).digest()))
        if previous:
            if previous.get("id_token"):
                params["id_token_hint"] = previous["id_token"]
            if previous.get("email"):
                params["login_hint"] = previous["email"]
        else:
            params["agent_name_hint"] = "DermatoAI"
        attempt.url = AUTHORIZE + "?" + urlencode(params)
        def listen():
            try:
                while not attempt.done.is_set() and not attempt.cancelled.is_set() and time.time() < attempt.deadline:
                    server.handle_request()
            finally:
                server.server_close()
        threading.Thread(target=listen, daemon=True, name="DermatoAI-OAuth-callback").start()
        return attempt

    def finish_login(self, attempt):
        with attempt.lock:
            if attempt.consumed or attempt.cancelled.is_set() or time.time() >= attempt.deadline:
                raise OAuthError("La autorización venció o ya fue utilizada. Inicie otra conexión.")
            if not attempt.done.is_set():
                raise OAuthError("Complete primero la autorización en el navegador y luego vuelva a finalizar.")
            attempt.consumed = True
            code, client_id = attempt.code, attempt.returned_client
            attempt.code = ""
        if attempt.error:
            raise OAuthError(attempt.error)
        token = self._json("POST", TOKEN, data=dict(grant_type="authorization_code", client_id=client_id,
                          code=code, code_verifier=attempt.verifier, redirect_uri=attempt.redirect_uri, resource=RESOURCE))
        claims = self._validate_id(token.get("id_token", ""), client_id, attempt.nonce)
        if attempt.previous_subject and claims["sub"] != attempt.previous_subject:
            raise OAuthError("La identidad recibida no corresponde a la cuenta seleccionada.")
        record = self._token_record(token)
        record.update(subject=claims["sub"], issuer=ISSUER, email=claims.get("email", "Cuenta ChatGPT"),
                      id_token=token["id_token"], ext_agent_host_id=self.host_id, client_id=client_id)
        with self.locked():
            data = self._read()
            existing = data["accounts"].get(client_id)
            if existing and existing["subject"] != claims["sub"]:
                raise OAuthError("Ese registro pertenece a otra identidad; no se reemplazaron sus credenciales.")
            data["accounts"][client_id] = record
            self._write(data)
        return client_id

    def _token_record(self, token, previous=None):
        scopes = token.get("scope")
        if scopes is None and previous:
            scopes = " ".join(previous.get("scopes", []))
        if not isinstance(scopes, str):
            raise OAuthError("No se recibió la lista de permisos concedidos.")
        if not isinstance(token.get("access_token"), str) or not token["access_token"]:
            raise OAuthError("No se recibió un access token válido.")
        if not isinstance(token.get("refresh_token"), str) or not token["refresh_token"]:
            raise OAuthError("No se recibió un refresh token para conservar esta conexión.")
        if str(token.get("token_type", "")).lower() != "bearer":
            raise OAuthError("Tipo de credencial no admitido.")
        try:
            expires = float(token["expires_in"])
            if not 0 < expires <= 86400:
                raise ValueError()
        except (KeyError, ValueError, TypeError):
            raise OAuthError("Vigencia de credencial inválida.") from None
        record = dict(previous or {})
        record.update(access_token=token["access_token"], refresh_token=token["refresh_token"],
                      scopes=scopes.split(), expires_at=time.time() + expires,
                      earliest_refresh_at=token.get("earliest_refresh_at"))
        return record

    def access_token(self, client_id):
        with self.locked():  # serializa refresh y escritura, incluso entre procesos
            state = self._read()
            record = state["accounts"].get(client_id)
            if not record or not record.get("access_token"):
                raise OAuthError("Conecte primero su cuenta con Continuar con ChatGPT.")
            if not REQUIRED <= set(record.get("scopes", [])):
                raise OAuthError("La cuenta inició sesión pero no autorizó el uso del plan. Reconecte y conceda los permisos.")
            now = time.time()
            if now < record.get("expires_at", 0) - 60:
                return record["access_token"]
            earliest = record.get("earliest_refresh_at")
            if isinstance(earliest, (float, int)) and earliest > now:
                if now < record.get("expires_at", 0):
                    return record["access_token"]
                raise OAuthError("OpenAI aún no permite renovar esta sesión. Vuelva a conectar.")
            token = self._json("POST", TOKEN, data=dict(grant_type="refresh_token", client_id=client_id,
                               refresh_token=record["refresh_token"], resource=RESOURCE))
            updated = self._token_record(token, record)
            if token.get("id_token"):
                claims = self._validate_id(token["id_token"], client_id)
                if claims["sub"] != record["subject"]:
                    raise OAuthError("La renovación devolvió otra identidad; no se guardó.")
                updated["id_token"] = token["id_token"]
            state["accounts"][client_id] = updated
            self._write(state)
            if not REQUIRED <= set(updated["scopes"]):
                raise OAuthError("La sesión renovada no tiene autorización para usar el plan.")
            return updated["access_token"]

    def models(self, client_id):
        token = self.access_token(client_id)
        data = self._json("GET", RESOURCE + "/models", headers={"Authorization": "Bearer " + token})
        models = data.get("models")
        if not isinstance(models, list):
            raise OAuthError("OpenAI no devolvió el catálogo de modelos de esta cuenta OAuth.")
        return [dict(slug=m["slug"], display_name=m.get("display_name", m["slug"]))
                for m in models if isinstance(m, dict) and m.get("visibility") == "list"
                and isinstance(m.get("slug"), str) and m["slug"]]

    def complete(self, client_id, model, system, content):
        # El uso del plan ChatGPT exige input como lista y rechaza max_output_tokens.
        # La lista multimodal ya preparada por llm.complete se conserva sin modificar.
        response_input = ([{"role": "user", "content": [{"type": "input_text", "text": content}]}]
                          if isinstance(content, str) else content)
        if not isinstance(response_input, list) or not response_input:
            raise OAuthError("La entrada del modelo debe contener al menos un mensaje.")
        token = self.access_token(client_id)
        response = self._request("POST", RESOURCE + "/responses", stream=True,
                    headers={"Authorization": "Bearer " + token},
                    json={"model": model, "instructions": system, "input": response_input,
                          "store": False, "stream": True})
        output, done, event_lines, size = [], False, [], 0
        response.encoding = "utf-8"
        try:
            for line in response.iter_lines(decode_unicode=True):
                size += len(line)
                if size > 2_000_000:
                    raise OAuthError("Respuesta del modelo demasiado grande.")
                if line.startswith("data:"):
                    event_lines.append(line[5:].lstrip())
                elif not line and event_lines:
                    payload = "\n".join(event_lines)
                    event_lines = []
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except ValueError:
                        raise OAuthError("Evento de respuesta inválido.") from None
                    kind = event.get("type")
                    if kind == "response.output_text.delta":
                        output.append(event.get("delta", ""))
                    elif kind == "response.completed":
                        done = True
                        break
                    elif kind in {"response.failed", "response.incomplete", "error"}:
                        code = (event.get("response", {}).get("error") or event.get("error") or {}).get("code", "")
                        message = {"subscription_sharing_usage_limit_exceeded": "Se alcanzó el límite de uso del plan.",
                                   "subscription_sharing_usage_unavailable": "El uso del plan no está disponible para esta solicitud."}.get(code, "La generación no se completó; revise el modelo y la autorización.")
                        raise OAuthError(message)
            if not done:
                raise OAuthError("La conexión terminó sin response.completed; no se presenta como respuesta completa.")
            if not "".join(output).strip():
                raise OAuthError("La generación terminó sin texto de salida.")
            return "".join(output).strip()
        except requests.RequestException:
            raise OAuthError("Se interrumpió la conexión durante la generación.") from None
        finally:
            response.close()

    def sign_out(self, client_id):
        confirmed = False
        with self.locked():
            state = self._read()
            record = state["accounts"].get(client_id)
            if not record:
                return True
            if record.get("refresh_token"):
                try:
                    endpoint = self.discovery()["revocation_endpoint"]
                    for attempt in range(3):
                        try:
                            response = self._request("POST", endpoint, data=dict(token=record["refresh_token"],
                                         token_type_hint="refresh_token", client_id=client_id))
                            response.close()
                            confirmed = True
                            break
                        except OAuthError:
                            if attempt < 2:
                                time.sleep(.3 * (attempt + 1))
                except OAuthError:
                    pass
            else:
                confirmed = True
            for key in ("access_token", "refresh_token", "id_token", "expires_at", "earliest_refresh_at"):
                record.pop(key, None)
            state["accounts"][client_id] = record
            self._write(state)
        return confirmed

    def session(self, client_id):
        return OAuthSession(self, client_id)

@dataclass(repr=False)
class OAuthSession:
    broker: OAuthBroker = field(repr=False)
    client_id: str
    def complete(self, model, system, content):
        return self.broker.complete(self.client_id, model, system, content)
    def __repr__(self):
        return "OAuthSession(<cuenta registrada>)"
