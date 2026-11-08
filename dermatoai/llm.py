"""Transporte explícito de texto y, solo si se solicita, imagen anonimizada."""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlsplit

@dataclass
class LLMConfig:
    provider: str = "Informe local (sin LLM)"
    model: str = ""
    api_key: str = field(default="", repr=False)
    consent: bool = False
    oauth_session: object = field(default=None, repr=False)

def request_json(url, payload=None, headers=None, timeout=90):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    # No redirects: evita reenviar credenciales a destinos no previstos.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=timeout) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError("Respuesta del proveedor demasiado grande.")
        return json.loads(body)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"El proveedor respondió HTTP {exc.code}; revise acceso, modelo y cuota.") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("No se pudo conectar al proveedor dentro del plazo.") from None

def complete(config, system, payload, *, image_b64=None):
    if not config.model.strip():
        raise ValueError("Indique el nombre del modelo instalado o habilitado.")
    content = json.dumps(payload, ensure_ascii=False)
    user = {"role": "user", "content": content}
    if image_b64:
        user["images"] = [image_b64]
    if config.provider == "Ollama local":
        if "cloud" in config.model.lower():
            raise ValueError("El modo local no admite modelos marcados como cloud.")
        reply = request_json("http://127.0.0.1:11434/api/chat", {
            "model": config.model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system}, user],
            "options": {"temperature": 0, "num_predict": 1800}}, timeout=120)
        return reply["message"]["content"]
    if not config.consent:
        raise PermissionError("Autorice el envío de consulta, puntuaciones y fragmentos al proveedor externo.")
    response_input = content
    if image_b64:
        response_input = [{"role": "user", "content": [
            {"type": "input_text", "text": content},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + image_b64}]}]
    if config.provider == "ChatGPT OAuth":
        if config.oauth_session is None:
            raise PermissionError("Primero complete el inicio de sesión oficial con ChatGPT.")
        return config.oauth_session.complete(config.model, system, response_input)
    if config.provider != "OpenAI API":
        raise ValueError("Proveedor no implementado.")
    key = config.api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise PermissionError("Falta OPENAI_API_KEY o una clave en la barra lateral.")
    reply = request_json("https://api.openai.com/v1/responses", {
        "model": config.model, "instructions": system, "input": response_input,
        "store": False, "max_output_tokens": 3000,
        "text": {"format": {"type": "json_object"}}}, {"Authorization": "Bearer " + key}, timeout=120)
    parts = [item.get("text", "") for block in reply.get("output", []) if block.get("type") == "message"
             for item in block.get("content", []) if item.get("type") == "output_text"]
    if not parts:
        raise ValueError("El proveedor no devolvió texto; revise el modelo o el presupuesto de salida.")
    return "".join(parts)

def connection_status(config):
    if config.provider == "Informe local (sin LLM)":
        return "Disponible: informe determinista y fragmentos, sin generación LLM."
    if config.provider == "Ollama local":
        reply = request_json("http://127.0.0.1:11434/api/tags", timeout=5)
        models = [m.get("name", "") for m in reply.get("models", [])]
        return "Ollama conectado. Modelos: " + (", ".join(models) or "ninguno; ejecute ollama pull MODELO")
    if config.provider == "ChatGPT OAuth":
        if config.oauth_session is None:
            return "Prepare el inicio de sesión, pulse Continuar con ChatGPT y finalice la conexión."
        models = config.oauth_session.broker.models(config.oauth_session.client_id)
        return (f"Conexión OAuth verificada: {len(models)} modelos disponibles para esta cuenta. "
                "Use Probar generación para comprobar una respuesta del modelo.")
    key = config.api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return "Falta la clave de API."
    request_json("https://api.openai.com/v1/models", headers={"Authorization": "Bearer " + key}, timeout=10)
    return "API conectada. La disponibilidad del modelo se comprueba al generar."

def generation_status(config):
    """Prueba explícita de inferencia: no envía documentos, historial ni fotografías."""
    if config.provider == "Informe local (sin LLM)":
        raise ValueError("Seleccione un motor LLM para probar la generación.")
    raw = complete(config, 'Devuelve solo este objeto JSON, sin texto adicional: {"ok": true}.',
                   {"task": "Prueba de generación de DermatoAI. Responde con ok=true."})
    reply = json.loads(raw)
    if not isinstance(reply, dict) or reply.get("ok") is not True:
        raise ValueError("El modelo respondió, pero no devolvió el JSON esperado en la prueba.")
    return f"Generación verificada con {config.provider} · {config.model}. Ya puede consultar los documentos."
