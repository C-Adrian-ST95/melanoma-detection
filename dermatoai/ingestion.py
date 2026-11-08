"""Solo texto documental; HTTPS público y dominios autorizados, sin ejecutar contenido."""
import hashlib
import http.client
import ipaddress
import io
import re
import socket
import ssl
import unicodedata
from urllib.parse import urlsplit, urljoin
from .settings import ALLOWED_DOMAINS

def clean(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", "".join(c for c in text if c.isprintable() or c.isspace())).strip()

def chunks(text, size=650, overlap=120):
    if not 0 <= overlap < size:
        raise ValueError("Solapamiento debe ser menor que el tamaño.")
    text = clean(text)
    for start in range(0, len(text), size - overlap):
        part = text[start:start + size]
        if part:
            yield part
        if start + size >= len(text):
            break

def pdf_pages(data):
    from pypdf import PdfReader
    if len(data) > 25 * 1024 * 1024 or not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("Use un PDF válido de hasta 25 MiB.")
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("El PDF requiere contraseña; use una copia autorizada sin contraseña.")
    if len(reader.pages) > 300:
        raise ValueError("Máximo 300 páginas por documento.")
    pages = [(i + 1, clean(page.extract_text() or "")) for i, page in enumerate(reader.pages)]
    if sum(len(t) for _, t in pages) > 2_000_000:
        raise ValueError("Demasiado texto en un documento.")
    pages = [(page, text) for page, text in pages if len(text) >= 30]
    if not pages:
        raise ValueError("No hay texto extraíble. Un PDF escaneado requiere OCR previo.")
    return pages

def validated_url(url, resolver=socket.getaddrinfo):
    u = urlsplit(url)
    host = (u.hostname or "").lower()
    if (u.scheme != "https" or u.username or u.password or u.port not in (None, 443)
            or not any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)):
        raise ValueError("Solo HTTPS de dominios médicos autorizados; sin credenciales ni puertos alternativos.")
    addresses = list(dict.fromkeys(info[4][0] for info in resolver(host, 443, type=socket.SOCK_STREAM)))
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError("Dirección privada, local o reservada bloqueada.")
    return u, addresses

class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address):
        super().__init__(host, timeout=20, context=ssl.create_default_context())
        self.address = address
    def connect(self):
        sock = socket.create_connection((self.address, 443), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise

def fetch_public(url):
    for _ in range(4):
        u, ips = validated_url(url)
        conn = PinnedHTTPS(u.hostname, ips[0])
        try:
            path = u.path or "/"
            if u.query:
                path += "?" + u.query
            conn.request("GET", path, headers={"User-Agent": "DermatoAI-Academic/1.0", "Accept-Encoding": "identity"})
            response = conn.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                url = urljoin(url, response.getheader("Location", ""))
                continue  # cada destino se valida de nuevo; IP fijada a la conexión TLS
            if response.status != 200:
                raise ValueError(f"La organización respondió HTTP {response.status}.")
            content_type = response.getheader("Content-Type", "").lower()
            if not any(t in content_type for t in ("text/html", "application/pdf")):
                raise ValueError("Solo HTML o PDF público.")
            body = response.read(25 * 1024 * 1024 + 1)
            if len(body) > 25 * 1024 * 1024:
                raise ValueError("Respuesta demasiado grande.")
            return body, content_type, url
        finally:
            conn.close()
    raise ValueError("Demasiadas redirecciones.")

def web_pages(url):
    from bs4 import BeautifulSoup
    body, content_type, final_url = fetch_public(url)
    if "application/pdf" in content_type:
        return urlsplit(final_url).path.rsplit("/", 1)[-1], pdf_pages(body), final_url, hashlib.sha256(body).hexdigest()
    soup = BeautifulSoup(body, "html.parser")
    title = clean(soup.title.get_text()) if soup.title else urlsplit(final_url).hostname
    main = soup.find("main") or soup.find("article") or soup
    for element in main.find_all(["script", "style", "nav", "header", "footer", "form", "iframe", "noscript"]):
        element.decompose()
    text = clean(main.get_text(" "))
    if len(text) < 100 or len(text) > 2_000_000:
        raise ValueError("Texto web vacío o demasiado grande.")
    return title, [(None, text)], final_url, hashlib.sha256(body).hexdigest()
