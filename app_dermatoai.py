"""DermatoAI v2: documentos persistentes, consulta libre y análisis separados."""
import hashlib
import json
from pathlib import Path
import streamlit as st
from dermatoai.settings import ROOT, CLASSES, NAMES, NOTICE, SCOPE
from dermatoai.classifier import Classifier, checkpoint_files, image_from_bytes
from dermatoai.knowledge import KnowledgeBase
from dermatoai.llm import LLMConfig, connection_status, generation_status
from dermatoai.service import respond, make_report

st.set_page_config(page_title="DermatoAI · Investigación", page_icon="🔬", layout="wide")
st.markdown('''<style>.block-container{max-width:1250px;padding-top:2rem}
[data-testid="stSidebar"]{background:#eef5f5}h1,h2,h3{color:#123d45}</style>''', unsafe_allow_html=True)
DOCS = ROOT / "knowledge_dermatoai" / "documents"
DATABASE = ROOT / "knowledge_dermatoai" / "rag.sqlite3"
for key, value in {"result": None, "report": None, "chat": [], "visual": None, "xai": None,
                   "uploader_version": 0, "sync_report": None}.items():
    st.session_state.setdefault(key, value)

@st.cache_resource(max_entries=1)
def load_classifier(path, size, modified, device):
    return Classifier(path, device)

@st.cache_resource(max_entries=1)
def load_encoder(path):
    from dermatoai.rag import LocalEncoder
    return LocalEncoder(path)

kb = KnowledgeBase(DATABASE, DOCS)
if st.session_state.sync_report is None:
    try:
        st.session_state.sync_report = kb.sync()
    except Exception as exc:
        st.session_state.sync_report = {"errors": [{"error": str(exc)}]}

st.sidebar.title("DermatoAI")
st.sidebar.caption("Documentos · clasificación · explicación")
provider = st.sidebar.selectbox("Motor de explicación", ["Informe local (sin LLM)", "Ollama local", "OpenAI API", "ChatGPT OAuth"])
model_name, api_key, consent, oauth_session, oauth_context = "", "", False, None, ""
if provider == "Ollama local":
    model_name = st.sidebar.text_input("Modelo instalado en Ollama", value="")
    st.sidebar.caption("Escriba el nombre exacto de un modelo local instalado. Para fotografías, debe admitir imágenes.")
elif provider == "OpenAI API":
    model_name = st.sidebar.text_input("Modelo habilitado en su API", value="")
    api_key = st.sidebar.text_input("API key (solo sesión)", type="password")
    consent = st.sidebar.checkbox("Autorizo enviar consulta y fragmentos a OpenAI", value=False)
    st.sidebar.caption("La API y la suscripción de ChatGPT tienen accesos y facturación separados.")
elif provider == "ChatGPT OAuth":
    try:
        from dermatoai.oauth_ui import render_oauth_controls
        oauth_session, model_name, consent, oauth_context = render_oauth_controls()
    except ImportError:
        st.sidebar.error("Faltan dependencias OAuth. Instale requirements-dermatoai-v2.txt con el Python de este entorno.")
if provider != "ChatGPT OAuth":
    pending = st.session_state.pop("oauth_attempt", None)
    if pending:
        pending.cancel()
config = LLMConfig(provider, model_name, api_key, consent, oauth_session)
if st.sidebar.button("Comprobar conexión"):
    try:
        st.sidebar.info(connection_status(config))
    except Exception as exc:
        st.sidebar.error(str(exc)[:260])
if st.sidebar.button("Probar generación", disabled=provider == "Informe local (sin LLM)"):
    try:
        with st.spinner("Probando el modelo seleccionado…"):
            st.sidebar.success(generation_status(config))
    except Exception as exc:
        st.sidebar.error(str(exc)[:300])
if provider != "Informe local (sin LLM)":
    st.sidebar.caption("La prueba solicita una respuesta breve al motor elegido; no envía documentos ni fotos.")

st.sidebar.divider()
use_rag = st.sidebar.checkbox("Usar documentos", value=True)
method_label = st.sidebar.selectbox("Búsqueda documental", ["Palabras clave (BM25)", "Semántica", "Híbrida"])
method = {"Palabras clave (BM25)": "lexical", "Semántica": "semantic", "Híbrida": "hybrid"}[method_label]
k = st.sidebar.slider("Fragmentos por consulta", 1, 12, 6)
detail = st.sidebar.selectbox("Detalle de respuesta", ["Detallada", "Breve"])
category_filter = st.sidebar.multiselect("Categorías del contexto (vacío = todas)", list(CLASSES), format_func=lambda c: f"{c} · {NAMES[c]}")
language_filter = st.sidebar.multiselect("Idiomas de los pasajes (vacío = todos)", ["es", "en", "und"], default=["es", "und"],
    format_func=lambda l: {"es":"Español", "en":"Inglés", "und":"Sin determinar"}[l])
include_review = st.sidebar.checkbox("Incluir pasajes apartados para revisión", value=False)
if include_review:
    st.sidebar.warning("Está permitiendo fragmentos que las reglas automáticas apartaron. Compruebe las referencias.")
encoder, encoder_error = None, ""
if method != "lexical":
    try:
        encoder = load_encoder(str(ROOT / "knowledge_local" / "embedding_model"))
    except Exception as exc:
        encoder_error = str(exc)[:240]
        st.sidebar.warning(encoder_error)

sources = kb.list_sources()
lookup = {s["id"]: s for s in sources}
if "documents_selected" not in st.session_state:
    st.session_state.documents_selected = list(lookup)
else:
    st.session_state.documents_selected = [i for i in st.session_state.documents_selected if i in lookup]
for ident in st.session_state.pop("add_to_selection", []):
    if ident in lookup and ident not in st.session_state.documents_selected:
        st.session_state.documents_selected.append(ident)
selected = st.sidebar.multiselect("Documentos permitidos", list(lookup), key="documents_selected",
                                  format_func=lambda i: lookup[i]["title"])
st.sidebar.caption(f"{len(sources)} documentos · {sum(s['ready_fragments'] or 0 for s in sources)} pasajes aptos / {sum(s['fragments'] for s in sources)} guardados")
signature = json.dumps([selected, [(i, lookup[i]["sha256"], lookup[i]["metadata"],lookup[i]["pipeline"],lookup[i]["updated"]) for i in selected],
                        use_rag, method, k, detail, provider, model_name, oauth_context, consent, bool(api_key),
                        category_filter, language_filter, include_review, kb.revision()])
if st.session_state.get("context_signature") != signature:
    st.session_state.update(context_signature=signature, report=None, chat=[], visual=None)
if st.sidebar.button("Limpiar análisis y conversación"):
    st.session_state.update(result=None, report=None, chat=[], visual=None, xai=None)
    st.session_state.uploader_version += 1
    st.rerun()
st.sidebar.caption("Los documentos se guardan en este equipo. Las fotos se mantienen en memoria; su envío visual se activa aparte.")

st.title("Análisis de lesiones cutáneas")
st.caption("DermatoAI v2.2 · Clasificador experimental y recuperación documental verificable")
st.info(NOTICE)
analysis_tab, docs_tab, checks_tab, help_tab = st.tabs(["Análisis y preguntas", "Documentos", "Evaluación y examen", "Guía"])


def ask(question, overview=False):
    return respond(question, config, kb, selected, result=st.session_state.result,
                   history=st.session_state.chat, detail=detail, k=k, method=method,
                   encoder=encoder, overview=overview, use_rag=use_rag, categories=category_filter,
                   languages=language_filter, include_review=include_review)


def show_response(response):
    if response.get("warning"):
        st.warning(response["warning"])
    st.caption((f"Redacción con LLM · {response['provider']} · {response.get('model', '')}" if response["generated"] else "Respuesta local / evidencia extractiva") +
               f" · búsqueda {response['retrieval_ms']:.0f} ms · total {response['elapsed_ms']:.0f} ms")
    st.markdown(response["markdown"])
    for hit in response["hits"]:
        with st.expander(f"[{hit['citation_id']}] {hit['title']} · página {hit['page'] or 'no aplica'}"):
            st.text(hit["text"])
            st.caption(f"Sección: {hit.get('section','no registrada')} · idioma: {hit.get('language','und')} · categorías: {', '.join(hit.get('categories',[])) or 'sin clasificar'} · calidad: {hit.get('quality','no registrada')}")
            st.caption(f"Archivo: {hit['filename']} · organización: {hit.get('organization') or 'no registrada'} · ranking: {hit['score']:.4f}")
    if response.get("claims"):
        with st.expander("Soporte literal de las afirmaciones generadas"):
            st.json(response["claims"])

with docs_tab:
    st.subheader("1 · Incorporar y revisar documentos")
    st.write("Cargue PDF con texto, TXT o Markdown. Se conservarán el archivo, el texto por página y sus fragmentos en SQLite.")
    st.code(str(DOCS))
    with st.form("document_upload", clear_on_submit=True):
        upload_doc = st.file_uploader("Documento", type=["pdf", "txt", "md"])
        title = st.text_input("Título")
        organization = st.text_input("Institución / autor")
        url = st.text_input("URL original (opcional)")
        license_value = st.text_input("Licencia o permiso de uso académico")
        document_categories = st.multiselect("Categorías del documento (opcionales; se suman a sugerencias por palabras)",list(CLASSES))
        document_language = st.selectbox("Idioma declarado",["auto","es","en"],format_func=lambda v:{"auto":"Detectar por pasaje","es":"Español","en":"Inglés"}[v])
        document_modality = st.selectbox("Modalidad descrita en el documento",["not_specified","clinical","dermoscopy","mixed"],
            format_func=lambda v:{"not_specified":"No especificada","clinical":"Fotografía / examen clínico","dermoscopy":"Dermatoscopia","mixed":"Mixta"}[v])
        permitted = st.checkbox("He comprobado que puedo usar este documento y que no contiene datos personales no autorizados")
        submitted = st.form_submit_button("Guardar e indexar documento")
    if submitted:
        if upload_doc is None or not permitted:
            st.warning("Elija un documento y confirme su permiso de uso.")
        else:
            try:
                status = kb.save_upload(upload_doc.name, upload_doc.getvalue(),
                        {"title": title or upload_doc.name, "organization": organization, "url": url, "license": license_value, "categories": document_categories, "language": document_language, "modality": document_modality})
                st.session_state.add_to_selection = [s["id"] for s in kb.list_sources() if s["filename"] == upload_doc.name]
                st.session_state.sync_report = {"last_upload": upload_doc.name, "status": status}
                st.rerun()
            except Exception as exc:
                st.error(str(exc)[:400])
    if st.button("Preparar / actualizar documentos de la carpeta"):
        try:
            st.session_state.sync_report = kb.sync()
            st.session_state.add_to_selection = [s["id"] for s in kb.list_sources() if s["id"] not in lookup]
            st.rerun()
        except Exception as exc:
            st.error(str(exc)[:400])
    with st.expander("Estado de la última carga"):
        st.json(st.session_state.sync_report)
    if not sources:
        st.warning("La base está vacía. Incorpore documentos antes de pedir una explicación con fuentes. No hay PDF médicos incluidos de forma ficticia.")
    else:
        st.dataframe([{key: s[key] for key in ("filename", "title", "organization", "license", "fragments", "ready_fragments")} for s in sources], hide_index=True)
        inspect_id = st.selectbox("Documento para inspeccionar", list(lookup), format_func=lambda i: lookup[i]["title"])
        st.json(json.loads(lookup[inspect_id]["preparation_report"] or "{}"))
        fragments = kb.rows([inspect_id],include_review=True)
        part = st.selectbox("Fragmento / página", list(range(len(fragments))),
                            format_func=lambda i: f"Fragmento {i+1} · página {fragments[i]['page'] or 'no aplica'}")
        fragment = fragments[part]
        st.caption(f"Sección: {fragment['section']} · categorías sugeridas: {', '.join(fragment['categories']) or 'sin clasificar'} · idioma: {fragment['language']}")
        st.text_area("Texto realmente extraído", fragment["text"], height=180, disabled=True)
        if fragment['quality'] != 'ready':
            st.warning("Apartado de las consultas normales: " + ", ".join(fragment['reasons']))
            approved = st.checkbox("Revisé el original y autorizo este pasaje para consultas", value=bool(fragment['approved']), key="approve_"+fragment['id']+"_"+lookup[inspect_id]["sha256"]+"_"+hashlib.sha256(lookup[inspect_id]["metadata"].encode()).hexdigest()[:12])
            if st.button("Guardar revisión del pasaje"):
                kb.approve(fragment['id'],approved)
                st.rerun()
        with st.expander("Página completa extraída y versión de origen"):
            raw_pages=kb.raw_pages(inspect_id)
            current=next((r for r in raw_pages if r['page']==fragment['page']),None)
            st.text(current['raw_text'] if current else "Reprocese el original para conservar el texto por página.")
        with st.expander("Editar metadatos del documento"):
            meta=json.loads(lookup[inspect_id]['metadata'])
            with st.form('metadata_'+inspect_id):
                edit_title=st.text_input('Título registrado',meta.get('title',''))
                edit_org=st.text_input('Institución registrada',meta.get('organization',''))
                edit_url=st.text_input('URL registrada',meta.get('url',''))
                edit_license=st.text_input('Licencia registrada',meta.get('license',''))
                edit_categories=st.multiselect('Categorías registradas',list(CLASSES),default=meta.get('categories',[]))
                edit_language=st.selectbox('Idioma registrado',['auto','es','en'],index=['auto','es','en'].index(meta.get('language','auto')) if meta.get('language','auto') in ['auto','es','en'] else 0)
                edit_modality=st.selectbox('Modalidad registrada',['not_specified','clinical','dermoscopy','mixed'],index=['not_specified','clinical','dermoscopy','mixed'].index(meta.get('modality','not_specified')))
                save_meta=st.form_submit_button('Actualizar metadatos y reprocesar')
            if save_meta:
                try:
                    name=lookup[inspect_id]['filename']
                    kb.save_upload(name,(DOCS/name).read_bytes(),dict(title=edit_title,organization=edit_org,url=edit_url,license=edit_license,
                         categories=edit_categories,language=edit_language,modality=edit_modality))
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc)[:300])
        st.download_button("Exportar texto y fragmentos seleccionados", json.dumps(fragments, ensure_ascii=False, indent=2), "fragmentos.json", "application/json")
        confirmed_delete = st.checkbox("Eliminar este documento, su archivo y sus fragmentos")
        if st.button("Eliminar documento", disabled=not confirmed_delete):
            kb.delete(inspect_id)
            st.session_state.sync_report = None
            st.rerun()
    with st.expander("Embeddings opcionales para búsqueda semántica"):
        st.write("La búsqueda por palabras funciona sin descargar modelos. Para comparar recuperación semántica o híbrida:")
        st.code("python -m scripts_dermatoai.download_embeddings\npython -m scripts_dermatoai.manage_knowledge vectors")
        if st.button("Calcular / actualizar embeddings", disabled=not sources):
            try:
                with st.spinner("Calculando vectores locales…"):
                    enc = load_encoder(str(ROOT / "knowledge_local" / "embedding_model"))
                    count = kb.build_vectors(enc)
                st.success(f"{count} embeddings guardados. Puede elegir Semántica o Híbrida.")
            except Exception as exc:
                st.error(str(exc)[:400])
    st.download_button("Descargar copia de SQLite", DATABASE.read_bytes(), "rag_dermatoai_backup.sqlite3", "application/octet-stream")
    st.caption("Una copia completa incluye SQLite, documents y catalog.json. Los cambios de documentos requieren recalcular sus embeddings si utiliza búsqueda semántica.")

with analysis_tab:
    st.subheader("2 · Fotografía y resultado experimental")
    col1, col2 = st.columns([1, 1.4])
    image = None
    with col1:
        photo = st.file_uploader("Fotografía anonimizada (opcional para consultar documentos)", type=["jpg", "jpeg", "png", "webp"], key=f"photo_{st.session_state.uploader_version}")
        if photo:
            try:
                body = photo.getvalue()
                image = image_from_bytes(body)
                st.image(image, width=320)
                digest = hashlib.sha256(body).hexdigest()
                if st.session_state.get("image_fingerprint") != digest:
                    st.session_state.update(image_fingerprint=digest, result=None, visual=None, report=None, chat=[], xai=None)
            except Exception as exc:
                st.error(str(exc)[:300])
        elif st.session_state.result is not None or st.session_state.visual is not None:
            st.session_state.update(result=None, visual=None, report=None, chat=[], xai=None)
        checkpoints = checkpoint_files()
        checkpoint_names = [p.name for p in checkpoints]
        checkpoint = st.selectbox("Checkpoint entrenado", checkpoint_names or ["Sin pesos .pt"], index=checkpoint_names.index("vits16_robust_s1.pt") if "vits16_robust_s1.pt" in checkpoint_names else 0)
        device = st.selectbox("Dispositivo", ["cpu", "cuda"])
        checkpoint_key = (checkpoint, device)
        if st.session_state.get("checkpoint_key") != checkpoint_key:
            st.session_state.update(checkpoint_key=checkpoint_key, result=None, report=None, xai=None, chat=[])
        if not checkpoints:
            st.warning("Faltan los pesos entrenados. Puede consultar documentos sin ellos.")
            st.code("results/checkpoints/<nombre_original_de_la_corrida>.pt")
        if st.button("Analizar imagen", type="primary", disabled=image is None or not checkpoints):
            try:
                path = ROOT / "results" / "checkpoints" / checkpoint
                with st.spinner("Ejecutando el modelo original…"):
                    classifier = load_classifier(str(path), path.stat().st_size, path.stat().st_mtime_ns, device)
                    st.session_state.result = classifier.predict(image)
                st.session_state.update(report=None, xai=None, chat=[])
            except Exception as exc:
                st.error("No se pudo clasificar: " + str(exc)[:400])
        st.caption(SCOPE)
    with col2:
        result = st.session_state.result
        if result:
            from dermatoai.agent import base_report
            st.metric("Clase con mayor puntuación", result["top_class"])
            st.metric("Puntuación softmax", f"{result['top_score']:.4f}")
            st.bar_chart({"Puntuación": result["scores"]}, color="#157f88")
            with st.expander("Interpretación y datos técnicos"):
                st.markdown(base_report(result))
                st.json(result)
            if st.button("Mostrar mapa del modelo"):
                try:
                    path = ROOT / "results" / "checkpoints" / checkpoint
                    classifier = load_classifier(str(path), path.stat().st_size, path.stat().st_mtime_ns, device)
                    st.session_state.xai = classifier.explain(image)
                except Exception as exc:
                    st.error(str(exc)[:400])
            if st.session_state.xai is not None:
                overlay, label = st.session_state.xai
                st.image(overlay, caption=label, width=320)
                st.caption("Mapa del comportamiento del modelo; no constituye un hallazgo clínico ni lo lee el redactor.")
        else:
            st.write("La consulta documental funciona aunque no haya fotografía o checkpoint.")
    with st.expander("Descripción visual opcional de la fotografía"):
        st.write("Utiliza un modelo que admita imágenes. Es una descripción experimental separada del clasificador y de los PDF; no interpreta clínicamente el mapa de atención.")
        visual_consent = st.checkbox("Activo la lectura visual y autorizo procesar esta fotografía con el motor elegido", value=False)
        if st.button("Describir fotografía", disabled=image is None or not visual_consent):
            try:
                from dermatoai.vision import describe
                with st.spinner("Solicitando descripción visual…"):
                    st.session_state.visual = describe(image, config, consent=visual_consent)
                st.session_state.report = None
            except Exception as exc:
                st.error("Descripción visual no disponible: " + str(exc)[:350])
        if st.session_state.visual:
            st.markdown(st.session_state.visual["markdown"])
    st.subheader("3 · Consultar los documentos")
    st.caption("Cada pregunta vuelve a buscar evidencia. Las preguntas sugeridas son opcionales; no son respuestas almacenadas.")
    overview = st.checkbox("Panorama de pasajes aptos (hasta 12; sin lectura integral del corpus)", value=False)
    suggestion = st.selectbox("Pregunta sugerida", ["Explica el resultado experimental según los documentos seleccionados.",
        "Resume los temas principales de los documentos seleccionados.", "¿Qué limitaciones describen las fuentes seleccionadas?"])
    question = st.text_area("Escribe tu pregunta", placeholder="Puedes preguntar sin cargar una imagen.", height=90)
    qa, qb = st.columns(2)
    run = qa.button("Responder mi pregunta", type="primary")
    suggested = qb.button("Consultar pregunta sugerida")
    if run or suggested:
        try:
            with st.spinner("Buscando y preparando la respuesta…"):
                response = ask(suggestion if suggested else question, overview)
            st.session_state.chat.append(response)
        except Exception as exc:
            st.error("No se pudo consultar: " + str(exc)[:400])
    if st.button("Generar / actualizar informe", disabled=not selected):
        try:
            response = ask("Explica el resultado experimental y el contexto documental pertinente." if result else "Explica los temas principales de los documentos seleccionados.", overview=not bool(result))
            st.session_state.report = make_report(response, result, st.session_state.visual)
        except Exception as exc:
            st.error(str(exc)[:400])
    if st.session_state.report:
        with st.expander("Informe completo", expanded=True):
            st.markdown(st.session_state.report)
            st.download_button("Descargar informe", st.session_state.report, "informe_dermatoai.md", "text/markdown")
    for chat in st.session_state.chat:
        with st.chat_message("user"):
            st.text(chat["question"])
        with st.chat_message("assistant"):
            show_response(chat)
    if st.session_state.chat:
        st.download_button("Exportar conversación y evidencias", json.dumps(st.session_state.chat, ensure_ascii=False, indent=2), "conversacion_dermatoai.json", "application/json")

with checks_tab:
    from dermatoai.ui_evaluation import render_evaluation
    render_evaluation(ROOT, config, kb)

with help_tab:
    st.subheader("Ruta de preparación")
    st.markdown('''1. Abra **Documentos**, incorpore archivos y compruebe el texto extraído.
2. Seleccione documentos en la barra lateral. BM25 funciona sin descarga de embeddings.
3. Configure un LLM solo si desea redacción generada; el modo local muestra pasajes.
4. Escriba preguntas libres en **Análisis y preguntas**. La fotografía es opcional.
5. Para clasificación, conserve el checkpoint original y los archivos src del repositorio.
6. Revise **Evaluación y examen** y el manual PDF antes de presentar resultados.''')
    st.code("python -m scripts_dermatoai.check_v2\npython -m pytest tests_dermatoai -q")
    st.caption("No se incluyen pesos, fotografías ni PDF médicos en la extensión. Consulte el manual para crear el corpus, medirlo y declarar lo pendiente.")
