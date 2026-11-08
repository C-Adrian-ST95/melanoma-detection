"""Controles locales OAuth; no cambia el agente RAG ni el clasificador."""
import streamlit as st
from .oauth_chatgpt import OAuthBroker, OAuthError

@st.cache_resource(max_entries=1)
def local_broker():
    return OAuthBroker()

def render_oauth_controls():
    try:
        broker = local_broker()
        accounts = broker.accounts()
    except Exception:
        st.sidebar.error("No se pudo abrir la configuración OAuth local. Revise los permisos de su perfil de usuario.")
        return None, "", False, ""
    by_id = {account["client_id"]: account for account in accounts}
    options = [""] + list(by_id)
    next_id = st.session_state.pop("oauth_next_account", None)
    if next_id in by_id:
        st.session_state.oauth_account = next_id
    if st.session_state.get("oauth_account", "") not in options:
        st.session_state.oauth_account = ""
    selected = st.sidebar.selectbox("Cuenta ChatGPT", options, key="oauth_account",
                        format_func=lambda ident: by_id[ident]["label"] if ident else "Conectar otra cuenta / espacio")
    flash = st.session_state.pop("oauth_flash", "")
    if flash:
        st.sidebar.info(flash)
    st.sidebar.caption("Conexión propia de DermatoAI. Las autorizaciones de otras aplicaciones se mantienen.")
    if st.sidebar.button("Preparar inicio de sesión"):
        previous = st.session_state.pop("oauth_attempt", None)
        if previous:
            previous.cancel()
        try:
            st.session_state.oauth_attempt = broker.start_login(selected or None)
        except OAuthError as exc:
            st.sidebar.error(str(exc))
    attempt = st.session_state.get("oauth_attempt")
    if attempt:
        st.sidebar.link_button("Continuar con ChatGPT", attempt.url)
        st.sidebar.caption("Autoriza en el navegador; después regresa aquí. El retorno debe abrirse en este mismo equipo.")
        if st.sidebar.button("Finalizar conexión"):
            try:
                client_id = broker.finish_login(attempt)
                st.session_state.oauth_next_account = client_id
                st.session_state.pop("oauth_attempt", None)
                st.session_state.oauth_models = []
                st.session_state.oauth_models_owner = client_id
                try:
                    st.session_state.oauth_models = broker.models(client_id)
                    message = "Cuenta conectada. Seleccione un modelo de los disponibles."
                except OAuthError as exc:
                    message = "Identidad verificada. " + str(exc)
                st.session_state.oauth_flash = message
                st.session_state.update(report=None, chat=[])
                st.rerun()
            except OAuthError as exc:
                st.sidebar.error(str(exc))
                if attempt.consumed:
                    st.session_state.pop("oauth_attempt", None)
        if st.sidebar.button("Cancelar inicio de sesión"):
            attempt.cancel()
            st.session_state.pop("oauth_attempt", None)
            st.rerun()
    current = by_id.get(selected)
    if not current or not current["connected"]:
        st.sidebar.caption("Sin sesión activa para esta selección.")
        return None, "", False, selected
    if not current["plan_enabled"]:
        st.sidebar.warning("Identidad conectada sin permiso de inferencia. Reconecte y autorice el uso del plan.")
    else:
        st.sidebar.success("Cuenta conectada con permiso de uso del plan.")
    if st.sidebar.button("Consultar / actualizar modelos"):
        try:
            st.session_state.oauth_models = broker.models(selected)
            st.session_state.oauth_models_owner = selected
        except OAuthError as exc:
            st.sidebar.error(str(exc))
    models = st.session_state.get("oauth_models", []) if st.session_state.get("oauth_models_owner") == selected else []
    labels = {item["slug"]: item["display_name"] for item in models}
    model = st.sidebar.selectbox("Modelo disponible en esta cuenta", list(labels),
              format_func=lambda slug: labels[slug], key="oauth_model_" + selected) if labels else ""
    if not labels:
        st.sidebar.caption("Consulte modelos para cargar el catálogo autorizado de esta cuenta.")
    consent = st.sidebar.checkbox("Autorizo enviar consulta, puntuaciones y fragmentos seleccionados a ChatGPT",
                                  value=False, key="oauth_data_consent")
    st.sidebar.caption("Este permiso corresponde al texto. La imagen requiere activar su envío por separado. Uso y límites: Ajustes de ChatGPT → Uso.")
    if st.sidebar.button("Cerrar sesión de esta conexión"):
        try:
            confirmed = broker.sign_out(selected)
            st.session_state.oauth_flash = ("Sesión cerrada y revocación confirmada." if confirmed else
                "Sesión cerrada en este equipo. No se confirmó la revocación remota; puede desconectar DermatoAI en Ajustes de ChatGPT.")
            st.session_state.update(report=None, chat=[], oauth_models=[])
            st.rerun()
        except OAuthError as exc:
            st.sidebar.error(str(exc))
    return broker.session(selected), model, consent, selected
