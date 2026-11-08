"""Evidencia del examen: resultados nuevos e históricos identificados por separado."""
import json
import streamlit as st

def render_evaluation(root, config, kb):
    st.subheader("Evaluación RAG con preguntas anotadas")
    st.write("Suba JSONL con id, question, relevant:[{filename,page}], unanswerable y allowed_files opcional. Use preguntas de prueba que no haya empleado para ajustar parámetros.")
    cases_file = st.file_uploader("Casos de prueba RAG (JSONL)", type=['jsonl'])
    if st.button("Medir línea base BM25 y ablación sin recuperación", disabled=cases_file is None):
        try:
            from .rag_metrics import evaluate_cases
            cases = [json.loads(line) for line in cases_file.getvalue().decode('utf-8-sig').splitlines() if line.strip()]
            measured = evaluate_cases(kb, cases)
            measured['ablation_without_retrieval'] = evaluate_cases(kb, cases, method='none')
            st.session_state.rag_evaluation = measured
        except Exception as exc:
            st.error(str(exc)[:400])
    if st.session_state.get('rag_evaluation'):
        report=st.session_state.rag_evaluation
        st.json({k:v for k,v in report.items() if k!='cases'})
        st.caption("Se ejecuta el mismo conjunto sin recuperación como control. Los resultados corresponden a recuperación documental, no a calidad de generación.")
        st.download_button("Descargar evaluación RAG", json.dumps(report,ensure_ascii=False,indent=2), 'evaluacion_rag.json','application/json')
    st.subheader("Clasificador: resultados del repositorio")
    path=root/'results_dermatoai/audit/historical_comparison.csv'
    if path.exists():
        import pandas as pd
        st.caption("Resultados históricos del repositorio; no son ejecuciones nuevas de esta entrega.")
        st.dataframe(pd.read_csv(path),hide_index=True)
    else:
        st.code('python -m scripts_dermatoai.audit')
    st.write("Para una evaluación propia del clasificador use el lote etiquetado, el mismo split en todas las corridas y los comandos del manual. No confunda las pruebas de software con la evaluación del modelo.")
    st.subheader("Confirmaciones para el examen final")
    confirmations=[('R1','Arquitectura, formulación, entrenamiento propio y aporte del equipo documentados'),
        ('R2','Reutilización del repositorio y contribución original del equipo declaradas'),
        ('R3','Línea base evaluada sobre las mismas particiones'),
        ('R4','Ablación, métricas, ausencia de fuga y semillas verificadas con resultados propios'),
        ('R5','Mapa de atención / Grad-CAM o análisis de errores revisado'),
        ('R6','Hardware, horas de GPU, costo, sesgos y riesgos registrados'),
        ('ENTREGA','Artículo IEEE máximo 12 páginas, 6 referencias científicas, contribuciones, diapositivas PDF, repo y pesos disponibles')]
    confirmed={key:st.checkbox(label,key='exam_'+key) for key,label in confirmations}
    st.caption("Marcar casillas registra su revisión; no demuestra por sí solo el cumplimiento. Conserve los archivos que sustentan cada punto.")
    st.download_button("Exportar mis confirmaciones",json.dumps(confirmed,ensure_ascii=False,indent=2),'confirmaciones_examen.json','application/json')
