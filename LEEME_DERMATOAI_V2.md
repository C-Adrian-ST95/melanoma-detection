# DermatoAI v2.2: entrega completa de la extensión

Lea primero `docs_dermatoai/v2/CORRECCION_LLM_400.md`: corrige la solicitud OAuth del RAG y añade **Probar generación**. El PDF `docs_dermatoai/v2/Manual_DermatoAI_v2.pdf` conserva las instrucciones generales de v2.1; esta guía explica los cambios de v2.2. Esta entrega sustituye los archivos de la extensión DermatoAI de entregas anteriores; no necesita aplicar antes los parches anteriores. Se coloca sobre su repositorio original completo.

## Empezar en Windows

1. Detenga Streamlit y respalde su carpeta actual.
2. Extraiga este ZIP en la raíz de `melanoma-detection-main`, donde están `src`, `results` y `app.py`. El ZIP no incluye ni sobrescribe esos archivos originales.
3. Ejecute `powershell -ExecutionPolicy Bypass -File .\PREPARAR_DERMATOAI.ps1` para preparar un entorno independiente.
4. Ejecute `powershell -ExecutionPolicy Bypass -File .\INICIAR_DERMATOAI.ps1`.
5. Abra **Documentos**, cargue un PDF con texto y revise los fragmentos. También puede copiar archivos a `knowledge_dermatoai/documents` y sincronizar.
6. Consulte libremente sin imagen. Configure un motor LLM solo si desea una explicación generada. Para clasificar necesita sus pesos BEST en `results/checkpoints`.

La base nueva es `knowledge_dermatoai/rag.sqlite3`. No se borra ni se migra automáticamente su índice anterior. La consulta BM25 funciona sin embeddings; el manual explica la opción semántica/híbrida.

## Qué contiene / qué debe aportar

Incluye interfaz, módulos, scripts, pruebas, plantillas de evaluación, manifiesto de integridad y manual editable/PDF. No incluye `src`, datasets, pesos .pt, fotos, adjuntos originales del examen, artículo ni diapositivas. El corpus médico inicial está vacío: incorpore sus documentos y sus metadatos autorizados.

Comprobaciones: `python -m scripts_dermatoai.check_v2`, `python -m pytest tests_dermatoai -q`. Las pruebas incluyen el contrato de la solicitud OAuth, generación con citas, diagnóstico de errores HTTP y comparación temática. Las pruebas no son entrenamiento ni evaluación clínica; las conexiones LLM reales, sus pesos y sus documentos requieren validación en su equipo.

Revise `examples_dermatoai/v2/confirmaciones.csv` antes de presentar. El manual relaciona R1-R6 y los siete criterios del examen con evidencias y pendientes. Se conservan las referencias a resultados históricos como históricos, sin atribuirlos como experimentos propios.
