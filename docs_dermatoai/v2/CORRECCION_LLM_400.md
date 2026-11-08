# Corrección 2.2: generación LLM del RAG

La captura mostraba una llamada rechazada con HTTP 400 y el modo extractivo de respaldo. No era una respuesta redactada por el LLM. La versión anterior ocultaba el campo rechazado y sugería reconectar incluso cuando fallaba el formato de la solicitud.

## Defectos corregidos

El transporte de **ChatGPT OAuth** enviaba las consultas de texto como una cadena en `input` y añadía `max_output_tokens`. El uso del plan ChatGPT exige `input` como lista y no admite ese parámetro. Se corrigieron ambos puntos, conservando `store=false`, `stream=true` y el modelo elegido por la cuenta. La llamada con imágenes conserva su lista multimodal; sigue requiriendo un modelo compatible y la activación visual explícita.

Fuente oficial consultada el 7 de noviembre de 2026:
https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations

El transporte con API key es independiente: conserva sus parámetros admitidos por la API habitual. No se reemplazan credenciales ni se cambian permisos de otras aplicaciones.

## Aplicar en su instalación

1. Detenga Streamlit con **Ctrl+C** en la terminal.
2. Haga una copia de su carpeta de trabajo. Extraiga `DermatoAI_RAG_v2.zip` en la raíz de `melanoma-detection-main`, reemplazando los archivos de la extensión. No cree una segunda carpeta `dermatoai` dentro de la existente.
3. Inicie de nuevo con `INICIAR_DERMATOAI.ps1` o el mismo comando de Streamlit que ya utilizaba. La cabecera debe indicar **v2.2**. Esta corrección no añade dependencias respecto de v2.1.
4. En **Motor de explicación**, seleccione el proveedor que utiliza, la cuenta conectada y un modelo habilitado. Mantenga la autorización de envío de consulta y fragmentos.
5. Pulse **Probar generación**. Este botón hace una petición breve al modelo, sin documentos, historial ni fotos. Utiliza el acceso/cuota del motor seleccionado. Debe aparecer **Generación verificada**; comprobar el catálogo de modelos por sí solo no prueba la generación.
6. Seleccione sus documentos y formule de nuevo la pregunta. Las respuestas guardadas en la conversación anterior no se regeneran solas. La respuesta nueva debe indicar **Redacción con LLM**, proveedor y modelo, y conservar las citas y sus pasajes desplegables.

Si ya preparó los PDF y embeddings con v2.1, esta corrección no requiere volver a prepararlos ni volver a calcularlos. No borre `knowledge_dermatoai`, `rag.sqlite3`, `catalog.json` ni la carpeta local del encoder para resolver un HTTP 400.

## Qué recibe el LLM y qué se presenta

Cada pregunta recupera pasajes actuales de las fuentes seleccionadas. El LLM recibe la consulta, el nivel de detalle, esos pasajes con sus metadatos, el resultado experimental si existe y un historial breve. La consulta documental no incluye los píxeles de la foto.

La respuesta generada son párrafos explicativos con citas `[S1]`, `[S2]`, etc. Se muestran las referencias con documento, sección y página, y un desplegable con el soporte literal. Si existe resultado del clasificador, puede añadirse la relación conceptual con ese resultado, sin convertirlo en diagnóstico ni atribuir a la imagen signos de la literatura.

Una respuesta solo se marca como generada después de terminar el streaming y comprobar el JSON, los identificadores de las citas y las citas literales en los pasajes actuales. Esto no acredita por sí solo fidelidad semántica; revise las paráfrasis antes de presentar.

Si faltan fuentes o no hay evidencia recuperada, no se pide al LLM que invente una respuesta. Si la llamada o su validación falla, se informa del fallo y se conservan los pasajes originales como respaldo. Ya no se sugiere simplemente conectar un LLM cuando el proveedor elegido intentó responder y falló.

## Comparaciones y pertinencia

Para preguntas explícitas sobre diferencias entre dos categorías reconocidas, se consideran hasta 24 candidatos dentro de los filtros elegidos, se priorizan los que mencionan ambos temas y las secciones de definición o diagnóstico, y se apartan pasajes de otras categorías o de secciones de tratamiento cuando no se solicita ese tema. Los identificadores se asignan de nuevo a los pasajes finales antes de llamar al LLM.

Son reglas de recuperación, no respuestas predeterminadas ni validación médica. La selección de documentos, idioma, calidad y categorías manuales se conserva. Si su corpus no contiene una comparación respaldada, debe incorporar una fuente pertinente; activar un LLM no añade documentos ausentes.

## Si vuelve a fallar

- **HTTP 400:** solicitud o modelo incompatible; el aviso incluye un código o campo cuando el proveedor los entrega y están reconocidos. Comparta ese aviso completo y el nombre del modelo, sin credenciales.
- **HTTP 401:** sesión inválida; vuelva a conectar la cuenta.
- **HTTP 403:** compruebe los permisos y el acceso al modelo.
- **HTTP 429:** revise límites de uso/cuota y espere a que vuelvan a estar disponibles.
- **HTTP 5xx:** fallo temporal del proveedor; vuelva a intentar.
- **JSON/citas inválidas:** el modelo respondió, pero el contenido no superó la comprobación. Revise los pasajes y el soporte literal; no es necesariamente un problema de conexión.
- **Encoder/embeddings:** es un requisito de recuperación semántica separado de la generación. BM25 permite consultar sin encoder; usar un LLM no calcula los vectores.

No comparta tokens, archivos de autorización ni la URL completa del callback. Los nuevos avisos no muestran el cuerpo de error del proveedor, que podría repetir datos de la consulta.

## Archivos modificados y validación

Se modificaron `dermatoai/oauth_chatgpt.py`, `dermatoai/llm.py`, `dermatoai/service.py` y `app_dermatoai.py`, junto con pruebas, esta guía, el LEEME y el manifiesto. El clasificador y los 106 archivos originales conservan sus hashes. El PDF operativo de v2.1 continúa siendo la guía general; este documento describe la corrección de v2.2.

La validación incluye un transporte OAuth simulado que rechaza expresamente `input` como cadena y `max_output_tokens`, un recorrido SQLite → recuperación → OAuth → streaming → validación de citas, y el botón de generación en Streamlit. No se dispone aquí de su sesión ni se ha probado una llamada con su cuenta real. La comprobación final en su equipo es **Probar generación** y una consulta nueva con sus fuentes.
