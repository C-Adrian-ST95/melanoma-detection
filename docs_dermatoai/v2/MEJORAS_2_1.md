# Actualización 2.1: preparar documentos antes de responder

Se conserva la carpeta `knowledge_dermatoai/documents` y SQLite `knowledge_dermatoai/rag.sqlite3`. El clasificador, las etiquetas de evaluación y los archivos originales no cambian.

## Aplicar sobre la versión 2

1. Detenga Streamlit y respalde `knowledge_dermatoai` completo.
2. Extraiga este ZIP sobre el mismo repositorio, reemplazando solo los archivos de la extensión. Es la extensión completa actualizada, no requiere aplicar otros parches.
3. Actualice las dependencias con el Python que utiliza para Streamlit: instale `requirements-dermatoai-v2.txt`. Se añadió pdfplumber para conservar posiciones de texto.
4. Inicie de nuevo la página. Abra Documentos y pulse **Preparar / actualizar documentos de la carpeta**.
5. Revise pasajes aptos y apartados; confirme secciones, páginas, categorías e idioma.
6. Si usa búsqueda semántica o híbrida, vuelva a calcular embeddings después del reprocesado.

La migración conserva una copia inicial `rag.sqlite3.pre_21.bak` antes de ampliar el esquema de fragmentos. Si existía de otra ejecución, no se sobrescribe. La copia de respaldo completa de toda la carpeta sigue siendo responsabilidad del usuario. El índice previo no se borra. Los pasajes antiguos quedan pendientes hasta reprocesar sus archivos originales; si un PDF original falta, no se inventa una nueva extracción.

## Cambios en la ingesta

- Orden de texto de dos columnas estimado por posiciones; muestra una advertencia para que revise las páginas detectadas.
- Limpieza de encabezados y pies repetidos y unión de palabras cortadas por guion entre líneas.
- Separación por títulos, párrafos y frases. Se conserva la página; no se mezclan secciones. Límite de 1.500 caracteres y hasta 180 de solapamiento mediante frases completas.
- Guarda el texto de la página y los fragmentos en SQLite. El PDF original queda intacto.
- Cada pasaje conserva sección, categorías sugeridas, idioma estimado, modalidad declarada y estado de calidad.
- Las categorías sugeridas se basan en palabras presentes; no validan el contenido clínico. Puede añadir categorías documentales manualmente.
- El idioma se estima con palabras frecuentes; no es un detector lingüístico validado. En textos cortos puede quedar sin determinar.

## Apartar sin destruir

Referencias bibliográficas, secuencias muy numéricas y texto poco legible se conservan pero quedan fuera de las consultas normales. Son reglas heurísticas: no detectan todas las tablas ni garantizan que cada pasaje marcado sea irrelevante.

En Documentos seleccione el fragmento, compare el texto y la página original, y autorice individualmente un pasaje si corresponde. También existe la opción avanzada **Incluir pasajes apartados para revisión**, desactivada por defecto. No active esa opción solo para forzar una respuesta.

Se puede editar título, institución, URL, licencia, categorías, idioma y modalidad en **Editar metadatos del documento**. Reprocesar sustituye los fragmentos y reinicia aprobaciones de esa versión, e invalida sus embeddings antiguos.

## Qué verá al preguntar

La búsqueda filtra documentos, categorías, idiomas y calidad antes de ordenar los resultados. Español y sin determinar son los idiomas iniciales de la interfaz. Una lista vacía de categorías o idiomas significa todos; una lista vacía de documentos significa ninguna fuente.

El Panorama ya no toma posiciones arbitrarias del PDF: prioriza resúmenes/introducciones y diversidad de secciones, con un máximo de 12 pasajes aptos. Aun así no es una lectura completa de todo el corpus. Para una categoría concreta, prefiera una pregunta y su filtro correspondiente.

Si no hay un LLM disponible, se muestran hasta tres pasajes breves, de hasta 420 caracteres, con las referencias desplegables para el texto completo. No se presenta ese modo como una explicación generada. Un LLM conectado recibe los pasajes recuperados y debe devolver afirmaciones con soporte literal.

## Validación y referencias

La suite actual contiene 100 pruebas aprobadas de software. Se comprobaron controles de calidad, revisión manual, idiomas, fuentes, migración y páginas. Se ensayó también la extracción de un artículo real de nueve páginas y una búsqueda híbrida con encoder real sobre texto artificial. Ninguna de estas pruebas acredita fidelidad clínica de las respuestas ni sustituye la evaluación propia exigida por el examen.

Consulte `ARTICULOS_CIENTIFICOS_ES.md`: contiene seis referencias en español y un complemento, enlaces a sus PDF y límites de aplicación. No se distribuyen textos completos de terceros. Use el catálogo de ejemplo solo después de revisar los archivos y sus permisos; no reemplace a ciegas su catálogo existente.
