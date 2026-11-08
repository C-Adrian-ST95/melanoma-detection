# Artículo IEEE con apéndice RAG

Extraiga el ZIP en una carpeta. La carpeta `paper` contiene el artículo completo y sus fuentes. `results` reúne las figuras y tablas que se incluyen desde el LaTeX. `evaluacion` contiene los casos, el inventario y los JSON de las tres corridas utilizados en el apéndice.

## Compilar con Tectonic

En PowerShell, situado en la carpeta extraída, ejecute:

```powershell
cd paper
tectonic -X compile main.tex --keep-logs
```

El resultado se guarda como `paper/main.pdf`. Conserve la estructura de carpetas, ya que el documento lee figuras y tablas desde `../results`.

El apéndice se incorpora mediante `\input{rag_ampliacion.tex}` antes de la bibliografía. Las nuevas referencias están añadidas a `refs.bib`.

## Verificación de la entrega

El PDF adjunto se generó con XeLaTeX y BibTeX, usando una configuración local de idioma español. El ejecutable de Tectonic no estaba instalado y la descarga no respondió desde el entorno de compilación. Las fuentes conservan la configuración original de Babel para Tectonic. El PDF adjunto tiene 10 páginas.

El apéndice documenta la evaluación de recuperación con 20 preguntas. La evaluación de las respuestas del LLM y la revisión de sus citas requieren las conversaciones de la aplicación. Complete las tareas de los autores en la tabla de contribuciones para la entrega del curso.
