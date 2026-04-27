# LangGraph Reputation Router (Single File)

Sistema simple con 3 agentes expertos (Historia, Literatura, Biologia) para preguntas sobre animales mitologicos, con enrutamiento por reputacion y feedback humano.

## Quick start

1) Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

2) Ejecutar con OpenAI (requiere `OPENAI_API_KEY`):

```powershell
$env:OPENAI_API_KEY="your_key_here"
python main.py --question "Que puedes decirme del grifo?" --user julian
```

## Log y trazabilidad

- Cada respuesta y feedback se guarda en `feedback_log.txt` con usuario y timestamp.
- Si hay feedback negativo en preguntas similares, el router redirige y lo informa.
- Si hay feedback positivo, lo indica y mantiene el experto.

## Guardar imagen del grafo

```powershell
python main.py --save-graph langgraph.png --question "Habla del fenix"
```

## Guardar solo la imagen del grafo (sin pregunta)

```powershell
python main.py --save-graph-only langgraph.png
```

## Human feedback (MITL)

Cuando el grafo llega a `human_feedback`, se pausa con `interrupt`:

- Ingresa `1` a `5`, o `Muy negativo` / `Muy positivo`.
- Se reanuda y guarda el evento en `feedback_log.txt`.

## Files

- `main.py` - implementacion completa.
- `feedback_log.txt` - log con usuario, timestamp, pregunta y feedback.
- `requirements.txt` - dependencias.
