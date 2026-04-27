# Explicacion del codigo (LangGraph + enrutamiento con feedback)

Este documento explica **de forma simple** lo que hace cada parte del codigo y en que se basa el enrutamiento. La idea es que puedas entender el flujo completo sin leer todo el archivo de golpe.

## Objetivo general

El programa hace lo siguiente:

1) Recibe una **pregunta sobre animales mitologicos**.
2) Decide **que experto** (Historia, Literatura o Biologia) debe responder, usando:
   - Analisis de la pregunta.
   - Analisis de TODO el historial de feedback guardado en un TXT.
   - Un razonamiento adicional usando el LLM para elegir mejor.
3) Genera una respuesta **en streaming** con OpenAI.
4) Pide feedback humano (1-5 o Muy positivo/Muy negativo).
5) Guarda el feedback y el ruteo en un archivo de log

---

## Estructura del archivo

### 1) Constantes y configuracion basica

- `LOG_FILE`: ruta del archivo donde se guarda el historial de feedback (`feedback_log.txt`).
- `AGENTS`: mapa de los 3 expertos disponibles.
- `VALID_AGENTS`: conjunto con las claves validas (`historia`, `literatura`, `biologia`).

### 2) Estado del grafo (GraphState)

`GraphState` es un modelo Pydantic que representa el **estado compartido** entre nodos del grafo:

- `user_question`: la pregunta del usuario.
- `selected_agent`: experto elegido por el router.
- `expert_response`: respuesta final del experto.
- `response_streamed`: marca si ya se imprimio la respuesta en streaming.
- `feedback_text` y `feedback_score`: feedback humano.
- `routing_message`: mensaje de por que se eligio ese experto.

---

## Flujo de datos y funciones principales

### 3) Streaming con OpenAI

**Funcion:** `stream_chat`

- Llama al LLM en modo streaming.
- Imprime token a token en la terminal.
- Devuelve el texto completo para guardarlo en el estado.

### 4) Log y trazabilidad

**Funciones:**
- `log_event`: escribe cada evento como JSON en `feedback_log.txt`.
- `load_events`: lee el historial del TXT.
- `send_trace`: si existe `TRACE_URL`, envia el evento a un API externo.

Cada linea del log tiene:

- `timestamp`
- `user`
- `question`
- `event_type` (routing / feedback)
- `agent`, `feedback`, `sentiment`, etc.

---

## Enrutamiento "inteligente" basado en feedback

### 5) Analisis del historial

**Funciones:**
- `build_feedback_stats`: recorre TODO el log y calcula:
  - Conteo global de feedback (positivo/negativo/neutral) por agente.
  - Conteo para preguntas similares.
  - Ejemplos de preguntas similares (top 5).
- `similarity`: calcula similitud simple por tokens (interseccion / union).

La similitud se usa para ver si una pregunta nueva se parece a una anterior.

### 6) Eleccion del agente con LLM

**Funcion:** `choose_agent_with_llm`

- Prepara un JSON con la pregunta, el experto sugerido y los stats del log.
- Pide al LLM que elija el agente **y justifique** por que.
- Si el LLM no devuelve JSON valido, se usa `fallback_agent_from_stats`.

### 7) Fallback determinista

**Funcion:** `fallback_agent_from_stats`

- Si el experto preferido tiene muchos negativos en preguntas similares,
  cambia al agente con mas positivos y menos negativos.
- Si no hay historial relevante, se queda con el experto sugerido.

---

## Clasificacion inicial por la pregunta

**Funcion:** `classify_question`

Detecta palabras clave:

- Literatura: poema, novela, mito, simbolo...
- Historia: origen, cronologia, civilizacion...
- Biologia: especie, evolucion, ecosistema...

Esto no decide por si solo. Solo da un **experto sugerido** para el router.

---

## Construccion del grafo (LangGraph)

**Funcion:** `build_graph`

Nodos:

1) `router`
   - Llama a `choose_agent_with_llm`.
   - Registra el evento de ruteo en el log.
   - Guarda `routing_message`.

2) `historia` / `literatura` / `biologia`
   - Cada uno tiene un prompt largo y especializado.
   - Responde **directo** y en español.

3) `human_feedback`
   - Pausa el grafo (MITL).
   - Recibe feedback y lo guarda en el log.

4) `update_reputation`
   - Esta vacio (se deja por compatibilidad del flujo).

El grafo final es:

`router -> experto -> human_feedback -> update_reputation -> END`

---

## Ejecucion interactiva

**Funcion:** `run_interactive`

1) Ejecuta el grafo hasta el punto de interrupcion.
2) Muestra la respuesta (si no se imprimio por streaming).
3) Pide feedback humano.
4) Reanuda el grafo y muestra resumen final.

---

## Ejecucion no interactiva

**Funcion:** `run_demo`

- Permite pasar el feedback desde linea de comandos.
- Util para pruebas rapidas.

---

## main() y parametros

Argumentos:

- `--log`: ruta del archivo de log.
- `--user`: nombre del usuario.
- `--question`: pregunta directa, sin input interactivo.
- `--feedback`: feedback predefinido (demo).
- `--demo`: modo no interactivo.
- `--save-graph` / `--save-graph-only`: guardar el grafo en PNG.

---

## Por que se eligio este enfoque

- **TXT como base de datos simple**: es facil de auditar y portar.
- **LLM en el router**: permite decisiones mas profundas que solo palabras clave.
- **Fallback determinista**: evita fallos si el LLM no devuelve JSON valido.
- **Streaming**: mejora la UX mostrando la respuesta en tiempo real.
- **Trazabilidad externa**: si necesitas auditoria, activas `TRACE_URL`.

---

## Resultado esperado

Cuando preguntas algo nuevo:

- El router decide el experto con base en historial + LLM.
- Se explica con un mensaje (ej: "Has sido redirigido a...").
- La respuesta sale en streaming.
- Tu feedback queda guardado y afecta futuras preguntas similares.

---

## Glosario rapido

- **MITL**: Man-In-The-Loop (humano interviene con feedback).
- **Routing**: proceso de elegir al mejor experto.
- **Feedback similar**: preguntas pasadas con tokens parecidos.
- **Trazabilidad**: envio de eventos a una API externa.

