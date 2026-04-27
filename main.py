import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

LOG_FILE = "feedback_log.txt"
AGENTS = {
    "historia": "Especialista en Historia",
    "literatura": "Especialista en Literatura",
    "biologia": "Especialista en Biologia",
}
VALID_AGENTS = set(AGENTS.keys())


class GraphState(BaseModel):
    user_question: str = Field(..., description="Pregunta del usuario")
    selected_agent: Optional[str] = None
    expert_response: Optional[str] = None
    response_streamed: bool = False
    feedback_text: Optional[str] = None
    feedback_score: Optional[int] = None
    routing_message: Optional[str] = None


def stream_chat(llm, messages) -> str:
    full_text = ""
    for chunk in llm.stream(messages):
        token = getattr(chunk, "content", "") or ""
        if token:
            print(token, end="", flush=True)
            full_text += token
    print("")
    return full_text


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: Dict[str, Any], log_path: str) -> None:
    payload = {"timestamp": now_utc_iso(), **event}
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

def load_events(log_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(log_path):
        return []
    events = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def build_feedback_stats(question: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_counts = {agent: {"positivo": 0, "negativo": 0, "neutral": 0} for agent in VALID_AGENTS}
    similar_counts = {agent: {"positivo": 0, "negativo": 0, "neutral": 0} for agent in VALID_AGENTS}
    similar_examples = []

    for e in events:
        if e.get("event_type") != "feedback":
            continue
        agent = e.get("agent")
        if agent not in VALID_AGENTS:
            continue
        sentiment = e.get("sentiment", "neutral")
        if sentiment not in {"positivo", "negativo", "neutral"}:
            sentiment = "neutral"
        all_counts[agent][sentiment] += 1

        score_sim = similarity(question, e.get("question", ""))
        if score_sim >= 0.45:
            similar_counts[agent][sentiment] += 1
            similar_examples.append(
                {
                    "question": e.get("question"),
                    "agent": agent,
                    "sentiment": sentiment,
                    "score": round(score_sim, 2),
                }
            )

    similar_examples = sorted(similar_examples, key=lambda x: x["score"], reverse=True)[:5]
    return {
        "all_counts": all_counts,
        "similar_counts": similar_counts,
        "similar_examples": similar_examples,
        "total_events": len(events),
    }


def fallback_agent_from_stats(preferred: Optional[str], stats: Dict[str, Any]) -> str:
    preferred_agent = preferred if preferred in VALID_AGENTS else "historia"
    similar = stats.get("similar_counts", {})
    if preferred_agent in similar and similar[preferred_agent]["negativo"] > 0:
        ranked = sorted(
            VALID_AGENTS,
            key=lambda a: (
                similar.get(a, {}).get("positivo", 0),
                -similar.get(a, {}).get("negativo", 0),
            ),
            reverse=True,
        )
        return ranked[0]
    return preferred_agent


def choose_agent_with_llm(question: str, preferred: Optional[str], log_path: str, llm) -> Dict[str, Any]:
    events = load_events(log_path)
    stats = build_feedback_stats(question, events)
    preferred_agent = preferred if preferred in VALID_AGENTS else "historia"

    system_text = (
        "Eres un router experto. Debes elegir el mejor agente entre: "
        f"{', '.join(VALID_AGENTS)}. Analiza la pregunta, el experto sugerido, "
        "y el feedback historico (global y similar). Responde SOLO JSON con "
        "las claves: agent, reason."
    )
    human_text = json.dumps(
        {
            "question": question,
            "preferred": preferred_agent,
            "feedback_stats": stats,
        },
        ensure_ascii=True,
    )

    try:
        result = llm.invoke([SystemMessage(content=system_text), HumanMessage(content=human_text)])
        payload = json.loads(str(result.content).strip())
        agent = payload.get("agent")
        reason = payload.get("reason", "Decision por analisis de feedback.")
        if agent in VALID_AGENTS:
            return {"agent": agent, "reason": reason}
    except Exception:
        pass

    fallback = fallback_agent_from_stats(preferred_agent, stats)
    reason = "Decision por fallback basado en feedback similar."
    return {"agent": fallback, "reason": reason}


def tokenize(text: str) -> set:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in cleaned.split() if token}


def similarity(a: str, b: str) -> float:
    a_set = tokenize(a)
    b_set = tokenize(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(len(a_set | b_set), 1)


def interpret_sentiment(feedback_text: str, score: Optional[int]) -> str:
    text = (feedback_text or "").lower()
    if "negativo" in text:
        return "negativo"
    if "positivo" in text:
        return "positivo"
    if score is None:
        return "neutral"
    if score <= 2:
        return "negativo"
    if score >= 4:
        return "positivo"
    return "neutral"



def classify_question(question: str) -> Optional[str]:
    text = question.lower()
    if any(word in text for word in ["poema", "novela", "cuento", "epica", "mito", "simbolo", "metafora"]):
        return "literatura"
    if any(word in text for word in ["origen", "cronologia", "imperio", "culto", "civilizacion", "antiguedad"]):
        return "historia"
    if any(word in text for word in ["anatomia", "especie", "evolucion", "ecosistema", "taxonomia", "biologia"]):
        return "biologia"
    return None


def build_graph(log_path: str, llm, user_id: str) -> Any:

    def router_node(state: GraphState) -> Dict[str, Any]:
        preferred = classify_question(state.user_question)
        decision = choose_agent_with_llm(state.user_question, preferred, log_path, llm)
        log_event(
            {
                "event_type": "routing",
                "user": user_id,
                "question": state.user_question,
                "preferred": preferred,
                "selected_agent": decision["agent"],
                "reason": decision["reason"],
            },
            log_path,
        )
        return {"selected_agent": decision["agent"], "routing_message": decision["reason"]}

    def expert_node_factory(role_key: str):
        prompt_map = {
            "historia": (
                "Eres un historiador especializado en mitologia comparada. "
                "Analiza la criatura mitologica desde su origen temporal, "
                "contexto cultural, versiones regionales, y evidencia textual. "
                "Incluye referencias a civilizaciones relevantes, cronologia "
                "aproximada, y como evoluciono el mito. Responde en espanol. "
                "Formato Markdown con secciones: Titulo, Resumen, Contexto, "
                "Fuentes, Interpretacion, Recomendacion."
            ),
            "literatura": (
                "Eres un experto en literatura clasica y comparada. "
                "Analiza la criatura mitologica desde sus representaciones "
                "literarias, simbolismos, arquetipos narrativos y figuras retoricas. "
                "Menciona obras, generos y motivos recurrentes. Responde en espanol. "
                "Formato Markdown con secciones: Titulo, Resumen, Lectura literaria, "
                "Simbolismo, Obras relacionadas, Recomendacion."
            ),
            "biologia": (
                "Eres un biologo especializado en zoologia y etologia. "
                "Interpreta la criatura mitologica desde analogos biologicos, "
                "posibles inspiraciones en fauna real, adaptaciones hipoteticas, "
                "y coherencia con ecosistemas. Responde en espanol. "
                "Formato Markdown con secciones: Titulo, Resumen, Analogos biologicos, "
                "Adaptaciones, Ecosistema, Recomendacion."
            ),
        }
        strict_suffix = " Responde directo y no repitas la pregunta ni el prompt."
        system_text = prompt_map[role_key] + strict_suffix

        def expert_node(state: GraphState) -> Dict[str, Any]:
            human_text = (
                "Pregunta del usuario (no la repitas): "
                f"{state.user_question}"
            )
            print("\n=== Respuesta del experto (streaming) ===")
            content = stream_chat(llm, [SystemMessage(content=system_text), HumanMessage(content=human_text)])
            return {"expert_response": content, "response_streamed": True}

        return expert_node

    def human_feedback_node(state: GraphState) -> Dict[str, Any]:
        prompt = (
            "Califica la respuesta del experto (1-5) o escribe Muy negativo/Muy positivo."
        )
        payload = {
            "prompt": prompt,
            "agent": state.selected_agent,
            "response": state.expert_response,
        }
        feedback = interrupt(payload)
        score = parse_feedback(feedback)
        sentiment = interpret_sentiment(str(feedback), score)
        log_event(
            {
                "event_type": "feedback",
                "user": user_id,
                "question": state.user_question,
                "agent": state.selected_agent,
                "feedback_text": str(feedback),
                "feedback_score": score,
                "sentiment": sentiment,
            },
            log_path,
        )
        return {"feedback_text": str(feedback), "feedback_score": score}

    def update_reputation_node(state: GraphState) -> Dict[str, Any]:
        return {}

    def route_to_expert(state: GraphState) -> str:
        if state.selected_agent in VALID_AGENTS:
            return state.selected_agent
        return "historia"

    builder = StateGraph(GraphState)
    builder.add_node("router", router_node)
    builder.add_node("historia", expert_node_factory("historia"))
    builder.add_node("literatura", expert_node_factory("literatura"))
    builder.add_node("biologia", expert_node_factory("biologia"))
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("update_reputation", update_reputation_node)

    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        route_to_expert,
        {
            "historia": "historia",
            "literatura": "literatura",
            "biologia": "biologia",
        },
    )
    builder.add_edge("historia", "human_feedback")
    builder.add_edge("literatura", "human_feedback")
    builder.add_edge("biologia", "human_feedback")
    builder.add_edge("human_feedback", "update_reputation")
    builder.add_edge("update_reputation", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def get_llm() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)


def print_final_summary(state: Dict[str, Any]) -> None:
    summary = {
        "selected_agent": state.get("selected_agent"),
        "feedback_score": state.get("feedback_score"),
    }
    print("\n--- Resumen final ---")
    print(json.dumps(summary, indent=2))


def save_graph_artifacts(graph, output_path: str) -> None:
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as handle:
            handle.write(png_bytes)
        print(f"Grafo guardado en: {output_path}")
    except Exception as exc:
        mermaid_path = os.path.splitext(output_path)[0] + ".mmd"
        mermaid_text = graph.get_graph().draw_mermaid()
        with open(mermaid_path, "w", encoding="utf-8") as handle:
            handle.write(mermaid_text)
        print(f"No se pudo generar PNG ({exc}). Mermaid guardado en: {mermaid_path}")


def run_interactive(graph, initial_state: GraphState, feedback: Optional[str]) -> None:
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    # 1. Primera ejecución: llegará hasta el interrupt
    graph.invoke(initial_state, config=config)

    # 2. Obtenemos el estado actual para ver qué escribió el experto
    state_snapshot = graph.get_state(config)

    # Si el estado está en un punto de interrupción (interrupt)
    if state_snapshot.next:
        routing_message = state_snapshot.values.get("routing_message")
        if routing_message:
            print(f"\n{routing_message}")
        expert_res = state_snapshot.values.get("expert_response", "Sin respuesta")
        if not state_snapshot.values.get("response_streamed"):
            print("\n=== Respuesta del experto ===")
            print(expert_res)

        print("\n=== Feedback humano ===")
        if feedback is None:
            feedback = input("Califica (1-5 o Muy negativo/Muy positivo): ").strip()

        # 3. Reanudamos la ejecución pasando el valor al interrupt
        resumed = graph.invoke(Command(resume=feedback), config=config)

        # El resultado final suele estar en el último snapshot después de reanudar
        final_state = graph.get_state(config)
        print_final_summary(final_state.values)
    else:
        print_final_summary(state_snapshot.values)


def run_demo(graph, initial_state: GraphState, feedback: str) -> None:
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config=config)
    if isinstance(result, dict) and "__interrupt__" in result:
        resumed = graph.invoke(Command(resume=feedback), config=config)
        print_final_summary(resumed if isinstance(resumed, dict) else {})
        return
    print_final_summary(result if isinstance(result, dict) else {})


def parse_feedback(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 5 else None
    if isinstance(value, str):
        text = value.strip().lower()
        if text.isdigit():
            num = int(text)
            return num if 1 <= num <= 5 else None
        if "muy positivo" in text or text in {"positivo", "approve", "approved", "ok", "yes"}:
            return 5
        if "muy negativo" in text or text in {"negativo", "reject", "rejected", "no"}:
            return 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph reputation router demo")
    parser.add_argument("--log", default=LOG_FILE, help="Ruta del log de feedback")
    parser.add_argument("--user", default="anon", help="Usuario para auditoria")
    parser.add_argument("--feedback", default=None, help="Feedback to resume the graph")
    parser.add_argument("--demo", action="store_true", help="Run a non-interactive demo")
    parser.add_argument("--question", default=None, help="Pregunta inicial (omite input interactivo)")
    parser.add_argument("--save-graph", default=None, help="Guarda el grafo como imagen PNG")
    parser.add_argument("--save-graph-only", default=None, help="Guarda el grafo y termina sin pedir pregunta")
    args = parser.parse_args()

    llm = get_llm()
    graph = build_graph(args.log, llm, args.user)

    if args.save_graph_only:
        save_graph_artifacts(graph, args.save_graph_only)
        return

    question = args.question or input("\nHaz una pregunta sobre animales mitologicos.\nPregunta: ").strip()
    state = GraphState(user_question=question)

    if args.demo:
        feedback = args.feedback or "5"
        run_demo(graph, state, feedback)
    else:
        run_interactive(graph, state, args.feedback)

    if args.save_graph:
        save_graph_artifacts(graph, args.save_graph)


if __name__ == "__main__":
    main()
