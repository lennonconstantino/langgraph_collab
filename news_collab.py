import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

class NewsAgentState(TypedDict):
    original_news: str
    plan: List[Dict[str, str]] | None
    current_task_idx: int
    intermediate_results: Dict[str, str]
    final_response: str | None
    error: str | None
    current_task_description: str | None
    current_specialist_type: str | None
    current_task_id: str | None
    specialist_result: str | None

def planner_node(state: NewsAgentState) -> Dict[str, Any] :
    news = state["original_news"]
    plan = [
        {
            "task_id": "summarize_news",
            "specialist_type": "summarizer",
            "description": f"Resuma a seguinte notícia de forma clara e objetiva: {news}"
        },
        {
            "task_id": "analyze_news",
            "specialist_type": "analyst",
            "description": "Analise o resumo da notícia, destacando pontos importantes, possíveis vieses e impacto social/polítco"
        },
        {
            "task_id": "suggest_questions",
            "specialist_type": "questioner",
            "description": "Sugira perguntas para reflexão ou debate baseadas na análise da notícia. Use o resultado da tarefa"
        }
    ]
    return {
        "plan": plan,
        "current_task_idx": 0,
        "intermediate_results": {},
        "error": None,
        "specialist_result": None
    }

def prepare_next_task_node(state: NewsAgentState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)
    if not plan or current_task_idx >= len(plan):
        return {"current_task_id": None, "current_task _description": None, "current_specialist_type": None}
    current_task = plan[current_task_idx]
    return {
        "current_task_id": current_task["task_id"],
        "current_task_description": current_task["description"],
        "current_specialist_type": current_task["specialist_type"],
    }

def summarizer_node(state: NewsAgentState) -> Dict[str, str]:
    desc = state.get("current_task_description", "")
    news = state.get("original_news", "")

    # Aqui, apenas um resumo simples (poderia ser LLM
    if len(news) > 400:
        resumo = news[:400] + "... (resumo truncado)"
    else:
        resumo = news
    return {"specialist_result": f"Resumo: {resumo}"}

def analyst_node(state: NewsAgentState) -> Dict[str, str]:
    prev_results = state.get("intermediate_results", {})
    resumo = prev_results.get("summarize_news", "sem resumo")
    # Análise simples
    analise = f"Análise: Pontos importantes do resumo: {resumo[:100]}... (análise simplificada)"
    analise += "\nPossíveis vieses: Não identificado (exemplo). \nImpacto: A notícia pode impactar a opinião pública."
    return {"specialist_result": analise}

def questioner_node(state: NewsAgentState) -> Dict[str, str]:
    prev_results = state.get("intermediate_results", {})
    analise = prev_results.get("analyze_news", "sem análise")
    perguntas = [
        "Quais são as fontes dessa notícia?",
        "Como essa notícia pode afetar diferentes grupos sociais?",
        "Há outros pontos de vista sobre o tema?"
    ]
    return {"specialist_result": "Perguntas para reflexão: " + ", ".join(perguntas)}

def collect_result_and_advance_node(state: NewsAgentState) -> Dict[str, Any]:
    current_task_id = state.get("current_task_id")
    specialist_output = state.get("specialist_result", "Nenhum resultado do especialista encontrado no estado.")
    updated_intermediate_results = state.get("intermediate_results", {}).copy()
    if current_task_id:
        updated_intermediate_results[current_task_id] = specialist_output
    new_idx = state.get("current_task_idx", 0) + 1
    return {
        "intermediate_results": updated_intermediate_results,
        "current_task_idx": new_idx,
        "specialist_result": None
    }

def synthesis_node(state: NewsAgentState) -> Dict[str, str | None]:
    original_news = state["original_news"]
    intermediate_results = state.get("intermediate_results", {})
    resposta = f"Notícia original: {original_news}\n"
    for task_id, result in intermediate_results.items():
        resposta += f"- {task_id}: {result}\n"
    return {"final_response": resposta, "error": None}

def should_execute_task_or_synthesize(state: NewsAgentState) -> str:
    if state.get("error"):
        return "error_handler"

    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)
    if current_task_idx < len(plan):
        return "prepare_next_task"
    else:
        return "synthesize_response"

def specialist_router_node(state: NewsAgentState) -> str:
    specialist_type = state.get("current_specialist_type")
    if specialist_type == "summarizer":
        return "summarizer"
    elif specialist_type == "analyst":
        return "analyst"
    elif specialist_type == "questioner":
        return "questioner"
    else:
        return "error_handler"

def error_node(state: NewsAgentState) -> Dict[str, str | None]:
    error_message = state.get("error", "Erro desconhecido no workflow")
    return {"final_response": f"Ocorreu um erro: {error_message}"}

# Construção do workflow
workflow_builder = StateGraph(NewsAgentState)
workflow_builder.add_node("planner", planner_node)
workflow_builder.add_node("prepare_next_task", prepare_next_task_node)
workflow_builder.add_node("summarizer", summarizer_node)
workflow_builder.add_node("analyst", analyst_node)
workflow_builder.add_node("questioner", questioner_node)
workflow_builder.add_node("collect_and_advance", collect_result_and_advance_node)
workflow_builder.add_node("synthesize_response", synthesis_node)
workflow_builder.add_node("error_handler", error_node)

workflow_builder.set_entry_point("planner")
workflow_builder.add_conditional_edges(
    "planner", should_execute_task_or_synthesize,
    {
        "prepare_next_task": "prepare_next_task",
        "synthesize_response": "synthesize_response",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_conditional_edges(
    "prepare_next_task", specialist_router_node,
    {
        "summarizer": "summarizer",
        "analyst": "analyst",
        "questioner": "questioner",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_edge("summarizer", "collect_and_advance")
workflow_builder.add_edge("analyst", "collect_and_advance")
workflow_builder.add_edge("questioner", "collect_and_advance")
workflow_builder.add_conditional_edges(
    "collect_and_advance",
    should_execute_task_or_synthesize,
    {
        "prepare_next_task": "prepare_next_task",
        "synthesize_response": "synthesize_response",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_edge("synthesize_response", END)
workflow_builder.add_edge("error_handler", END)

news_workflow = workflow_builder.compile()
