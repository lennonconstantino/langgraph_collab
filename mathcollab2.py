import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from models import models

# Estado compartilhado
class SimpleAgentState(TypedDict):
    original_query: str
    plan: List[Dict[str, str]] | None
    current_task_idx: int
    intermediate_results: Dict[str, str]
    final_response: str | None
    error: str | None
    current_task_description: str | None
    current_specialist_type: str | None
    current_task_id: str | None
    specialist_result: str | None

# planejador simples divide em cálculo e explicacao
def planner_node(state: SimpleAgentState) -> Dict[str, Any]:
    query = state["original_query"]
    plan = [
        {
            "task_id": "do_math",
            "specialist_type": "mathematician",
            "description": f"Resolva a seguinte expressão matemática: {query}"
        },
        {
            "task_id": "explain_result",
            "specialist_type": "writer",
            "description":f"Explique me linguagem simples o resultado do cálculo feito na tarefa 'do_math'."
        }
    ]
    return {
        "plan": plan,
        "current_task_idx": 0,
        "intermediate_results": {},
        "error": None,
        "specialist_type": None
    }

def prepare_next_task_node(state: SimpleAgentState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)
    if not plan or current_task_idx >= len(plan):
        return {"current_task_id": None,
                "current_task_description": None,
                "current_specialist_type": None}
    current_task = plan[current_task_idx]
    return {
        "current_task_id": current_task["task_id"],
        "current_task_description": current_task["description"],
        "current_specialist_type": current_task["specialist_type"],
    }

def mathematician_node(state: SimpleAgentState) -> Dict[str, str]:
    desc = state.get("current_task_description", "")
    prompt = f"Você é um especialista em matemática. Resolva a expressão abaixo e forneça apenas o resultado numérico final.\nExemplo {desc}\n"
    try:
        response = models["gpt_4o"].invoke([HumanMessage(content=prompt)])
        return {"specialist_result": response.content.strip()}
    except Exception as e:
        return {"specialist_result": f"Erro ao calcular: {e}"}

def writer_node(state: SimpleAgentState) -> Dict[str, str]:
    prev_results = state.get("intermediate_results", {})
    math_result = prev_results.get("do_math", "sem resultado")
    original_query = state.get("original_query", "")
    prompt = (
        f"Explique detalhadamente, passo a passo, como resolver a expressão matemática abaixo, considerando a ordem das operações\n"
        f"Expressão: {original_query}\n"
        f"Resultado final: {math_result}\n"
    )
    try:
        response = models["gpt_4o"].invoke([HumanMessage(content=prompt)])
        return {"specialist_result": response.content.strip()}
    except Exception as e:
        return {"specialist_result": f"Erro na explicação: {e}"}

def collect_result_and_advance_node(state: SimpleAgentState) -> Dict[str, str]:
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

def synthesis_node(state: SimpleAgentState) -> Dict[str, str | None]:
    original_query = state["original_query"]
    intermediate_results = state.get("intermediate_results", {})
    response = f"Pergunta original: {original_query}\n"
    for task_id, result in intermediate_results.items():
        response += f"-{task_id}: {result}\n"
    return {"final_response": response, "error": None}

def should_execute_task_or_synthesize(state: SimpleAgentState) -> str:
    if state.get("error"):
        return "error_handler"

    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)
    if current_task_idx < len(plan):
        return "prepare_next_task"
    else:
        return "synthesize_response"

def specialist_router_node(state: SimpleAgentState) -> str:
    specialist_type = state.get("current_specialist_type")
    if specialist_type == "mathematician":
        return "mathematician"
    elif specialist_type == "writer":
        return "writer"
    else:
        return "error_handler"

def error_node(state: SimpleAgentState) -> Dict[str, str | None]:
    error_message = state.get("error", "Erro desconhecido no workflow")
    return {"final_response": f"Ocorreu um erro: {error_message}"}

# Construção do workflow
workflow_builder = StateGraph(SimpleAgentState)
workflow_builder.add_node("planner", planner_node)
workflow_builder.add_node("prepare_next_task", prepare_next_task_node)
workflow_builder.add_node("mathematician", mathematician_node)
workflow_builder.add_node("writer", writer_node)
workflow_builder.add_node("collect_and_advance", collect_result_and_advance_node)
workflow_builder.add_node("synthesize_response", synthesis_node)
workflow_builder.add_node("error_handler", error_node)

workflow_builder.set_entry_point("planner")

workflow_builder. add_conditional_edges(
    "planner",
    should_execute_task_or_synthesize, {
        "prepare_next_task": "prepare_next_task",
        "synthesize_response": "synthesize_response",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_conditional_edges(
    "prepare_next_task",
    specialist_router_node, {
        "mathematician": "mathematician",
        "writer": "writer",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_edge("mathematician", "collect_and_advance")
workflow_builder.add_edge("writer", "collect_and_advance")

workflow_builder. add_conditional_edges(
    "collect_and_advance",
    should_execute_task_or_synthesize,{
        "prepare_next_task": "prepare_next_task",
        "synthesize_response": "synthesize_response",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_edge("synthesize_response", END)
workflow_builder.add_edge("error_handler", END)

simple_workflow = workflow_builder. compile()
