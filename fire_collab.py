import os
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import Tool

from models import models
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from firecrawl import FirecrawlApp

import logging

class CollaborativeAgentState(TypedDict) :
    original_query: str
    plan: List[Dict[str, str]] | None
    intermediate_results: Dict[str, str]
    current_task_idx: int
    final_response: str | None
    error: str | None

    current_task_description: str | None
    current_specialist_type: str | None
    current_task_id: str | None
    specialist_result: str | None

# Firecrawl tools
def create_firecrawl_search_tool():
    def search_func (query: str) -> str:
        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        result = app.search (
            query=query,
            Limit=10,
            scrape_options={"formats": ["markdown"]}
        )
        return "\n\n".join([
            f"**{res['title']}**\n{res['url']}\n{res['markdown'][:200]}..."
            for res in result.data
        ])

    return Tool(
        name="Web Search",
        func=search_func,
        description="Realiza buscas na web usando a API do Firecrawl"
    )

def create_firecrawl_scrape_tool():
    def scrape_func(url: str) -> str:
        loader = FireCrawlLoader(
            api_key=os.getenv("'FIRECRAWL_API_KEY"),
            url=url,
            mode="scrape",
            params={"formats": ["markdown"]}
        )
        docs = loader.load()
        return docs[0].page_content if docs else "Nenhum conteúdo web pôde ser obtido."

    return Tool(
        name="scrape_website",
        func=scrape_func,
        description="Extrai o conteúdo de um site usando a API do Firecrawl"
    )

def planner_node(state: CollaborativeAgentState) -> Dict[str, Any]:
    """
    Recebe a consulta original e cria um plano de sub-tarefas,
    atribuindo um tipo de especialista para cada uma.
    """
    query = state["original_query"]

    system_message_planner = SystemMessage(
        content=f"""
        Você é um planejador especialista em um sistema multi-agente.
        Sua função é decompor uma consulta complexa do usuário em uma sequência de sub-tarefas executáveis.
        Para cada sub-tarefa, você deve especificar:
            1. task_id: Um identificador único para a tarefa (e.g., "task_1", "task_2").
            2. "specialist_type*: O tipo de especialista necessário. Tipos válidos são: "researcher", "writer".
            3. "description: Uma descrição clara e concisa da sub-tarefa para o especialista.
        A ordem das tarefas no plano é importante.
        Por exemplo, uma tarefa de 'writer' que depende de pesquisa deve vir depois da tarefa de 'researcher"
        Responda APENAS com um objeto JSON contendo uma lista chamada "plan" com as sub-tarefas.
        
        Exemplo de Consulta: "Escreva um breve resumo sobre os avanços recentes em carros autônomos. "
        Exemplo de Resposta JSON:
        {{
            "plan": [
                {{
                    "task_id": "research_autonomous_cars",
                    "specialist_type": "researcher",
                    "description": "Pesquise os avanços mais recentes e significativos na tecnologia de carros autônomos
                }},
                {{
                    "task_id": "write_summary_autonomous_cars",
                    "specialist_type": "writer",
                    "description": "Com base na pesquisa sobre carros autônomos (especialmente os resultados de 'research_autonomous_cars')
                }},
            ]        
        }} 
        """
    )
    human_message = HumanMessage(content=query)

    try:
        response = models["gpt_4o"].invoke([system_message_planner, human_message])
        plan_json = json.loads(response.content)
        plan = plan_json.get("plan", [])
        return {
            "plan": plan,
            "current_task_idx": 0,
            "intermediate_results": {},
            "error": None,
            "specialist_result": None
        }
    except Exception as e:
        return {"error": f"Erro ao gerar plano: {str(e)}", "plan": [], "specialist_result": None}

def prepare_next_task_node(state: CollaborativeAgentState) -> Dict[str, Any]:
    """
    Prepara os detalhes da tarefa atual para serem passados ao roteador de especialistas.
    """
    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)

    if not plan or current_task_idx >= len(plan):
        return {"current_task_id": None, "current_task_description": None, "current_specialist_type": None}

    current_task = plan[current_task_idx]
    return {
        "current_task_id": current_task["task_id"],
        "current_task_description": current_task["description"],
        "current_specialist_type": current_task["specialist_type"],
    }

def researcher_node(state: CollaborativeAgentState) -> Dict[str, str]:
    """Executa uma sub-tarefa de pesquisa."""
    task_description = state.get("current_task_description")
    original_query_for_llm = state.get("original_query")  # Pode ser útil para dar mais contexto ao LLM final

    if not task_description:
        return {"specialist_result": "Erro: Descrição da tarefa não encontrada ou vazia."}

    search_tool = create_firecrawl_search_tool()
    scraped_content_for_llm = "Nenhum conteúdo web pôde ser obtido."

    try:
        search_results = search_tool.run(task_description)

        search_results_list = []
        if isinstance(search_results, str):
            try:
                loaded_json = json.loads(search_results)
                if isinstance(loaded_json, list):
                    search_results_list = loaded_json
                elif isinstance(loaded_json, dict) and "data" in loaded_json and isinstance(loaded_json["data"], list):
                    search_results_list = loaded_json["data"]
                elif isinstance(loaded_json, dict) and loaded_json.get('success') is True and "data" in loaded_json and isinstance(loaded_json["data"], list):
                    search_results_list = loaded_json["data"]
            except json.JSONDecodeError:
                search_results_list = []
        elif isinstance(search_results, list):
            search_results_list = search_results
        elif isinstance(search_results, dict):
            if "data" in search_results and isinstance(search_results["data"], list):
                search_results_list = search_results["data"]
            elif search_results.get('success') is True and "data" in search_results and isinstance(search_results["data"], list):
                search_results_list = search_results["data"]
            else:
                search_results_list = []
        else:
            search_results_list = []

        if not search_results_list:
            logging.error("Nenhuma URL utilizável encontrada após processar resultados da busca Firecrawl.")
        else:
            first_url = None
            for item in search_results_list:
                if isinstance(item, dict) and item.get("url"):
                    first_url = item.get("ur]")
                    break

            if first_url:
                scrape_tool = create_firecrawl_scrape_tool()
                scraped_data = scrape_tool.run(first_url)

                if isinstance(scraped_data, dict) and "markdown" in scraped_data:
                    scraped_content_for_llm = scraped_data["markdown"]
                elif isinstance(scraped_data, str):
                    scraped_content_for_llm = scraped_data
                else:
                    logging.error(f"Não foi possível extrair o conteúdo markdown da URL. Retorno: {scraped_data}")
            else:
                logging.error("Nenhuma URL válida encontrada dentro dos resultados da busca processados.")

    except Exception as e:
        scraped_content_for_llm = f"Erro ao tentar obter conteúdo da web: {str(e)}. Usarei meu conhecimento geral."

    system_message_researcher = SystemMessage(
        content="""
        Você é um agente de pesquisa especialista.
        Sua tarefa é responder à pergunta/descrição fornecida.
        Se um conteúdo da web foi fornecido, baseie sua resposta PRIMARIAMENTE nesse conteúdo.
        Se não, use seu conhecimento geral.
        Forneça uma resposta concisa e informativa com os principais achados.
        """
    )

    llm_prompt_content = f"Descrição da Tarefa de Pesquisa {task_description}\n\n"
    if scraped_content_for_llm and "Erro ao tentar obter conteúdo da web" not in scraped_content_for_llm and "Nenhum conteúdo web pode ser obtido" not in scraped_content_for_llm:
        llm_prompt_content += f"Contexto Obtido da Web (use isso como fonte principal): \n{scraped_content_for_llm}\n\n"
    else:
        llm_prompt_content += "Não foi possível obter conteúdo da web para esta tarefa ou ocorreu um erro. Por favor, responda usando seu conhecimento geral.\n\n"
    llm_prompt_content += "Com base no contexto acima (se disponível) e na descrição da tarefa, forneça sua pesquisa:"

    human_message = HumanMessage(content=llm_prompt_content)

    try:
        response = models["gpt_4o"].invoke([system_message_researcher, human_message])
        return {"specialist_result": response.content}
    except Exception as e:
        logging.error(f"Erro no LLM do pesquisador: {e}")
        return {"specialist_result": f"Erro no LLM da pesquisa: {str(e)}"}

def writer_node(state: CollaborativeAgentState) -> Dict[str, str]:
    """Executa uma sub-tarefa de escrita, utilizando resultados anteriores se disponíveis."""
    task_description = state.get("current_task_description")
    intermediate_results = state.get("intermediate_results", {})

    if not task_description:
        logging.error("Erro: Descrição da tarefa não encontrada ou vazia para o escritor.")
        return {"specialist_result": "Erro: Descrição da tarefa de escrita não encontrada ou vazia. "}

    context = "Contexto de tarefas anteriores:\n"
    if not intermediate_results:
        context += "Nenhum resultado de tarefas anteriores disponível. \n"
    for task_id, result in intermediate_results.items():
        context += f"'- Resultado da tarefa '{task_id}': {result}\n\n"

    system_message_writer = SystemMessage(
        content="Você é um agente escritor especialista. Sua tarefa é redigir um texto claro, coeso e bem estruturado com base na descrição da tarefa e no contexto fornecido (resultados de tarefas anteriores, se houver). Siga as instruções da descrição da tarefa"
    )
    prompt_content = f"{context}Descrição da Tarefa Atual: \n{task_description}"
    human_message = HumanMessage(content=prompt_content)

    try:
        response = models["gpt_4o"].invoke([system_message_writer, human_message])
        return {"specialist_result": response.content}
    except Exception as e:
        logging.error(f"Erro no escritor: {e}")
        return {"specialist_result": f"Erro na escrita: {str(e)}"}

def collect_result_and_advance_node(state: CollaborativeAgentState) -> Dict[str, Any] :
    """
    Coleta o resultado do especialista, armazena e incrementa o índice da tarefa.
    Este nó precisa ser chamado com o resultado do especialista.
    Para simplificar, vamos fazer com que os nós especialistas retornem um dict com "specialist_result" e este nó será chamado após eles, pegando esse valor do estado.
    Alternativa: Modificar a forma como os nós são chamados ou usar partial para
    passar o resultado.

    Para este exemplo, vamos assumir que o `specialist_result` é colocado no estado pelo nó especialista
    e este nó apenas o processa.
    """
    current_task_id = state.get("current_task_id")
    specialist_output = state.get("specialist_result", "Nenhum resultado do especialista encontrado no estado.")
    updated_intermediate_results = state.get("intermediate_results", {}).copy()
    if current_task_id:
        updated_intermediate_results[current_task_id] = specialist_output

    new_idx = state.get("current_task_idx", 0) + 1

    return {
        "intermediate_results": updated_intermediate_results,
        "current_task_idx": new_idx,
        "specialist_ result": None  # Limpa para a próxima iteração
    }

def synthesis_node(state: CollaborativeAgentState) -> Dict[str, str | None]:
    """Sintetiza os resultados intermediários em uma resposta final."""

    original_query = state["original_query"]
    intermediate_results = state.get("intermediate_results", {})

    context = "Consulta Original do Usuário:\n" + original_query + "\n\nResultados das Sub-tarefas Executadas: \n"
    if not intermediate_results:
        context += "Nenhum resultado de sub-tarefas disponível. \n"
    for task_id, result in intermediate_results.items():
        context += f"- Resultado da tarefa '{task_id}': {result}\n\n"

    system_message_synthesis = SystemMessage(
        content="Você é um assistente de IA especialista em sintetizar informações. Sua tarefa é pegar a consulta original do usuario"
    )
    human_message = HumanMessage(content=context)
    try:
        response = models["gpt_4o"].invoke([system_message_synthesis, human_message])
        return {"final_response": response.content, "error": None}
    except Exception as e:
        logging.error(f"Erro na síntese: {e}")
        return {"final_response": None, "error": f"Erro ao sintetizar resposta: {str(e)}"}

def should_execute_task_or_synthesize(state: CollaborativeAgentState) -> str:
    """Decide se continua executando tarefas do plano ou se parte para a síntese."""
    if state.get("error"):
        logging.error(f"Erro detectado: {state['error']}. Finalizando.")
        return "error_handler"

    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)
    if current_task_idx < len(plan):
        return "prepare_next_task"
    else:
        return "synthesize_response"

def specialist_router_node(state: CollaborativeAgentState) -> str:
    """Roteia para o nó especialista correto com base no tipo de tarefa atual."""
    specialist_type = state.get("current_specialist_type")
    if specialist_type == "researcher":
        return "researcher"
    elif specialist_type == "writer":
        return "writer"
    else:
        logging.error(f"Tipo de especialista desconhecido: {specialist_type}. Lidando como erro.")
        return "error_handler"

def error_node(state: CollaborativeAgentState) -> Dict[str, str| None]:
    """Nó simples para lidar com erros e finalizar."""
    error_message = state.get("error", "Erro desconhecido no workflow.")
    logging.error(f"Erro no workflow: {error_message}")
    return {"final_response": f"Ocorreu um erro: (error_message)"}

workflow_builder = StateGraph(CollaborativeAgentState)
workflow_builder.add_node("planner", planner_node)
workflow_builder.add_node("prepare_next_task", prepare_next_task_node)
workflow_builder.add_node("researcher", researcher_node)
workflow_builder.add_node("writer", writer_node)
workflow_builder.add_node("collect_and_advance" , collect_result_and_advance_node)
workflow_builder.add_node("synthesize_response", synthesis_node)
workflow_builder.add_node("error_handler", error_node)

workflow_builder.set_entry_point ("planner")

workflow_builder.add_conditional_edges(
    "planner",
    should_execute_task_or_synthesize,
    {
        "prepare_next_task": "prepare_next_task",
        "synthesize_response": "synthesize_response",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_conditional_edges(
    "prepare_next_task",
    specialist_router_node,
    {
        "researcher": "researcher",
        "writer": "writer",
        "error_handler" : "error_handler"
    }
)

workflow_builder.add_edge("researcher", "collect_and_advance")
workflow_builder.add_edge("writer", "collect_and_advance")

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

collaborative_workflow = workflow_builder.compile()
