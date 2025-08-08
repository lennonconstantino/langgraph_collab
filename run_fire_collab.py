from fire_collab import collaborative_workflow

def main():
    # Estado inicial com a consulta do usuário
    initial_state = {
        "original_query": "Gere um relatório completo e atualizado sobre a empresa Google, incluindo informações sobre seus produtos, serviços, finanças, mercado, concorrentes, tendências e perspectivas. ",
        "plan": None,
        "intermediate_results": {},
        "current_task_idx": 0,
        "final_response": None,
        "error": None,
        "current_task_description": None,
        "current_specialist_type": None,
        "current_task_id": None,
        "specialist_result": None
    }

    # executa workflow
    result = collaborative_workflow.invoke(initial_state)

    # Acessa a resposta final
    print(result["final_response"])

if __name__ == "__main__":
    main()