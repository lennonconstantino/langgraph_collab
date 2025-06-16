from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

_ = load_dotenv() # forcar a execucao

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_PROVIDER_MAP = {
    "openai": ChatOpenAI,
    "google": ChatGoogleGenerativeAI
}

MODEL_CONFIGS = [
    {
        "key_name": "gemini_2.5_flash",
        "provider" : "google",
        "model_name": "gemini-2.5-flash-preview-04-17",
        "temprature": 1.0,
    },
    {
        "key_name": "o4",
        "provider": "openai",
        "model_name": "o4-mini-2025-04-16",
    },
    {
        "key_name": "gpt_4o",
        "provider": "openai",
        "model_name": "gpt-4o-2024-08-06",
    },
]

def _create_chat_model(model_name: str, provider: str, temperature: float | None = None):
    if provider not in _PROVIDER_MAP:
        raise ValueError(f"Provedor nao suportado: {provider}. Provedores suportados sao: {list(_PROVIDER_MAP.keys())}")

    model_class = _PROVIDER_MAP[provider]
    params = {"model": model_name}
    if temperature is not None:
        params["temperature"] = temperature

    return model_class(**params)

models = {}

for config in MODEL_CONFIGS:
    models[config["key_name"]] = _create_chat_model(
        model_name=config["model_name"],
        provider=config["provider"],
        temperature=config.get("temperature")
    )

if __name__ == "__main__":
    print()