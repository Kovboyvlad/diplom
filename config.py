import os
import urllib.parse
from dotenv import load_dotenv

# Типы диаграмм
DIAGRAM_TYPES = {
    "Class (Классовая)": "class",
    "Sequence (Последовательности)": "sequence",
    "Component (Компонентная)": "component",
    "Activity (Деятельности)": "activity",
}

# Доступные модели: отображаемое имя → ID модели для LiteLLM/CrewAI
MODELS = {
    "ChatGPT (gpt-5)": "gpt-5",
    "Claude (claude-3-5-haiku)": "anthropic/claude-3-5-haiku-20241022",
    "Gemini (gemini-3-flash-preview)": "gemini/gemini-3-flash-preview",
}


def setup() -> str:
    """Загружает .env, настраивает прокси. Возвращает имя модели OpenAI."""
    load_dotenv()

    proxy_host = os.getenv("PROXY_HOST", "")
    proxy_port = os.getenv("PROXY_PORT", "")

    if proxy_host and proxy_port:
        user = urllib.parse.quote(os.getenv("PROXY_USER", ""))
        pwd  = urllib.parse.quote(os.getenv("PROXY_PASS", ""))
        proxy_url = f"http://{user}:{pwd}@{proxy_host}:{proxy_port}"
        os.environ["HTTP_PROXY"]  = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url

    # Gemini — LiteLLM читает GEMINI_API_KEY автоматически после load_dotenv()
    # Claude  — LiteLLM читает ANTHROPIC_API_KEY автоматически после load_dotenv()

    return os.getenv("OPENAI_MODEL_NAME", "gpt-5")
