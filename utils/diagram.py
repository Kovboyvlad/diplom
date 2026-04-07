import re
import zlib
import base64
import requests


# PlantUML использует свой вариант base64
_PLANTUML_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
_BASE64_ALPHABET   = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_TRANS = str.maketrans(_BASE64_ALPHABET, _PLANTUML_ALPHABET)


def _encode_plantuml(puml_code: str) -> str:
    # PlantUML ожидает raw DEFLATE (без zlib-заголовка), wbits=-15
    compress_obj = zlib.compressobj(9, zlib.DEFLATED, -15)
    compressed = compress_obj.compress(puml_code.encode("utf-8")) + compress_obj.flush()
    b64 = base64.b64encode(compressed).decode("ascii")
    return b64.translate(_TRANS)


def clean_output(raw: str) -> str:
    """Убирает markdown-обёртку ```plantuml ... ``` из вывода GPT."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    return raw


def validate_puml(puml_code: str) -> bool:
    """Проверяет, что код начинается с @startuml и заканчивается @enduml."""
    return "@startuml" in puml_code and "@enduml" in puml_code


def render_png(puml_code: str, timeout: int = 15) -> bytes | None:
    """
    Отправляет PlantUML-код на публичный сервер и возвращает PNG как байты.
    Возвращает None при ошибке.
    """
    encoded = _encode_plantuml(puml_code)
    url = f"http://www.plantuml.com/plantuml/png/{encoded}"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None
