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


def render_png(puml_code: str, timeout: int = 60) -> bytes | None:
    """
    Отправляет PlantUML-код на сервер и возвращает PNG как байты.
    Порядок попыток: kroki.io → plantuml.com POST → plantuml.com GET.
    Возвращает None при ошибке.
    """
    # 1. kroki.io — надёжнее для больших диаграмм, HTTPS
    try:
        response = requests.post(
            "https://kroki.io/plantuml/png",
            data=puml_code.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=timeout,
        )
        if response.status_code == 200 and response.content[:4] == b"\x89PNG":
            return response.content
    except requests.RequestException:
        pass

    # 2. plantuml.com POST — запасной вариант
    try:
        response = requests.post(
            "http://www.plantuml.com/plantuml/png/",
            data=puml_code.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=timeout,
        )
        if response.status_code == 200 and response.content[:4] == b"\x89PNG":
            return response.content
    except requests.RequestException:
        pass

    # 3. plantuml.com GET — для коротких диаграмм
    try:
        encoded = _encode_plantuml(puml_code)
        response = requests.get(
            f"http://www.plantuml.com/plantuml/png/{encoded}",
            timeout=timeout,
        )
        if response.status_code == 200 and response.content[:4] == b"\x89PNG":
            return response.content
    except requests.RequestException:
        pass

    return None
