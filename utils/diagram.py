import re
import zlib
import base64
import requests


_PLANTUML_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
_BASE64_ALPHABET   = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_TRANS = str.maketrans(_BASE64_ALPHABET, _PLANTUML_ALPHABET)


def _encode_plantuml(puml_code: str) -> str:
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


def _fit_to_canvas(puml_code: str) -> str:
    """Вставляет директиву масштабирования чтобы широкие диаграммы не обрезались."""
    directive = "scale max 4000 width\n"
    return puml_code.replace("@startuml", "@startuml\n" + directive, 1)


def render_png(puml_code: str, timeout: int = 60) -> bytes | None:
    """
    Отправляет PlantUML-код на сервер и возвращает PNG как байты.
    Порядок попыток: kroki.io → plantuml.com POST → plantuml.com GET.
    Возвращает None при ошибке.
    """
    puml_code = _fit_to_canvas(puml_code)

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
