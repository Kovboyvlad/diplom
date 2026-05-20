import base64
import re
import zlib

import requests


_PLANTUML_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
_BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_TRANS = str.maketrans(_BASE64_ALPHABET, _PLANTUML_ALPHABET)


def _encode_plantuml(puml_code: str) -> str:
    compress_obj = zlib.compressobj(9, zlib.DEFLATED, -15)
    compressed = compress_obj.compress(puml_code.encode("utf-8")) + compress_obj.flush()
    b64 = base64.b64encode(compressed).decode("ascii")
    return b64.translate(_TRANS)


def clean_output(raw: str) -> str:
    """Remove markdown wrapper and keep only @startuml ... @enduml."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    start = raw.find("@startuml")
    end = raw.rfind("@enduml")
    if start != -1 and end != -1:
        raw = raw[start:end + len("@enduml")].strip()
    raw = _normalize_plantuml_envelope(raw)
    return raw


def _normalize_plantuml_envelope(puml_code: str) -> str:
    """Keep one @startuml and one final @enduml when an LLM duplicates wrappers."""
    lines = (puml_code or "").splitlines()
    if not lines:
        return ""

    normalized: list[str] = []
    seen_start = False
    for line in lines:
        stripped = line.strip().lower()
        if stripped == "@startuml":
            if seen_start:
                continue
            seen_start = True
            normalized.append(line)
            continue
        if stripped == "@enduml":
            continue
        normalized.append(line)

    if any(line.strip().lower() == "@enduml" for line in lines):
        normalized.append("@enduml")
    return "\n".join(normalized).strip()


def _normalize_trace_ids(line: str) -> str:
    """Repair common LLM typos in trace comments without touching domain text."""
    stripped = line.strip()
    if not stripped.startswith("'") or "trace" not in stripped.lower():
        return line
    return re.sub(r"\bRE[_\s-]*Q[-_\s]*(\d{3}(?:-\d+)?)\b", r"REQ-\1", line, flags=re.IGNORECASE)


def validate_puml(puml_code: str) -> bool:
    """Basic envelope validation. Rendering is the real syntax validation."""
    return "@startuml" in puml_code and "@enduml" in puml_code


def repair_puml(puml_code: str, diagram_type: str, strict: bool = False) -> str:
    """
    Fix common LLM PlantUML syntax mistakes without changing domain meaning.
    This is a renderability helper, not a UML critic.
    """
    lines = clean_output(puml_code).splitlines()
    repaired: list[str] = []

    for line in lines:
        line = _normalize_trace_ids(line)
        stripped = line.strip()

        if diagram_type == "sequence":
            if stripped.startswith("external "):
                line = line.replace("external ", "participant ", 1)
            elif stripped.startswith("component "):
                line = line.replace("component ", "participant ", 1)
            elif strict and stripped.startswith("exception "):
                indent = line[:len(line) - len(line.lstrip())]
                condition = stripped.replace("exception ", "", 1).strip()
                repaired.append(f"{indent}opt {condition}")
                continue

        elif diagram_type == "class":
            if strict and line.rstrip().endswith(" >"):
                line = line.rstrip()[:-2]

        elif diagram_type == "component":
            line = line.replace(" --( ", " --> ")
            line = line.replace(" --) ", " --> ")
            line = line.replace("--(", "-->")
            line = line.replace("--)", "-->")
            if line.rstrip().endswith(" >"):
                line = line.rstrip()[:-2]
            if strict:
                match = re.match(r'^(\s*)\["([^"]+)"\]\s+as\s+(\w+)(.*)$', line)
                if match:
                    indent, name, alias, tail = match.groups()
                    stereotype = ""
                    if "<<" in tail and ">>" in tail:
                        stereotype = " " + tail[tail.find("<<"):tail.find(">>") + 2]
                    line = f'{indent}component "{name}" as {alias}{stereotype}'
            if stripped.startswith("actor "):
                line = line.replace("actor ", "component ", 1)
            elif stripped.startswith("class "):
                line = line.replace("class ", "component ", 1)
            elif stripped.startswith("enum "):
                line = line.replace("enum ", "component ", 1)

        elif diagram_type == "activity":
            if strict:
                if stripped in {"(*)", "(*) -->", "--> (*)", "end repeat"}:
                    continue
                if stripped == "break":
                    continue
                if stripped.startswith("(*) -->"):
                    line = line.replace("(*) -->", "", 1).strip()
                    if not line:
                        continue
                elif stripped.endswith("--> (*)"):
                    line = line[:line.rfind("--> (*)")].rstrip()
                    if not line:
                        continue
            if stripped.startswith("note ") and ":" in stripped:
                prefix, text = line.split(":", 1)
                indent = line[:len(line) - len(line.lstrip())]
                repaired.append(prefix)
                repaired.append(f"{indent}{text.strip()}")
                repaired.append(f"{indent}end note")
                continue

        repaired.append(line)

    return _normalize_plantuml_envelope("\n".join(repaired)).strip()


def _fit_to_canvas(puml_code: str) -> str:
    directive = "scale max 4000 width\n"
    return puml_code.replace("@startuml", "@startuml\n" + directive, 1)


def _response_error(response: requests.Response, source: str) -> str:
    text = (response.text or "").strip()
    if text:
        text = re.sub(r"\s+", " ", text)
        return f"{source}: HTTP {response.status_code} - {text[:1200]}"
    return f"{source}: HTTP {response.status_code}"


def render_png(puml_code: str, timeout: int = 60) -> tuple[bytes | None, str | None]:
    """
    Send PlantUML to renderers and return either (PNG bytes, None) or
    (None, combined renderer diagnostics). Diagnostics are preserved for fixer agents.
    """
    puml_code = _fit_to_canvas(puml_code)
    errors: list[str] = []

    try:
        response = requests.post(
            "https://kroki.io/plantuml/png",
            data=puml_code.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=timeout,
        )
        if response.status_code == 200 and response.content[:4] == b"\x89PNG":
            return response.content, None
        errors.append(_response_error(response, "kroki.io"))
    except requests.RequestException as e:
        errors.append(f"kroki.io unavailable: {e}")

    try:
        response = requests.post(
            "http://www.plantuml.com/plantuml/png/",
            data=puml_code.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=timeout,
        )
        if response.status_code == 200 and response.content[:4] == b"\x89PNG":
            return response.content, None
        errors.append(_response_error(response, "plantuml.com POST"))
    except requests.RequestException as e:
        errors.append(f"plantuml.com POST unavailable: {e}")

    try:
        encoded = _encode_plantuml(puml_code)
        response = requests.get(
            f"http://www.plantuml.com/plantuml/png/{encoded}",
            timeout=timeout,
        )
        if response.status_code == 200 and response.content[:4] == b"\x89PNG":
            return response.content, None
        errors.append(_response_error(response, "plantuml.com GET"))
    except requests.RequestException as e:
        errors.append(f"plantuml.com GET unavailable: {e}")

    if errors:
        return None, "\n".join(errors)
    return None, "PlantUML render failed without diagnostic details."


def _looks_like_svg(content: bytes) -> bool:
    head = content[:300].lstrip()
    return head.startswith(b"<svg") or b"<svg" in head


def render_svg(puml_code: str, timeout: int = 60) -> tuple[bytes | None, str | None]:
    """
    Render PlantUML as SVG. SVG is preferred for large UML diagrams because it
    remains readable when zoomed and is suitable for presentation/PDF export.
    """
    errors: list[str] = []

    try:
        response = requests.post(
            "https://kroki.io/plantuml/svg",
            data=puml_code.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=timeout,
        )
        if response.status_code == 200 and _looks_like_svg(response.content):
            return response.content, None
        errors.append(_response_error(response, "kroki.io SVG"))
    except requests.RequestException as e:
        errors.append(f"kroki.io SVG unavailable: {e}")

    try:
        response = requests.post(
            "http://www.plantuml.com/plantuml/svg/",
            data=puml_code.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=timeout,
        )
        if response.status_code == 200 and _looks_like_svg(response.content):
            return response.content, None
        errors.append(_response_error(response, "plantuml.com SVG POST"))
    except requests.RequestException as e:
        errors.append(f"plantuml.com SVG POST unavailable: {e}")

    try:
        encoded = _encode_plantuml(puml_code)
        response = requests.get(
            f"http://www.plantuml.com/plantuml/svg/{encoded}",
            timeout=timeout,
        )
        if response.status_code == 200 and _looks_like_svg(response.content):
            return response.content, None
        errors.append(_response_error(response, "plantuml.com SVG GET"))
    except requests.RequestException as e:
        errors.append(f"plantuml.com SVG GET unavailable: {e}")

    if errors:
        return None, "\n".join(errors)
    return None, "PlantUML SVG render failed without diagnostic details."
