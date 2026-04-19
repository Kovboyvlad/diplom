import re
import json
from pathlib import Path
from datetime import datetime


# ── Паттерны блоков по типу диаграммы ─────────────────────────────────────────
_BLOCK_PATTERNS = {
    "class":     r'^\s*(?:abstract\s+class|class|interface|enum)\s+["\']?(\w[\w\s]*)["\']?',
    "sequence":  r'^\s*(?:participant|actor|boundary|control|entity|database)\s+["\']?(\w[\w\s]*)["\']?',
    "component": r'^\s*(?:component|database|queue|node|artifact)\s+["\']?(\w[\w\s]*)["\']?',
    "activity":  r'^\s*:([^:;\n]+);',
}

_ARROW_RE = re.compile(r'-->|\.\.>|\*--|o--|<\|--|--|->>|->')

_STOP_WORDS = {
    "это", "для", "что", "при", "или", "если", "есть", "также", "через",
    "the", "and", "for", "with", "that", "this", "from", "which", "each",
}

_CAMEL_SPLIT_RE = re.compile(r'[A-ZА-Я][a-zа-я]+|[A-ZА-Я]+(?=[A-ZА-Я]|$)|[a-zа-я]+')


def _name_in_requirements(name: str, req_lower: str, req_words: set) -> bool:
    """
    Проверяет, упоминается ли имя блока в тексте требований.
    Учитывает PascalCase/camelCase: WMSServer → ['WMS', 'Server'].
    Двустороннее совпадение: части имени в требованиях ИЛИ слова требований в имени.
    """
    name_lower = name.lower()

    # 1. Прямое вхождение
    if name_lower in req_lower:
        return True

    # 2. Разбиваем PascalCase/camelCase на части и ищем каждую в требованиях
    parts = [p.lower() for p in _CAMEL_SPLIT_RE.findall(name) if len(p) > 2]
    if any(p in req_lower for p in parts):
        return True

    # 3. Слова из требований встречаются внутри имени (AGV → "agv" в "agvrobot")
    if any(w in name_lower for w in req_words if len(w) > 2):
        return True

    return False


def compute_metrics(puml_code: str, requirements: str = "", diagram_type: str = "class") -> dict:
    """
    Вычисляет метрики качества PlantUML-кода.
    Универсальные метрики — для всех типов диаграмм.
    Специфичные — только для соответствующего diagram_type.
    """
    m = {}

    # ── Универсальные ──────────────────────────────────────────────────────────

    m["syntax_valid"] = int("@startuml" in puml_code and "@enduml" in puml_code)
    m["diagram_lines"] = len(puml_code.splitlines())

    # Имена блоков (зависит от типа)
    pattern = _BLOCK_PATTERNS.get(diagram_type, _BLOCK_PATTERNS["class"])
    block_names = [
        n.strip() for n in re.findall(pattern, puml_code, re.MULTILINE | re.IGNORECASE)
        if n.strip()
    ]
    m["block_count"] = len(block_names)

    # Связи
    all_relations = re.findall(
        r'(\w[\w\s"]*)\s*(-->|\.\.>|\*--|o--|<\|--|<\|\.\.|\.\.|--|->>|->|\.>)\s*(\w[\w\s"]*)',
        puml_code,
    )
    m["relation_count"] = len(all_relations)
    m["relation_variety"] = len({r[1] for r in all_relations})
    m["avg_relations_per_block"] = (
        round(m["relation_count"] / m["block_count"], 2) if m["block_count"] > 0 else 0.0
    )

    # Покрытие требований (bag-of-words)
    req_words: set[str] = set()
    req_lower = requirements.lower()
    if requirements.strip():
        req_words = {
            w.lower()
            for w in re.findall(r'\b[а-яёa-zA-Z]{4,}\b', requirements)
            if w.lower() not in _STOP_WORDS
        }
        puml_lower = puml_code.lower()
        matched = sum(1 for w in req_words if w in puml_lower)
        m["entity_coverage_pct"] = round(matched / len(req_words) * 100, 1) if req_words else 0.0
    else:
        m["entity_coverage_pct"] = 0.0

    # Лишние элементы — блоки, название которых не соотносится с требованиями
    if block_names and requirements.strip():
        unmatched = sum(
            1 for name in block_names
            if not _name_in_requirements(name, req_lower, req_words)
        )
        m["excess_elements_pct"] = round(unmatched / len(block_names) * 100, 1)
    else:
        m["excess_elements_pct"] = 0.0

    # Изолированные узлы — блоки без единой стрелки
    relation_lines_text = " ".join(
        line for line in puml_code.splitlines() if _ARROW_RE.search(line)
    ).lower()
    m["isolated_nodes_count"] = sum(
        1 for name in block_names if name.lower() not in relation_lines_text
    )

    # ── Специфичные по типу ───────────────────────────────────────────────────

    if diagram_type == "class":
        m["attribute_count"] = len(re.findall(
            r'^\s*[+\-#~]\s+\w[\w\s]*\s*:', puml_code, re.MULTILINE
        ))
        m["method_count"] = len(re.findall(
            r'^\s*[+\-#~]\s+\w[\w\s]*\s*\(', puml_code, re.MULTILINE
        ))
        m["enum_count"] = len(re.findall(r'^\s*enum\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["interface_count"] = len(re.findall(r'^\s*interface\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["package_count"] = len(re.findall(r'^\s*package\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["has_notes"] = int(bool(re.search(r'\bnote\b', puml_code, re.IGNORECASE)))
        m["avg_attributes_per_class"] = (
            round(m["attribute_count"] / m["block_count"], 2) if m["block_count"] > 0 else 0.0
        )
        # Кратность связей: "1", "0..*", "1..1" рядом со стрелкой
        rel_lines = [l for l in puml_code.splitlines() if _ARROW_RE.search(l)]
        with_mult = [
            l for l in rel_lines
            if re.search(r'"\s*[\d\*]+(?:\s*\.\.\s*[\d\*]+)?\s*"', l)
        ]
        m["multiplicity_specified_pct"] = (
            round(len(with_mult) / len(rel_lines) * 100, 1) if rel_lines else 0.0
        )

    elif diagram_type == "sequence":
        m["alt_block_count"] = len(re.findall(r'^\s*alt\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["loop_count"] = len(re.findall(r'^\s*loop\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["activation_count"] = len(re.findall(r'^\s*activate\b', puml_code, re.MULTILINE | re.IGNORECASE))

    elif diagram_type == "component":
        m["interface_count"] = len(re.findall(r'^\s*interface\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["package_count"] = len(re.findall(r'^\s*package\b', puml_code, re.MULTILINE | re.IGNORECASE))

    elif diagram_type == "activity":
        # Уникальные swimlane-полосы
        swimlanes = set(re.findall(r'^\s*\|([^|]+)\|', puml_code, re.MULTILINE))
        m["swimlane_count"] = len(swimlanes)
        m["decision_count"] = len(re.findall(r'^\s*if\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["fork_count"] = len(re.findall(r'^\s*fork\b', puml_code, re.MULTILINE | re.IGNORECASE))

    return m


def save_history(
    requirements: str,
    diagram_type: str,
    model: str,
    puml_code: str,
    png_bytes: bytes | None,
    metrics: dict,
    critique: str | None = None,
    evaluation: str | None = None,
    generation_time_sec: float | None = None,
    experiment_id: str | None = None,
    base_dir: str = "history",
) -> Path:
    """Сохраняет результат генерации в папку history/<timestamp>."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path(base_dir) / f"{ts}_{diagram_type}"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "requirements.txt").write_text(requirements, encoding="utf-8")
    (folder / "diagram.puml").write_text(puml_code, encoding="utf-8")

    if png_bytes:
        (folder / "diagram.png").write_bytes(png_bytes)

    if critique:
        (folder / "critique.txt").write_text(critique, encoding="utf-8")

    if evaluation:
        (folder / "evaluation.txt").write_text(evaluation, encoding="utf-8")

    meta = {
        "timestamp":           ts,
        "diagram_type":        diagram_type,
        "model":               model,
        "generation_time_sec": round(generation_time_sec, 1) if generation_time_sec is not None else None,
        "experiment_id":       experiment_id,
    }
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return folder


def load_history(base_dir: str = "history") -> list[Path]:
    """Возвращает список папок истории, отсортированных от новых к старым."""
    p = Path(base_dir)
    if not p.exists():
        return []
    return sorted([d for d in p.iterdir() if d.is_dir()], reverse=True)
