import re
import json
from pathlib import Path
from datetime import datetime


_BLOCK_PATTERNS = {
    "class":     r'^\s*(?:abstract\s+class|class|interface|enum)\s+["\']?(\w[\w\s]*)["\']?',
    "sequence":  r'^\s*(?:participant|actor|boundary|control|entity|database)\s+["\']?([^"\n]+|"[^"]+")',
    "component": r'^\s*(?:component|database|queue|node|artifact|interface)\s+["\']?([^"\n{]+|"[^"]+")|^\s*\[([^\]]+)\]',
    "activity":  r'^\s*:([^:;\n]+);|^\s*"?([^";\n]+)"?\s*-->',
}

_ARROW_RE = re.compile(r'-->|\.\.>|\.\.\|>|\*--|o--|<\|--|<\|\.\.|\.\.|--|->>|->')

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

    if name_lower in req_lower:
        return True

    parts = [p.lower() for p in _CAMEL_SPLIT_RE.findall(name) if len(p) > 2]
    if any(p in req_lower for p in parts):
        return True

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

    m["syntax_valid"] = int("@startuml" in puml_code and "@enduml" in puml_code)
    m["diagram_lines"] = len(puml_code.splitlines())
    m["startuml_count"] = len(re.findall(r'^\s*@startuml\s*$', puml_code, re.MULTILINE | re.IGNORECASE))
    m["enduml_count"] = len(re.findall(r'^\s*@enduml\s*$', puml_code, re.MULTILINE | re.IGNORECASE))
    malformed_trace_ids = sorted(set(re.findall(r'\bRE[_\s]+Q[-_\s]*\d+(?:-\d+)?\b', puml_code, re.IGNORECASE)))
    m["malformed_trace_id_count"] = len(malformed_trace_ids)
    m["malformed_trace_ids"] = malformed_trace_ids[:20]

    pattern = _BLOCK_PATTERNS.get(diagram_type, _BLOCK_PATTERNS["class"])
    raw_matches = re.findall(pattern, puml_code, re.MULTILINE | re.IGNORECASE)
    block_names = []
    for match in raw_matches:
        if isinstance(match, tuple):
            name = next((m for m in match if m), "")
        else:
            name = match
        name = name.strip().strip('"\'')
        if name:
            block_names.append(name)
    m["block_count"] = len(block_names)

    all_relations = re.findall(
        r'([\w\[\]" ]+)\s*(-->|\.\.>|\.\.\|>|\*--|o--|<\|--|<\|\.\.|\.\.|--|->>|->|\.>)\s*([\w\[\]" ]+)',
        puml_code,
    )
    m["relation_count"] = len(all_relations)
    m["relation_variety"] = len({r[1] for r in all_relations})
    m["avg_relations_per_block"] = (
        round(m["relation_count"] / m["block_count"], 2) if m["block_count"] > 0 else 0.0
    )

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

    if block_names and requirements.strip():
        unmatched = sum(
            1 for name in block_names
            if not _name_in_requirements(name, req_lower, req_words)
        )
        m["excess_elements_pct"] = round(unmatched / len(block_names) * 100, 1)
    else:
        m["excess_elements_pct"] = 0.0

    relation_lines_text = " ".join(
        line for line in puml_code.splitlines() if _ARROW_RE.search(line)
    ).lower()
    m["isolated_nodes_count"] = sum(
        1 for name in block_names if name.lower() not in relation_lines_text
    )

    if diagram_type == "class":
        m["attribute_count"] = len(re.findall(
            r'^\s*[+\-#~]\s*\w[\w\s]*\s*:', puml_code, re.MULTILINE
        ))
        m["method_count"] = len(re.findall(
            r'^\s*[+\-#~]\s*\w[\w\s]*\s*\(', puml_code, re.MULTILINE
        ))
        m["enum_count"] = len(re.findall(r'^\s*enum\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["interface_count"] = len(re.findall(r'^\s*interface\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["package_count"] = len(re.findall(r'^\s*package\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["has_notes"] = int(bool(re.search(r'\bnote\b', puml_code, re.IGNORECASE)))
        m["avg_attributes_per_class"] = (
            round(m["attribute_count"] / m["block_count"], 2) if m["block_count"] > 0 else 0.0
        )
        rel_lines = [l for l in puml_code.splitlines() if _ARROW_RE.search(l)]
        with_mult = [
            l for l in rel_lines
            if re.search(r'"\s*[\d\*]+(?:\s*\.\.\s*[\d\*]+)?\s*"', l)
        ]
        m["multiplicity_specified_pct"] = (
            round(len(with_mult) / len(rel_lines) * 100, 1) if rel_lines else 0.0
        )
        m["class_quality_warnings"] = []
        if m["enum_count"] == 0:
            m["class_quality_warnings"].append("No enum declarations.")
        if m["relation_count"] < max(3, m["block_count"] // 2):
            m["class_quality_warnings"].append("Low relation count for the number of classifiers.")
        if m["isolated_nodes_count"] > 0:
            m["class_quality_warnings"].append("Some classifiers are isolated from relations.")

    elif diagram_type == "sequence":
        m["alt_block_count"] = len(re.findall(r'^\s*alt\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["loop_count"] = len(re.findall(r'^\s*loop\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["activation_count"] = len(re.findall(r'^\s*activate\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["sequence_quality_warnings"] = []
        if m["alt_block_count"] > 15 and m["loop_count"] == 0:
            m["sequence_quality_warnings"].append("Many alt blocks and no loops; sequence may be using alt as a generic container.")
        if m["activation_count"] == 0:
            m["sequence_quality_warnings"].append("No activation spans.")

    elif diagram_type == "component":
        m["interface_count"] = len(re.findall(r'^\s*interface\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["package_count"] = len(re.findall(r'^\s*package\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["component_quality_warnings"] = []
        if m["interface_count"] == 0:
            m["component_quality_warnings"].append("No explicit interfaces.")
        if m["relation_count"] < max(2, m["block_count"] // 2):
            m["component_quality_warnings"].append("Low dependency count for the number of components/interfaces.")

    elif diagram_type == "activity":
        swimlanes = set(re.findall(r'^\s*\|([^|]+)\|', puml_code, re.MULTILINE))
        m["swimlane_count"] = len(swimlanes)
        m["decision_count"] = len(re.findall(r'^\s*if\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["fork_count"] = len(re.findall(r'^\s*fork\b', puml_code, re.MULTILINE | re.IGNORECASE))
        m["activity_quality_warnings"] = []
        if m["enduml_count"] != 1:
            m["activity_quality_warnings"].append("Invalid PlantUML envelope count.")
        stop_positions = [match.start() for match in re.finditer(r'^\s*stop\s*$', puml_code, re.MULTILINE | re.IGNORECASE)]
        if len(stop_positions) > 1:
            m["activity_quality_warnings"].append("Multiple stop nodes; workflow may terminate inside alternatives.")
        elif stop_positions:
            tail = puml_code[stop_positions[0]:]
            if re.search(r'^\s*(if|repeat|fork|:|\|)', tail, re.MULTILINE | re.IGNORECASE):
                m["activity_quality_warnings"].append("Activity content continues after stop.")

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
    pipeline_mode: str = "multi",
    usage: dict | None = None,
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
        "pipeline_mode":       pipeline_mode,
        "prompt_tokens":       usage.get("prompt_tokens")     if usage else None,
        "completion_tokens":   usage.get("completion_tokens") if usage else None,
        "total_tokens":        usage.get("total_tokens")      if usage else None,
        "cost_usd":            usage.get("cost_usd")          if usage else None,
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


def save_mbse_history(
    requirements: str,
    model: str,
    rendered: dict,
    decomposed_raw: str,
    consistency_report: str,
    total_usage: dict,
    generation_time_sec: float,
    canonical_model: dict | None = None,
    model_issues: list[str] | None = None,
    requirement_facts: list[dict] | None = None,
    requirement_facts_raw: str | None = None,
    requirement_fact_issues: list[str] | None = None,
    system_design_raw: str | None = None,
    system_design_issues: list[str] | None = None,
    view_briefs: dict | None = None,
    view_brief_issues: list[str] | None = None,
    view_specs: dict | None = None,
    view_coverage: dict | None = None,
    diagram_quality_issues: dict | None = None,
    base_dir: str = "history_mbse",
) -> Path:
    """
    Сохраняет результат пакетной генерации в history_mbse/<timestamp>/.
    rendered: dict dtype → {puml, png, metrics, critique}
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path(base_dir) / f"{ts}_mbse"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "requirements.txt").write_text(requirements, encoding="utf-8")
    (folder / "decomposed.txt").write_text(decomposed_raw or "", encoding="utf-8")
    (folder / "consistency.txt").write_text(consistency_report or "", encoding="utf-8")
    if canonical_model is not None:
        (folder / "canonical_model.json").write_text(
            json.dumps(canonical_model, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if model_issues is not None:
        (folder / "model_issues.json").write_text(
            json.dumps(model_issues, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if requirement_facts is not None:
        (folder / "requirement_facts.json").write_text(
            json.dumps(requirement_facts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if requirement_facts_raw is not None:
        (folder / "requirement_facts_raw.txt").write_text(requirement_facts_raw or "", encoding="utf-8")
    if requirement_fact_issues is not None:
        (folder / "requirement_fact_issues.json").write_text(
            json.dumps(requirement_fact_issues, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if system_design_raw is not None:
        (folder / "system_design_raw.txt").write_text(system_design_raw or "", encoding="utf-8")
    if system_design_issues is not None:
        (folder / "system_design_issues.json").write_text(
            json.dumps(system_design_issues, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if view_briefs is not None:
        (folder / "view_briefs.json").write_text(
            json.dumps(view_briefs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if view_brief_issues is not None:
        (folder / "view_brief_issues.json").write_text(
            json.dumps(view_brief_issues, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if view_specs is not None:
        (folder / "view_specs.json").write_text(
            json.dumps(view_specs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if view_coverage is not None:
        (folder / "view_coverage.json").write_text(
            json.dumps(view_coverage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if diagram_quality_issues is not None:
        (folder / "diagram_quality_issues.json").write_text(
            json.dumps(diagram_quality_issues, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    meta = {
        "timestamp":           ts,
        "model":               model,
        "pipeline_mode":       "mbse",
        "generation_time_sec": round(generation_time_sec, 1),
        "prompt_tokens":       total_usage.get("prompt_tokens"),
        "completion_tokens":   total_usage.get("completion_tokens"),
        "total_tokens":        total_usage.get("total_tokens"),
        "cost_usd":            total_usage.get("cost_usd"),
    }
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for dtype, data in rendered.items():
        sub = folder / dtype
        sub.mkdir(exist_ok=True)
        if data.get("puml"):
            (sub / "diagram.puml").write_text(data["puml"], encoding="utf-8")
        if data.get("png"):
            (sub / "diagram.png").write_bytes(data["png"])
        if data.get("svg"):
            (sub / "diagram.svg").write_bytes(data["svg"])
        if data.get("metrics"):
            (sub / "metrics.json").write_text(
                json.dumps(data["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
        render_meta = {
            "valid": data.get("valid"),
            "render_status": data.get("render_status"),
            "render_error": data.get("render_error"),
            "render_fixer_used": data.get("render_fixer_used", False),
            "render_fixer_error_before": data.get("render_fixer_error_before"),
            "render_fixer_error_after": data.get("render_fixer_error_after"),
            "render_fixer_usage": data.get("render_fixer_usage"),
            "svg_status": "ok" if data.get("svg") else "not_saved",
            "svg_error": data.get("svg_error"),
        }
        (sub / "render.json").write_text(
            json.dumps(render_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if data.get("critique"):
            (sub / "critique.txt").write_text(data["critique"], encoding="utf-8")
        if data.get("semantic_coverage"):
            (sub / "semantic_coverage.json").write_text(
                json.dumps(data["semantic_coverage"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return folder


def load_mbse_history(base_dir: str = "history_mbse") -> list[Path]:
    """Возвращает список папок пакетной истории, отсортированных от новых к старым."""
    p = Path(base_dir)
    if not p.exists():
        return []
    return sorted([d for d in p.iterdir() if d.is_dir()], reverse=True)
