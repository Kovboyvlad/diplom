import base64
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config
from agents.pipeline import (
    run_pipeline,
    run_pipeline_slim,
    run_single_agent,
    run_evaluation,
    run_mbse_pipeline,
    run_render_fixer,
)
from utils.diagram import clean_output, validate_puml, render_png, render_svg, repair_puml
from utils.file_reader import read_uploaded_file
from utils.metrics import compute_metrics, save_history, load_history, save_mbse_history, load_mbse_history

config.setup()


def render_svg_preview(svg_bytes: bytes, height: int = 760) -> None:
    encoded = base64.b64encode(svg_bytes).decode("ascii")
    components.html(
        f"""
        <div style="height:{height}px; width:100%; overflow:auto; border:1px solid #ddd; background:white;">
          <img src="data:image/svg+xml;base64,{encoded}" style="max-width:none; height:auto;" />
        </div>
        """,
        height=height + 20,
        scrolling=True,
    )


# ── Метки метрик ──────────────────────────────────────────────────────────────
METRIC_LABELS = {
    # Универсальные
    "syntax_valid":             "Синтаксис",
    "diagram_lines":            "Строк кода",
    "block_count":              "Блоков",
    "relation_count":           "Связей",
    "relation_variety":         "Типов связей",
    "avg_relations_per_block":  "Связей/блок",
    "entity_coverage_pct":      "Покрытие req (%)",
    "excess_elements_pct":      "Лишних эл. (%)",
    "isolated_nodes_count":     "Изолир. узлов",
    # Class
    "attribute_count":          "Атрибутов",
    "method_count":             "Методов",
    "enum_count":               "Enum",
    "interface_count":          "Интерфейсов",
    "package_count":            "Пакетов",
    "has_notes":                "Заметки",
    "avg_attributes_per_class": "Атр./класс",
    "multiplicity_specified_pct": "Кратность связей (%)",
    # Sequence
    "alt_block_count":          "Ветвлений (alt)",
    "loop_count":               "Циклов (loop)",
    "activation_count":         "Активаций",
    # Activity
    "swimlane_count":           "Swimlane-полос",
    "decision_count":           "Решений (if)",
    "fork_count":               "Параллелей (fork)",
}

# ── Интерфейс ─────────────────────────────────────────────────────────────────

def _add_usage(total: dict, addition: dict | None) -> None:
    if not addition:
        return
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        if addition.get(key) is not None:
            total[key] = (total.get(key) or 0) + addition[key]
    if addition.get("cost_usd") is not None:
        total["cost_usd"] = round((total.get("cost_usd") or 0.0) + addition["cost_usd"], 6)


st.set_page_config(
    page_title="AI Diagram Generator",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
}

p, li, .stMarkdown {
    font-size: 16px;
    line-height: 1.6;
}

h1, h2, h3 {
    font-weight: 500;
    letter-spacing: -0.3px;
}

.stButton > button {
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
}

[data-testid="metric-container"] {
    background: transparent;
    border: 1px solid #E0DED9;
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("Генератор архитектурных диаграмм")
st.caption("Введите требования к системе — ИИ-агенты сгенерируют PlantUML-диаграмму.")

tab_generate, tab_mbse, tab_history, tab_analytics, tab_prompts = st.tabs(
    ["Генерация", "Пакетная генерация", "История", "Аналитика", "Настройки промптов"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка: Генерация
# ══════════════════════════════════════════════════════════════════════════════
with tab_generate:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Входные данные")

        uploaded = st.file_uploader(
            "Загрузить требования из файла (.txt или .pdf)",
            type=["txt", "pdf"],
        )

        default_text = ""
        if uploaded is not None:
            try:
                default_text = read_uploaded_file(uploaded)
                st.success(f"Файл «{uploaded.name}» загружен.")
            except Exception as e:
                st.error(f"Не удалось прочитать файл: {e}")

        requirements = st.text_area(
            "Текст требований",
            value=default_text,
            height=280,
        )

        diagram_type_label = st.selectbox(
            "Тип диаграммы",
            options=list(config.DIAGRAM_TYPES.keys()),
        )
        diagram_type = config.DIAGRAM_TYPES[diagram_type_label]

        model_label = st.selectbox(
            "Модель ИИ",
            options=list(config.MODELS.keys()),
        )
        model_id = config.MODELS[model_label]

        pipeline_mode = st.radio(
            "Режим генерации",
            options=[
                "Мультиагентный пайплайн (5 агентов)",
                "Мультиагентный slim (ограниченный контекст)",
                "Одиночный агент",
            ],
            help=(
                "Slim: каждый агент видит только вывод предыдущего — экономит токены. "
                "Baseline: только один агент-кодер без Analyst и Architect."
            ),
        )
        use_single_agent = pipeline_mode.startswith("Одиночный")
        use_slim = pipeline_mode.startswith("Мультиагентный slim")

        run_analysis = st.toggle("Требуется анализ результатов", value=False,
                                 help="Запускает агента-оценщика после генерации. Увеличивает время.")

        generate_btn = st.button("Сгенерировать", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Результат")

        if generate_btn:
            if not requirements.strip():
                st.warning("Введите текст требований.")
            else:
                status = st.empty()
                progress = st.progress(0, text="Запуск агентов...")

                try:
                    status.info("Агенты работают...")
                    progress.progress(10, text="Анализ требований...")

                    t_start = time.time()
                    if use_single_agent:
                        raw_result, critique, usage, intermediates = run_single_agent(requirements, diagram_type, model_id)
                    elif use_slim:
                        raw_result, critique, usage, intermediates = run_pipeline_slim(requirements, diagram_type, model_id)
                    else:
                        raw_result, critique, usage, intermediates = run_pipeline(requirements, diagram_type, model_id)
                    generation_time_sec = time.time() - t_start

                    progress.progress(75, text="Валидация...")
                    puml_code = clean_output(raw_result)

                    if not validate_puml(puml_code):
                        st.error("Агенты не сгенерировали корректный PlantUML-код. Попробуйте ещё раз.")
                        progress.empty()
                        status.empty()
                    else:
                        progress.progress(85, text="Вычисление метрик...")
                        metrics = compute_metrics(puml_code, requirements, diagram_type)

                        evaluation = None
                        if run_analysis:
                            progress.progress(88, text="Анализ результата агентом-оценщиком...")
                            evaluation = run_evaluation(requirements, puml_code, diagram_type, metrics, model_id)

                        progress.progress(94, text="Рендеринг PNG...")
                        png_bytes, render_error = render_png(puml_code)
                        metrics["syntax_valid"] = 1 if png_bytes else 0

                        # Сохраняем на диск и в историю
                        Path("diagram.puml").write_text(puml_code, encoding="utf-8")
                        if png_bytes:
                            Path("diagram.png").write_bytes(png_bytes)

                        if use_single_agent:
                            mode_str = "single"
                        elif use_slim:
                            mode_str = "multi_slim"
                        else:
                            mode_str = "multi"
                        save_history(requirements, diagram_type, model_id, puml_code, png_bytes, metrics, critique, evaluation, generation_time_sec,
                                     pipeline_mode=mode_str,
                                     usage=usage)

                        progress.progress(100, text="Готово!")
                        status.success("Диаграмма сгенерирована и сохранена в историю.")

                        # ── Метрики ───────────────────────────────────────────
                        st.subheader("Метрики качества")

                        # Универсальные — всегда 2 строки по 4
                        u1, u2, u3, u4 = st.columns(4)
                        u1.metric("Синтаксис",       "Да" if metrics["syntax_valid"] else "Нет")
                        u2.metric("Блоков",           metrics["block_count"])
                        u3.metric("Связей",           metrics["relation_count"])
                        u4.metric("Строк кода",       metrics["diagram_lines"])

                        u5, u6, u7, u8 = st.columns(4)
                        u5.metric("Покрытие req",     f"{metrics['entity_coverage_pct']}%")
                        u6.metric("Лишних эл.",       f"{metrics['excess_elements_pct']}%")
                        u7.metric("Изолир. узлов",    metrics["isolated_nodes_count"])
                        u8.metric("Время генерации",  f"{generation_time_sec:.1f} с")

                        t1, t2, t3, t4 = st.columns(4)
                        t1.metric("Prompt tokens",      usage.get("prompt_tokens") or "—")
                        t2.metric("Completion tokens",  usage.get("completion_tokens") or "—")
                        t3.metric("Total tokens",       usage.get("total_tokens") or "—")
                        cost = usage.get("cost_usd")
                        t4.metric("Стоимость",          f"${cost:.4f}" if cost else "—")

                        # Специфичные по типу — только если есть в метриках
                        type_specific_keys = [
                            k for k in metrics
                            if k not in {
                                "syntax_valid", "block_count", "relation_count", "diagram_lines",
                                "entity_coverage_pct", "excess_elements_pct", "isolated_nodes_count",
                                "relation_variety", "avg_relations_per_block",
                            }
                        ]
                        if type_specific_keys:
                            st.caption(f"Метрики для типа: **{diagram_type}**")
                            cols = st.columns(4)
                            for i, key in enumerate(type_specific_keys):
                                label = METRIC_LABELS.get(key, key)
                                val = metrics[key]
                                if isinstance(val, float):
                                    display = f"{val}%"  if "pct" in key else str(val)
                                else:
                                    display = "Да" if val == 1 and key == "has_notes" else str(val)
                                cols[i % 4].metric(label, display)

                        # ── Промежуточные результаты агентов (XAI) ───────────
                        if intermediates:
                            with st.expander("Рассуждения агентов — что передаётся между ними", expanded=False):
                                AGENT_LABELS = {
                                    "analyst":   "Аналитик — извлечённые сущности и требования",
                                    "architect": "Архитектор — спроектированная архитектура",
                                    "coder_v1":  "Кодер (v1) — первоначальный PlantUML до критика",
                                }
                                for key, label in AGENT_LABELS.items():
                                    if key in intermediates and intermediates[key]:
                                        st.markdown(f"**{label}**")
                                        st.text(intermediates[key])
                                        st.divider()

                        # ── Оценка агента-оценщика ────────────────────────────
                        if evaluation:
                            with st.expander("Анализ результатов (агент-оценщик)", expanded=True):
                                st.markdown(evaluation)

                        # ── Диаграмма ─────────────────────────────────────────
                        if png_bytes:
                            st.image(png_bytes, use_container_width=True)
                        else:
                            st.warning("PNG не отрисован — синтаксическая ошибка в коде.")
                            if render_error:
                                with st.expander("Детали ошибки рендеринга"):
                                    st.code(render_error, language="text")

                        st.subheader("PlantUML-код")
                        st.code(puml_code, language="text")

                        c_dl1, c_dl2 = st.columns(2)
                        with c_dl1:
                            st.download_button("Скачать .puml", puml_code,
                                               "diagram.puml", "text/plain",
                                               use_container_width=True)
                        with c_dl2:
                            if png_bytes:
                                st.download_button("Скачать .png", png_bytes,
                                                   "diagram.png", "image/png",
                                                   use_container_width=True)

                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"Ошибка: {e}")
        else:
            st.info("Заполните требования слева и нажмите «Сгенерировать».")

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка: MBSE — Комплект диаграмм
# ══════════════════════════════════════════════════════════════════════════════
with tab_mbse:
    st.subheader("Генерация полного комплекта UML-диаграмм")
    st.caption(
        "Система строит общий Project Context, а затем генерирует 4 UML-представления "
        "из исходных требований с единым глоссарием."
    )

    mb_col_left, mb_col_right = st.columns([1, 1], gap="large")

    with mb_col_left:
        st.subheader("Входные данные")

        mb_uploaded = st.file_uploader(
            "Загрузить требования (.txt или .pdf)",
            type=["txt", "pdf"],
            key="mbse_uploader",
        )

        if mb_uploaded is not None:
            try:
                st.session_state["mbse_requirements"] = read_uploaded_file(mb_uploaded)
                st.success(f"Файл «{mb_uploaded.name}» загружен.")
            except Exception as e:
                st.error(f"Не удалось прочитать файл: {e}")

        mb_requirements = st.text_area(
            "Текст требований",
            height=320,
            key="mbse_requirements",
        )

        mb_model_label = st.selectbox(
            "Модель ИИ",
            options=list(config.MODELS.keys()),
            key="mbse_model",
        )
        mb_model_id = config.MODELS[mb_model_label]

        mb_generate_btn = st.button(
            "Сгенерировать комплект",
            type="primary",
            use_container_width=True,
            key="mbse_generate",
        )

    with mb_col_right:
        st.subheader("Результат")

        if mb_generate_btn:
            if not mb_requirements.strip():
                st.warning("Введите текст требований.")
            else:
                mb_status  = st.empty()
                mb_progress = st.progress(0, text="Запуск MBSE-пайплайна...")

                try:
                    mb_status.info("Агенты работают...")
                    mb_progress.progress(5, text="Построение Project Context...")

                    t_start = time.time()
                    result = run_mbse_pipeline(mb_requirements, mb_model_id)

                    mb_progress.progress(90, text="Рендеринг диаграмм...")

                    DTYPE_LABELS = {
                        "class":     "Class — Классовая",
                        "sequence":  "Sequence — Последовательности",
                        "component": "Component — Компонентная",
                        "activity":  "Activity — Деятельности",
                    }

                    # Рендерим PNG для всех диаграмм
                    rendered: dict = {}
                    for dtype, data in result["diagrams"].items():
                        puml = repair_puml(clean_output(data["puml"]), dtype, strict=True)
                        valid = validate_puml(puml)
                        fixer_used = False
                        fixer_error_before = None
                        fixer_error_after = None
                        fixer_usage = {}
                        svg = None
                        svg_error = None
                        if valid:
                            png, render_err = render_png(puml)
                            if not png:
                                fixer_used = True
                                fixer_error_before = render_err
                                mb_status.info(f"РџРѕРІС‚РѕСЂРЅРѕРµ РёСЃРїСЂР°РІР»РµРЅРёРµ PlantUML РґР»СЏ {dtype}...")
                                try:
                                    fixed_raw, fixer_usage = run_render_fixer(
                                        puml_code=puml,
                                        diagram_type=dtype,
                                        render_error=render_err,
                                        model=mb_model_id,
                                    )
                                    _add_usage(result["total_usage"], fixer_usage)
                                    fixed_puml = repair_puml(clean_output(fixed_raw), dtype, strict=True)
                                    if validate_puml(fixed_puml):
                                        fixed_png, fixed_render_err = render_png(fixed_puml)
                                        puml = fixed_puml
                                        if fixed_png:
                                            png = fixed_png
                                            render_err = None
                                        else:
                                            render_err = fixed_render_err
                                        fixer_error_after = fixed_render_err
                                    else:
                                        fixer_error_after = "Render fixer returned invalid PlantUML."
                                except Exception as e:
                                    fixer_error_after = f"Render fixer failed: {e}"
                            if png:
                                svg, svg_error = render_svg(puml)
                        else:
                            png, render_err = None, None
                        rendered[dtype] = {
                            "puml":         puml,
                            "valid":        valid,
                            "png":          png,
                            "svg":          svg,
                            "svg_error":    svg_error,
                            "render_error": render_err,
                            "render_status": "ok" if png else ("render_error" if valid else "invalid_puml"),
                            "render_fixer_used": fixer_used,
                            "render_fixer_error_before": fixer_error_before,
                            "render_fixer_error_after": fixer_error_after,
                            "render_fixer_usage": fixer_usage,
                            "metrics":      compute_metrics(
                                puml,
                                result.get("view_specs", {}).get(dtype, mb_requirements),
                                dtype,
                            ) if valid else {},
                            "critique":     data["critique"],
                            "semantic_coverage": data.get("semantic_coverage"),
                        }
                        if rendered[dtype]["metrics"]:
                            rendered[dtype]["metrics"]["syntax_valid"] = 1 if png else 0

                    mb_progress.progress(100, text="Готово!")
                    total_time = time.time() - t_start
                    save_mbse_history(
                        requirements=mb_requirements,
                        model=mb_model_id,
                        rendered=rendered,
                        decomposed_raw=result.get("decomposed_raw", ""),
                        consistency_report=result.get("consistency_report", ""),
                        total_usage=result["total_usage"],
                        generation_time_sec=total_time,
                        canonical_model=result.get("canonical_model"),
                        model_issues=result.get("model_issues"),
                        requirement_facts=result.get("requirement_facts"),
                        requirement_facts_raw=result.get("requirement_facts_raw"),
                        requirement_fact_issues=result.get("requirement_fact_issues"),
                        system_design_raw=result.get("system_design_raw"),
                        system_design_issues=result.get("system_design_issues"),
                        view_briefs=result.get("view_briefs"),
                        view_brief_issues=result.get("view_brief_issues"),
                        view_specs=result.get("view_specs"),
                        view_coverage=result.get("view_coverage"),
                        diagram_quality_issues=result.get("diagram_quality_issues"),
                    )
                    mb_status.success(f"Комплект сгенерирован за {total_time:.1f} с. и сохранён в историю.")

                    # ── Project Context (XAI) ─────────────────────────────────
                    with st.expander("Project Context / общий глоссарий (JSON)", expanded=False):
                        st.json(result.get("canonical_model", {}))

                    model_issues = result.get("model_issues") or []
                    if model_issues:
                        with st.expander("Замечания валидации Project Context", expanded=False):
                            for issue in model_issues:
                                st.write(f"- {issue}")

                    with st.expander("View Briefs / распределение требований по диаграммам", expanded=False):
                        st.json(result.get("view_briefs", {}))

                    view_brief_issues = result.get("view_brief_issues") or []
                    if view_brief_issues:
                        with st.expander("Замечания валидации View Briefs", expanded=False):
                            for issue in view_brief_issues:
                                st.write(f"- {issue}")

                    view_coverage = result.get("view_coverage") or {}
                    if view_coverage:
                        vc1, vc2 = st.columns(2)
                        vc1.metric("Среднее покрытие view-spec", f"{view_coverage.get('avg_view_coverage_pct', '—')}%")
                        vc2.metric("Проблем модели", len(model_issues))

                    diagram_quality_issues = result.get("diagram_quality_issues") or {}
                    if any(diagram_quality_issues.get(dtype) for dtype in diagram_quality_issues):
                        with st.expander("Programmatic quality / hygiene warnings", expanded=True):
                            st.json(diagram_quality_issues)

                    with st.expander("Фактические входы для 4 диаграмм", expanded=False):
                        for dtype, spec in (result.get("view_specs") or {}).items():
                            st.markdown(f"**{dtype}**")
                            st.text(spec)
                            st.divider()

                    # ── Итоговые токены ────────────────────────────────────────
                    tu = result["total_usage"]
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("Prompt tokens",     tu.get("prompt_tokens") or "—")
                    tc2.metric("Completion tokens", tu.get("completion_tokens") or "—")
                    tc3.metric("Total tokens",      tu.get("total_tokens") or "—")
                    cost = tu.get("cost_usd")
                    tc4.metric("Стоимость",         f"${cost:.4f}" if cost else "—")
                    tc1.metric("Время генерации",   f"{total_time:.1f} с")

                    st.divider()

                    # ── Диаграммы по вкладкам ──────────────────────────────────
                    d_tabs = st.tabs([DTYPE_LABELS[d] for d in ["class", "sequence", "component", "activity"]])

                    for tab_obj, dtype in zip(d_tabs, ["class", "sequence", "component", "activity"]):
                        with tab_obj:
                            data = rendered[dtype]

                            if not data["valid"]:
                                st.error("Код не содержит корректный блок @startuml/@enduml.")
                            else:
                                m = data["metrics"]
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Рендер",       "✓" if data["png"] else "✗")
                                m2.metric("Блоков",       m.get("block_count", "—"))
                                m3.metric("Связей",       m.get("relation_count", "—"))
                                semantic = data.get("semantic_coverage") or {}
                                if semantic:
                                    m4.metric("Trace coverage", f"{semantic.get('coverage_pct', '—')}%")
                                else:
                                    m4.metric("Покрытие req", f"{m.get('entity_coverage_pct', '—')}%")

                            if data.get("svg"):
                                st.caption("Читаемая SVG-версия: можно прокручивать и масштабировать браузером без потери качества.")
                                render_svg_preview(data["svg"])
                            elif data["png"]:
                                st.image(data["png"], use_container_width=True)
                            else:
                                if data["valid"]:
                                    st.warning("PNG не отрисован — PlantUML-сервер вернул ошибку.")
                                else:
                                    st.warning("PNG не отрисован — код не содержит корректный @startuml/@enduml.")
                                if data.get("render_error"):
                                    with st.expander("Детали ошибки рендеринга"):
                                        st.code(data["render_error"], language="text")

                            st.code(data["puml"], language="text")

                            dl1, dl2, dl3 = st.columns(3)
                            with dl1:
                                st.download_button(
                                    "Скачать .puml", data["puml"],
                                    f"{dtype}.puml", "text/plain",
                                    use_container_width=True,
                                    key=f"mbse_dl_puml_{dtype}",
                                )
                            with dl2:
                                if data["png"]:
                                    st.download_button(
                                        "Скачать .png", data["png"],
                                        f"{dtype}.png", "image/png",
                                        use_container_width=True,
                                        key=f"mbse_dl_png_{dtype}",
                                    )
                            with dl3:
                                if data.get("svg"):
                                    st.download_button(
                                        "Скачать .svg", data["svg"],
                                        f"{dtype}.svg", "image/svg+xml",
                                        use_container_width=True,
                                        key=f"mbse_dl_svg_{dtype}",
                                    )

                            if data["critique"]:
                                with st.expander("Замечания критика", expanded=False):
                                    st.markdown(data["critique"])

                            warning_keys = [k for k in data.get("metrics", {}) if k.endswith("_quality_warnings")]
                            warnings = []
                            for key in warning_keys:
                                warnings.extend(data["metrics"].get(key) or [])
                            if warnings or data.get("metrics", {}).get("malformed_trace_id_count"):
                                with st.expander("Programmatic quality warnings", expanded=False):
                                    if data.get("metrics", {}).get("malformed_trace_id_count"):
                                        st.write(f"- Malformed trace IDs: {data['metrics'].get('malformed_trace_ids')}")
                                    for warning in warnings:
                                        st.write(f"- {warning}")

                            if data.get("semantic_coverage"):
                                with st.expander("Trace coverage / missing facts", expanded=False):
                                    st.json(data["semantic_coverage"])

                            inter = result["diagrams"][dtype].get("intermediates", {})
                            if inter:
                                with st.expander("Рассуждения агентов", expanded=False):
                                    AGENT_LABELS = {
                                        "analyst":   "Аналитик — извлечённые сущности",
                                        "architect": "Архитектор — спроектированная архитектура",
                                        "coder_v1":  "Кодер (v1) — до критика",
                                    }
                                    for key, label in AGENT_LABELS.items():
                                        if inter.get(key):
                                            st.markdown(f"**{label}**")
                                            st.text(inter[key])
                                            st.divider()

                    st.divider()

                    # ── Отчёт о согласованности ────────────────────────────────
                    with st.expander("Отчёт о согласованности диаграмм (Consistency Checker)", expanded=True):
                        st.markdown(result["consistency_report"])

                except Exception as e:
                    mb_progress.empty()
                    mb_status.empty()
                    st.error(f"Ошибка: {e}")
        else:
            st.info("Введите требования слева и нажмите «Сгенерировать комплект».")

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка: История
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("История генераций")

    items = load_history()

    if not items:
        st.info("История пуста. Сгенерируйте первую диаграмму!")
    else:
        for folder in items:
            meta_path     = folder / "meta.json"
            metrics_path  = folder / "metrics.json"
            puml_path     = folder / "diagram.puml"
            png_path      = folder / "diagram.png"
            svg_path      = folder / "diagram.svg"
            req_path      = folder / "requirements.txt"
            critique_path    = folder / "critique.txt"
            evaluation_path  = folder / "evaluation.txt"

            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            label = f"{meta.get('timestamp', folder.name)}  |  {meta.get('diagram_type', '')}  |  {meta.get('model', '')}"

            with st.expander(label, expanded=False):
                h_left, h_right = st.columns([1, 1])

                with h_left:
                    if req_path.exists():
                        st.text_area("Требования", req_path.read_text(encoding="utf-8"),
                                     height=150, disabled=True, key=f"req_{folder.name}")

                    if metrics_path.exists():
                        m = json.loads(metrics_path.read_text(encoding="utf-8"))
                        st.markdown("**Метрики**")
                        # Показываем все доступные метрики динамически
                        all_keys = list(m.keys())
                        cols_h = st.columns(4)
                        for i, key in enumerate(all_keys):
                            label = METRIC_LABELS.get(key, key)
                            val = m[key]
                            if isinstance(val, float):
                                display = f"{val}%" if "pct" in key else str(val)
                            elif key == "syntax_valid":
                                display = "✓" if val else "✗"
                            elif key == "has_notes":
                                display = "Да" if val else "Нет"
                            else:
                                display = str(val) if val is not None else "—"
                            cols_h[i % 4].metric(label, display)

                        gen_time = meta.get("generation_time_sec")
                        if gen_time is not None:
                            th, *_ = st.columns(4)
                            th.metric("Время генерации", f"{gen_time} с")

                    if evaluation_path.exists():
                        with st.expander("Оценка диаграммы", expanded=False):
                            st.markdown(evaluation_path.read_text(encoding="utf-8"))

                    if puml_path.exists():
                        puml_text = puml_path.read_text(encoding="utf-8")
                        st.code(puml_text, language="text")
                        st.download_button("Скачать .puml", puml_text,
                                           "diagram.puml", "text/plain",
                                           key=f"dl_puml_{folder.name}",
                                           use_container_width=True)

                with h_right:
                    if svg_path.exists():
                        render_svg_preview(svg_path.read_bytes())
                        with open(svg_path, "rb") as f:
                            st.download_button("Скачать .svg", f.read(),
                                               "diagram.svg", "image/svg+xml",
                                               key=f"dl_svg_{folder.name}",
                                               use_container_width=True)
                    elif png_path.exists():
                        st.image(str(png_path), use_container_width=True)
                    else:
                        st.caption("SVG/PNG не сохранён.")
                    if png_path.exists():
                        with open(png_path, "rb") as f:
                            st.download_button("Скачать .png", f.read(),
                                               "diagram.png", "image/png",
                                               key=f"dl_png_{folder.name}",
                                               use_container_width=True)

    st.divider()
    st.subheader("История пакетной генерации")

    mbse_items = load_mbse_history()

    if not mbse_items:
        st.info("Пакетных генераций ещё не было.")
    else:
        DTYPE_LABELS_H = {
            "class": "Class", "sequence": "Sequence",
            "component": "Component", "activity": "Activity",
        }
        for folder in mbse_items:
            meta_path = folder / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            cost = meta.get("cost_usd")
            cost_str = f"  |  ${cost:.4f}" if cost else ""
            label = f"{meta.get('timestamp', folder.name)}  |  {meta.get('model', '')}  |  {meta.get('generation_time_sec', '—')} с{cost_str}"

            with st.expander(label, expanded=False):
                req_path = folder / "requirements.txt"
                if req_path.exists():
                    st.text_area("Требования", req_path.read_text(encoding="utf-8"),
                                 height=120, disabled=True, key=f"mbse_req_{folder.name}")

                canonical_path = folder / "canonical_model.json"
                if canonical_path.exists():
                    with st.expander("Project Context", expanded=False):
                        st.json(json.loads(canonical_path.read_text(encoding="utf-8")))

                issues_path = folder / "model_issues.json"
                if issues_path.exists():
                    issues = json.loads(issues_path.read_text(encoding="utf-8"))
                    if issues:
                        with st.expander("Замечания валидации Project Context", expanded=False):
                            for issue in issues:
                                st.write(f"- {issue}")

                briefs_path = folder / "view_briefs.json"
                if briefs_path.exists():
                    with st.expander("View Briefs", expanded=False):
                        st.json(json.loads(briefs_path.read_text(encoding="utf-8")))

                brief_issues_path = folder / "view_brief_issues.json"
                if brief_issues_path.exists():
                    brief_issues = json.loads(brief_issues_path.read_text(encoding="utf-8"))
                    if brief_issues:
                        with st.expander("Замечания валидации View Briefs", expanded=False):
                            for issue in brief_issues:
                                st.write(f"- {issue}")

                coverage_path = folder / "view_coverage.json"
                if coverage_path.exists():
                    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
                    with st.expander("Покрытие view-spec", expanded=False):
                        st.json(coverage)

                consistency_path = folder / "consistency.txt"
                if consistency_path.exists():
                    with st.expander("Отчёт согласованности", expanded=False):
                        st.markdown(consistency_path.read_text(encoding="utf-8"))

                dtabs = st.tabs([DTYPE_LABELS_H.get(d, d) for d in ["class", "sequence", "component", "activity"]])
                for dtab, dtype in zip(dtabs, ["class", "sequence", "component", "activity"]):
                    with dtab:
                        sub = folder / dtype
                        if not sub.exists():
                            st.caption("Не сгенерировано.")
                            continue
                        sc_left, sc_right = st.columns([1, 1])
                        with sc_left:
                            m_path = sub / "metrics.json"
                            if m_path.exists():
                                m = json.loads(m_path.read_text(encoding="utf-8"))
                                cols_m = st.columns(4)
                                for i, (key, val) in enumerate(m.items()):
                                    lbl = METRIC_LABELS.get(key, key)
                                    if isinstance(val, float):
                                        disp = f"{val}%" if "pct" in key else str(val)
                                    elif key == "syntax_valid":
                                        disp = "✓" if val else "✗"
                                    else:
                                        disp = str(val) if val is not None else "—"
                                    cols_m[i % 4].metric(lbl, disp)
                            puml_path_s = sub / "diagram.puml"
                            if puml_path_s.exists():
                                puml_text = puml_path_s.read_text(encoding="utf-8")
                                st.code(puml_text, language="text")
                                st.download_button("Скачать .puml", puml_text,
                                                   f"{dtype}.puml", "text/plain",
                                                   key=f"mbse_h_puml_{folder.name}_{dtype}",
                                                   use_container_width=True)
                            semantic_path = sub / "semantic_coverage.json"
                            if semantic_path.exists():
                                with st.expander("Trace coverage", expanded=False):
                                    st.json(json.loads(semantic_path.read_text(encoding="utf-8")))
                        with sc_right:
                            svg_path_s = sub / "diagram.svg"
                            png_path_s = sub / "diagram.png"
                            if svg_path_s.exists():
                                render_svg_preview(svg_path_s.read_bytes())
                                with open(svg_path_s, "rb") as f:
                                    st.download_button("Скачать .svg", f.read(),
                                                       f"{dtype}.svg", "image/svg+xml",
                                                       key=f"mbse_h_svg_{folder.name}_{dtype}",
                                                       use_container_width=True)
                            elif png_path_s.exists():
                                st.image(str(png_path_s), use_container_width=True)
                            else:
                                st.caption("SVG/PNG не сохранён.")
                            if png_path_s.exists():
                                with open(png_path_s, "rb") as f:
                                    st.download_button("Скачать .png", f.read(),
                                                       f"{dtype}.png", "image/png",
                                                       key=f"mbse_h_png_{folder.name}_{dtype}",
                                                       use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка: Аналитика
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("Аналитика по истории генераций")

    items = load_history()

    if not items:
        st.info("История пуста. Сгенерируйте первую диаграмму!")
    else:
        # ── Собираем DataFrame динамически ────────────────────────────────────
        rows = []
        for folder in items:
            meta_path    = folder / "meta.json"
            metrics_path = folder / "metrics.json"
            if not meta_path.exists() or not metrics_path.exists():
                continue
            meta    = json.loads(meta_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            mode_raw = meta.get("pipeline_mode", "multi")
            mode_label = {"single": "1 агент", "multi_slim": "5 агентов slim", "multi": "5 агентов"}.get(mode_raw, mode_raw)
            row = {
                "Дата":              meta.get("timestamp", ""),
                "Тип":               meta.get("diagram_type", ""),
                "Модель":            meta.get("model", ""),
                "Режим":             mode_label,
                "Время (с)":         meta.get("generation_time_sec"),
                "Prompt tokens":     meta.get("prompt_tokens"),
                "Completion tokens": meta.get("completion_tokens"),
                "Total tokens":      meta.get("total_tokens"),
                "Стоимость ($)":     meta.get("cost_usd"),
            }
            row.update(metrics)   # все метрики как есть
            rows.append(row)

        df = pd.DataFrame(rows)

        # Переименовываем колонки метрик через METRIC_LABELS
        rename_map = {k: METRIC_LABELS[k] for k in df.columns if k in METRIC_LABELS}
        df_display = df.rename(columns=rename_map)

        # ── Фильтр по режиму ──────────────────────────────────────────────────
        modes_available = ["все"] + sorted(df["Режим"].unique().tolist())
        mode_filter = st.radio("Показать режим", modes_available, horizontal=True)
        df_filtered_mode = df if mode_filter == "все" else df[df["Режим"] == mode_filter]
        df_display_filtered = df_display if mode_filter == "все" else df_display[df["Режим"] == mode_filter]

        st.divider()

        # ── Сводные метрики ────────────────────────────────────────────────────
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Всего запусков",   len(df_filtered_mode))
        bc = df_filtered_mode.get("block_count", pd.Series(dtype=float)).dropna()
        a2.metric("Среднее блоков",   round(bc.mean(), 1) if not bc.empty else "—")
        rc = df_filtered_mode.get("relation_count", pd.Series(dtype=float)).dropna()
        a3.metric("Среднее связей",   round(rc.mean(), 1) if not rc.empty else "—")
        cp = df_filtered_mode.get("entity_coverage_pct", pd.Series(dtype=float)).dropna()
        a4.metric("Среднее покрытие", f"{round(cp.mean(), 1)}%" if not cp.empty else "—")
        times = df_filtered_mode["Время (с)"].dropna()
        a5.metric("Среднее время",    f"{round(times.mean(), 1)} с" if not times.empty else "—")

        st.divider()

        # ── Таблица ────────────────────────────────────────────────────────────
        st.markdown("**Все запуски**")
        st.dataframe(df_display_filtered, use_container_width=True, hide_index=True)

        st.divider()

        # ── Графики ────────────────────────────────────────────────────────────

        # 1. Сравнение режимов
        st.markdown("**Сравнение режимов генерации**")
        mode_cols = ["block_count", "entity_coverage_pct", "excess_elements_pct"]
        available_mode_cols = [c for c in mode_cols if c in df_filtered_mode.columns]
        if df_filtered_mode["Режим"].nunique() >= 1 and available_mode_cols:
            mode_chart = (
                df_filtered_mode.groupby("Режим")[available_mode_cols]
                .mean()
                .rename(columns={
                    "block_count": "Блоков",
                    "entity_coverage_pct": "Покрытие req (%)",
                    "excess_elements_pct": "Лишние эл. (%)",
                })
            )
            st.bar_chart(mode_chart)
        else:
            st.caption("Нет данных для сравнения режимов.")

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        # 2. Стоимость vs качество
        with chart_col1:
            st.markdown("**Стоимость vs покрытие требований**")
            scatter_cols = ["Стоимость ($)", "entity_coverage_pct", "Режим"]
            scatter_df = df_filtered_mode.copy()
            scatter_df = scatter_df.rename(columns={"entity_coverage_pct": "Покрытие (%)"})
            scatter_df = scatter_df[["Стоимость ($)", "Покрытие (%)", "Режим"]].dropna()
            if not scatter_df.empty:
                st.scatter_chart(scatter_df, x="Стоимость ($)", y="Покрытие (%)", color="Режим")
            else:
                st.caption("Нет данных о стоимости.")

        # 3. Токены по режимам
        with chart_col2:
            st.markdown("**Токены по режимам**")
            token_cols = ["Prompt tokens", "Completion tokens"]
            if all(c in df_filtered_mode.columns for c in token_cols):
                token_chart = (
                    df_filtered_mode.groupby("Режим")[token_cols]
                    .mean()
                    .dropna()
                )
                if not token_chart.empty:
                    st.bar_chart(token_chart)
                else:
                    st.caption("Нет данных о токенах.")
            else:
                st.caption("Нет данных о токенах.")

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка: Настройки промптов
# ══════════════════════════════════════════════════════════════════════════════
with tab_prompts:
    st.subheader("Настройки промптов агентов")
    st.caption(
        "Редактируйте goal и backstory каждого агента. "
        "Изменения сохраняются в prompts_config.json и применяются при следующем запуске пайплайна."
    )

    current_prompts = config.load_prompts()

    AGENT_SECTIONS = [
        ("analyst",   "Аналитик (Senior Business Analyst)"),
        ("architect", "Архитектор (Principal System Architect)"),
        ("coder",     "Кодер PlantUML (PlantUML Expert Engineer)"),
        ("critic",    "Критик (UML Diagram Critic)"),
    ]

    new_prompts: dict = {}

    st.markdown("### Роли агентов")
    for agent_key, agent_label in AGENT_SECTIONS:
        with st.expander(agent_label, expanded=True):
            agent_data = current_prompts.get(agent_key, config.DEFAULT_PROMPTS.get(agent_key, {}))
            goal_val = st.text_area(
                "Goal — цель агента",
                value=agent_data.get("goal", ""),
                height=80,
                key=f"prompt_goal_{agent_key}",
            )
            backstory_val = st.text_area(
                "Backstory — инструкции поведения",
                value=agent_data.get("backstory", ""),
                height=220,
                key=f"prompt_backstory_{agent_key}",
            )
            new_prompts[agent_key] = {"goal": goal_val, "backstory": backstory_val}

    st.divider()
    st.markdown("### Промпты кодера по типам диаграмм")
    st.caption("Задача, которую получает агент-кодер при генерации каждого типа диаграммы.")

    CODER_DTYPE_LABELS = {
        "class":     "Class — Классовая диаграмма",
        "sequence":  "Sequence — Диаграмма последовательности",
        "component": "Component — Компонентная диаграмма",
        "activity":  "Activity — Диаграмма деятельности",
    }

    current_coder_prompts = current_prompts.get("coder_prompts", config.DEFAULT_PROMPTS.get("coder_prompts", {}))
    new_coder_prompts: dict = {}

    for dtype, dtype_label in CODER_DTYPE_LABELS.items():
        with st.expander(dtype_label, expanded=False):
            default_val = config.DEFAULT_PROMPTS.get("coder_prompts", {}).get(dtype, "")
            coder_val = st.text_area(
                "Промпт задачи кодера",
                value=current_coder_prompts.get(dtype, default_val),
                height=320,
                key=f"prompt_coder_{dtype}",
            )
            new_coder_prompts[dtype] = coder_val

    new_prompts["coder_prompts"] = new_coder_prompts

    st.divider()
    btn_save, btn_reset = st.columns(2)

    with btn_save:
        if st.button("Сохранить настройки", type="primary", use_container_width=True, key="prompts_save"):
            config.save_prompts(new_prompts)
            st.success("Настройки сохранены в prompts_config.json")

    with btn_reset:
        if st.button("Сбросить к значениям по умолчанию", use_container_width=True, key="prompts_reset"):
            config.save_prompts(config.DEFAULT_PROMPTS)
            st.success("Настройки сброшены к значениям по умолчанию. Обновите страницу.")
            st.rerun()
