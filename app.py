import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from agents.pipeline import run_pipeline, run_pipeline_slim, run_single_agent, run_evaluation
from utils.diagram import clean_output, validate_puml, render_png
from utils.file_reader import read_uploaded_file
from utils.metrics import compute_metrics, save_history, load_history

config.setup()


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

st.set_page_config(
    page_title="AI Diagram Generator",
    page_icon=None,
    layout="wide",
)

st.title("AI-генератор архитектурных диаграмм")
st.caption("Введите требования к системе — ИИ-агенты сгенерируют PlantUML-диаграмму.")

tab_generate, tab_history, tab_analytics = st.tabs(
    ["Генерация", "История", "Аналитика"]
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
                        raw_result, critique, usage = run_single_agent(requirements, diagram_type, model_id)
                    elif use_slim:
                        raw_result, critique, usage = run_pipeline_slim(requirements, diagram_type, model_id)
                    else:
                        raw_result, critique, usage = run_pipeline(requirements, diagram_type, model_id)
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
                        png_bytes = render_png(puml_code)

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

                        # ── Оценка агента-оценщика ────────────────────────────
                        if evaluation:
                            with st.expander("Анализ результатов (агент-оценщик)", expanded=True):
                                st.markdown(evaluation)

                        # ── Диаграмма ─────────────────────────────────────────
                        if png_bytes:
                            st.image(png_bytes, use_container_width=True)
                        else:
                            st.warning("PNG не отрисован — вставь код ниже на plantuml.com")

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
                    if png_path.exists():
                        st.image(str(png_path), use_container_width=True)
                        with open(png_path, "rb") as f:
                            st.download_button("Скачать .png", f.read(),
                                               "diagram.png", "image/png",
                                               key=f"dl_png_{folder.name}",
                                               use_container_width=True)
                    else:
                        st.caption("PNG не сохранён.")

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
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Среднее кол-во блоков по моделям**")
            if df_filtered_mode["Модель"].nunique() > 1 and "block_count" in df_filtered_mode.columns:
                chart_data = df_filtered_mode.groupby("Модель")["block_count"].mean().rename("Блоков")
                st.bar_chart(chart_data)
            else:
                st.caption("Недостаточно данных (нужно ≥ 2 разных модели).")

        with chart_col2:
            st.markdown("**Среднее покрытие требований по типам диаграмм**")
            if df_filtered_mode["Тип"].nunique() > 1 and "entity_coverage_pct" in df_filtered_mode.columns:
                chart_data2 = df_filtered_mode.groupby("Тип")["entity_coverage_pct"].mean().rename("Покрытие %")
                st.bar_chart(chart_data2)
            else:
                st.caption("Недостаточно данных (нужно ≥ 2 разных типа).")
