import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from agents.pipeline import run_pipeline, run_evaluation
from utils.diagram import clean_output, validate_puml, render_png
from utils.file_reader import read_uploaded_file
from utils.metrics import compute_metrics, save_history, load_history

config.setup()

EXAMPLE_REQUIREMENTS = """\
ПРОЕКТ: Автоматизированная система управления складом (Smart Warehouse WMS).

1. ЦЕЛЬ СИСТЕМЫ:
Обеспечить автоматическое распределение задач между сервером и автономными
роботами-погрузчиками для перемещения паллет без участия человека.

2. КЛЮЧЕВЫЕ ЭЛЕМЕНТЫ (СУЩНОСТИ):
- WMS-Сервер (Central System): Главный управляющий модуль.
- Робот-AGV: Автономный погрузчик. Параметры: ID, Статус, BatteryLevel.
- Зона Приемки (Inbound Zone): Место появления новых грузов. Есть датчик.
- Стеллаж (Storage Rack): Конечное место хранения. Координаты (X, Y).
- Зарядная Станция (Charging Station): Место для подзарядки роботов.

3. ЛОГИКА:
- Датчик в Зоне Приемки отправляет сигнал "NewItem" на WMS-Сервер.
- Сервер выбирает ближайшего свободного робота с зарядом > 20%.
- Если заряд <= 20%, робот едет на Зарядную Станцию (статус ServiceMode).
"""

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
    # Component
    # (interface_count, package_count уже есть выше)
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

tab_generate, tab_history, tab_analytics, tab_experiments = st.tabs(
    ["Генерация", "История", "Аналитика", "Эксперименты"]
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

        default_text = EXAMPLE_REQUIREMENTS
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
                    raw_result, critique = run_pipeline(requirements, diagram_type, model_id)
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

                        save_history(requirements, diagram_type, model_id, puml_code, png_bytes, metrics, critique, evaluation, generation_time_sec)

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
            row = {
                "Дата":    meta.get("timestamp", ""),
                "Тип":     meta.get("diagram_type", ""),
                "Модель":  meta.get("model", ""),
                "Время (с)": meta.get("generation_time_sec"),
            }
            row.update(metrics)   # все метрики как есть
            rows.append(row)

        df = pd.DataFrame(rows)

        # Переименовываем колонки метрик через METRIC_LABELS
        rename_map = {k: METRIC_LABELS[k] for k in df.columns if k in METRIC_LABELS}
        df_display = df.rename(columns=rename_map)

        # ── Сводные метрики ────────────────────────────────────────────────────
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Всего запусков",   len(df))
        bc = df.get("block_count", pd.Series(dtype=float)).dropna()
        a2.metric("Среднее блоков",   round(bc.mean(), 1) if not bc.empty else "—")
        rc = df.get("relation_count", pd.Series(dtype=float)).dropna()
        a3.metric("Среднее связей",   round(rc.mean(), 1) if not rc.empty else "—")
        cp = df.get("entity_coverage_pct", pd.Series(dtype=float)).dropna()
        a4.metric("Среднее покрытие", f"{round(cp.mean(), 1)}%" if not cp.empty else "—")
        times = df["Время (с)"].dropna()
        a5.metric("Среднее время",    f"{round(times.mean(), 1)} с" if not times.empty else "—")

        st.divider()

        # ── Таблица ────────────────────────────────────────────────────────────
        st.markdown("**Все запуски**")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.divider()

        # ── Графики ────────────────────────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Среднее кол-во блоков по моделям**")
            if df["Модель"].nunique() > 1 and "block_count" in df.columns:
                chart_data = df.groupby("Модель")["block_count"].mean().rename("Блоков")
                st.bar_chart(chart_data)
            else:
                st.caption("Недостаточно данных (нужно ≥ 2 разных модели).")

        with chart_col2:
            st.markdown("**Среднее покрытие требований по типам диаграмм**")
            if df["Тип"].nunique() > 1 and "entity_coverage_pct" in df.columns:
                chart_data2 = df.groupby("Тип")["entity_coverage_pct"].mean().rename("Покрытие %")
                st.bar_chart(chart_data2)
            else:
                st.caption("Недостаточно данных (нужно ≥ 2 разных типа).")

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка: Эксперименты
# ══════════════════════════════════════════════════════════════════════════════
with tab_experiments:
    st.subheader("Результаты экспериментов")
    st.caption("Запускай эксперименты через `python experiments/runner.py`")

    results_root = Path("experiments/results")
    experiments = sorted(results_root.iterdir()) if results_root.exists() else []
    experiments = [e for e in experiments if e.is_dir() and (e / "summary.csv").exists()]

    if not experiments:
        st.info("Нет данных. Запусти `python experiments/runner.py` чтобы начать.")
    else:
        exp_names = [e.name for e in experiments]
        selected = st.selectbox("Выбрать эксперимент", exp_names)
        exp_dir = results_root / selected

        # Конфиг
        cfg_path = exp_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Папка кейсов", Path(cfg.get("cases_dir", "")).name)
            cc2.metric("Типов диаграмм", len(cfg.get("diagram_types", [])))
            cc3.metric("Моделей", len(cfg.get("models", [])))
            cc4.metric("Запущен", cfg.get("started_at", "—")[:10])

        st.divider()

        # Загружаем summary.csv
        df_exp = pd.read_csv(exp_dir / "summary.csv")
        df_ok = df_exp[df_exp.get("status", pd.Series(["ok"] * len(df_exp))) == "ok"]

        # ── Сводная таблица ────────────────────────────────────────────────────
        st.markdown("**Все запуски**")

        # Переименовываем метрики
        rename_exp = {k: METRIC_LABELS[k] for k in df_exp.columns if k in METRIC_LABELS}
        st.dataframe(df_exp.rename(columns=rename_exp), use_container_width=True, hide_index=True)

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                "Скачать summary.csv",
                (exp_dir / "summary.csv").read_bytes(),
                "summary.csv", "text/csv",
                use_container_width=True,
            )

        st.divider()

        # ── Сравнение моделей ──────────────────────────────────────────────────
        if not df_ok.empty and "model" in df_ok.columns:
            st.markdown("**Сравнение моделей**")

            numeric_cols = [c for c in df_ok.select_dtypes("number").columns
                            if c not in ("status",)]
            metric_choice = st.selectbox(
                "Метрика для сравнения",
                options=numeric_cols,
                format_func=lambda k: METRIC_LABELS.get(k, k),
            )

            # Фильтр по типу диаграммы
            types_available = df_ok["diagram_type"].unique().tolist()
            type_filter = st.selectbox("Тип диаграммы", ["все"] + types_available)
            df_filtered = df_ok if type_filter == "все" else df_ok[df_ok["diagram_type"] == type_filter]

            if "case" in df_filtered.columns and "model" in df_filtered.columns:
                pivot = (
                    df_filtered.groupby(["case", "model"])[metric_choice]
                    .mean()
                    .unstack("model")
                )
                st.markdown(f"Среднее `{METRIC_LABELS.get(metric_choice, metric_choice)}` по кейсам и моделям:")
                st.dataframe(pivot.round(2), use_container_width=True)
                st.bar_chart(pivot)
