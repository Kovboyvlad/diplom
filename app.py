import os
from pathlib import Path

import streamlit as st

import config
from agents.pipeline import run_pipeline, CODER_PROMPTS
from utils.diagram import clean_output, validate_puml, render_png
from utils.file_reader import read_uploaded_file

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

# ─── Интерфейс ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Diagram Generator",
    page_icon="📐",
    layout="wide",
)

st.title("📐 AI-генератор архитектурных диаграмм")
st.caption("Введите требования к системе — ИИ-агенты сгенерируют PlantUML-диаграмму.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Входные данные")

    # Загрузка файла
    uploaded = st.file_uploader(
        "Загрузить требования из файла (.txt или .pdf)",
        type=["txt", "pdf"],
        help="Текст из файла будет вставлен в поле ниже.",
    )

    default_text = EXAMPLE_REQUIREMENTS
    if uploaded is not None:
        try:
            default_text = read_uploaded_file(uploaded)
            st.success(f"Файл «{uploaded.name}» успешно загружен.")
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")

    requirements = st.text_area(
        "Текст требований",
        value=default_text,
        height=320,
        placeholder="Опишите систему: сущности, их роли и логику взаимодействия...",
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

    generate_btn = st.button("Сгенерировать", type="primary", use_container_width=True)

with col_right:
    st.subheader("Результат")

    if generate_btn:
        if not requirements.strip():
            st.warning("Пожалуйста, введите текст требований.")
        else:
            status_placeholder = st.empty()
            progress = st.progress(0, text="Запуск агентов...")

            try:
                status_placeholder.info("Агент 1/3: Бизнес-аналитик анализирует требования...")
                progress.progress(10, text="Анализ требований...")

                raw_result = run_pipeline(requirements, diagram_type, model_id)

                progress.progress(80, text="Очистка и валидация...")
                status_placeholder.info("Обрабатываю результат...")

                puml_code = clean_output(raw_result)

                if not validate_puml(puml_code):
                    st.error(
                        "Агенты не смогли сгенерировать корректный PlantUML-код "
                        "(отсутствует @startuml/@enduml). Попробуйте ещё раз."
                    )
                    progress.empty()
                    status_placeholder.empty()
                else:
                    progress.progress(90, text="Рендеринг PNG...")
                    status_placeholder.info("Отправляю диаграмму на рендеринг...")

                    png_bytes = render_png(puml_code)

                    Path("diagram.puml").write_text(puml_code, encoding="utf-8")
                    if png_bytes:
                        Path("diagram.png").write_bytes(png_bytes)

                    progress.progress(100, text="Готово!")
                    status_placeholder.success("Диаграмма сгенерирована!")

                    if png_bytes:
                        st.image(png_bytes, caption="Сгенерированная диаграмма", use_container_width=True)
                    else:
                        st.warning(
                            "Не удалось отрисовать PNG (нет сети или ошибка сервера PlantUML). "
                            "PlantUML-код ниже можно вставить на plantuml.com вручную."
                        )

                    st.subheader("PlantUML-код")
                    st.code(puml_code, language="text")

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "Скачать .puml",
                            data=puml_code,
                            file_name="diagram.puml",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    with col_dl2:
                        if png_bytes:
                            st.download_button(
                                "Скачать .png",
                                data=png_bytes,
                                file_name="diagram.png",
                                mime="image/png",
                                use_container_width=True,
                            )

            except Exception as e:
                progress.empty()
                status_placeholder.empty()
                st.error(f"Ошибка при выполнении агентов: {e}")
    else:
        st.info("Заполните требования слева и нажмите «Сгенерировать».")
