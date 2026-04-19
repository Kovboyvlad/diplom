"""
CLI-режим: запускает пайплайн с требованиями из REQUIREMENTS_TEXT и сохраняет diagram.puml / diagram.png.
Для интерактивного режима используй: streamlit run app.py
"""
import time
from pathlib import Path

import config
from agents.pipeline import run_pipeline
from utils.diagram import clean_output, validate_puml, render_png

REQUIREMENTS_TEXT = """\
ПРОЕКТ: Автоматизированная система управления складом (Smart Warehouse WMS).

1. ЦЕЛЬ СИСТЕМЫ:
Обеспечить автоматическое распределение задач между сервером и автономными
роботами-погрузчиками для перемещения паллет без участия человека.
"""

DIAGRAM_TYPE = "class"  # class | sequence | component | activity


def main():
    config.setup()
    print("### Запуск AI-агентов (CLI-режим)... ###")

    t_start = time.time()
    raw, critique = run_pipeline(REQUIREMENTS_TEXT, DIAGRAM_TYPE)
    generation_time_sec = time.time() - t_start
    puml_code = clean_output(raw)

    if not validate_puml(puml_code):
        print("[ОШИБКА] Сгенерированный код не содержит @startuml/@enduml. Попробуйте запустить снова.")
        return

    Path("diagram.puml").write_text(puml_code, encoding="utf-8")
    print("\nФайл diagram.puml сохранён.")

    print("Рендеринг PNG...")
    png = render_png(puml_code)
    if png:
        Path("diagram.png").write_bytes(png)
        print("Файл diagram.png сохранён.")
    else:
        print("[ПРЕДУПРЕЖДЕНИЕ] Не удалось отрисовать PNG (проверьте соединение).")

    print("\n### РЕЗУЛЬТАТ ###")
    print(puml_code)

    print(f"\nВремя генерации: {generation_time_sec:.1f} с")

    print("\n### ЗАМЕЧАНИЯ КРИТИКА ###")
    print(critique)


if __name__ == "__main__":
    main()
