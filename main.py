"""
CLI-режим: запускает пайплайн с требованиями из REQUIREMENTS_TEXT и сохраняет diagram.puml / diagram.png.
Для интерактивного режима используй: streamlit run app.py
"""
from pathlib import Path

import config
from agents.pipeline import run_pipeline
from utils.diagram import clean_output, validate_puml, render_png

REQUIREMENTS_TEXT = """\
ПРОЕКТ: Автоматизированная система управления складом (Smart Warehouse WMS).

1. ЦЕЛЬ СИСТЕМЫ:
Обеспечить автоматическое распределение задач между сервером и автономными
роботами-погрузчиками для перемещения паллет без участия человека.

2. КЛЮЧЕВЫЕ ЭЛЕМЕНТЫ (СУЩНОСТИ):
- WMS-Сервер (Central System): Главный управляющий модуль. Хранит БД заказов и карту склада.
- Робот-AGV (Automated Guided Vehicle): Автономный погрузчик. Параметры: ID, Статус, BatteryLevel.
- Зона Приемки (Inbound Zone): Место появления новых грузов. Оборудована датчиком.
- Стеллаж (Storage Rack): Конечное место хранения груза. Координаты (X, Y).
- Зарядная Станция (Charging Station): Место для подзарядки роботов.

3. ФУНКЦИОНАЛЬНАЯ ЛОГИКА:
- Инициация: Датчик в Зоне Приемки отправляет сигнал "NewItem" на WMS-Сервер.
- Распределение: Сервер выбирает ближайшего свободного робота с зарядом > 20%.
- Выполнение: Сервер передаёт роботу координаты груза и целевого Стеллажа.
- Исключение: Если заряд <= 20%, Робот едет на Зарядную Станцию (статус ServiceMode).
"""

DIAGRAM_TYPE = "class"  # class | sequence | component | activity


def main():
    config.setup()
    print("### Запуск AI-агентов (CLI-режим)... ###")

    raw = run_pipeline(REQUIREMENTS_TEXT, DIAGRAM_TYPE)
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


if __name__ == "__main__":
    main()
