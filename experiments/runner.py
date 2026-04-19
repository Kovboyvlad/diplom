"""
Экспериментальный раннер: автоматический прогон тест-кейсов через пайплайн агентов.

Запуск:
    source venv/Scripts/activate
    python experiments/runner.py

Конфигурация задаётся константами ниже.
Каждый запуск сохраняется в history/ (с experiment_id в meta.json).
Итоговая таблица метрик пишется в experiments/results/{EXPERIMENT_ID}/summary.csv.
"""

import sys
import csv
import json
import time
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from agents.pipeline import run_pipeline
from utils.diagram import clean_output, validate_puml, render_png
from utils.metrics import compute_metrics, save_history

# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА — редактируй здесь
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_ID = "exp_001"

# Папка с тест-кейсами: "cases" (структурированные) или "cases_natural" (живой текст)
CASES_DIR = Path(__file__).parent / "cases_natural"

# Какие файлы запускать (None = все .txt в папке)
CASE_FILES: list[str] | None = None
# Пример: CASE_FILES = ["01_library.txt", "02_atm.txt"]

# Типы диаграмм для прогона
DIAGRAM_TYPES = ["class", "sequence", "component", "activity"]

# Модели: {отображаемое_имя: ID для LiteLLM}
MODELS = {
    "gpt-5":   "gpt-5",
    "gemini":  "gemini/gemini-3-flash-preview",
    # "claude": "anthropic/claude-3-5-haiku-20241022",
}

# Пропускать уже выполненные комбинации при повторном запуске
SKIP_EXISTING = True

# ══════════════════════════════════════════════════════════════════════════════


def _results_dir() -> Path:
    d = Path(__file__).parent / "results" / EXPERIMENT_ID
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_existing(results_dir: Path) -> set[tuple]:
    """Возвращает множество уже выполненных комбинаций (case, diagram_type, model)."""
    summary = results_dir / "summary.csv"
    if not summary.exists():
        return set()
    done = set()
    with open(summary, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["case"], row["diagram_type"], row["model"]))
    return done


def _append_row(results_dir: Path, row: dict) -> None:
    summary = results_dir / "summary.csv"
    write_header = not summary.exists()
    with open(summary, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    config.setup()

    results_dir = _results_dir()
    done = _load_existing(results_dir) if SKIP_EXISTING else set()

    # Сохраняем конфиг эксперимента
    cfg_path = results_dir / "config.json"
    if not cfg_path.exists():
        cfg = {
            "experiment_id":  EXPERIMENT_ID,
            "cases_dir":      str(CASES_DIR),
            "diagram_types":  DIAGRAM_TYPES,
            "models":         list(MODELS.keys()),
            "started_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Список кейсов
    if CASE_FILES:
        cases = [CASES_DIR / f for f in CASE_FILES]
    else:
        cases = sorted(CASES_DIR.glob("*.txt"))

    total = len(cases) * len(DIAGRAM_TYPES) * len(MODELS)
    done_count = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"Эксперимент: {EXPERIMENT_ID}")
    print(f"Кейсов: {len(cases)} | Типов: {len(DIAGRAM_TYPES)} | Моделей: {len(MODELS)}")
    print(f"Всего запусков: {total}")
    print(f"Результаты: {results_dir}")
    print(f"{'='*60}\n")

    for case_path in cases:
        requirements = case_path.read_text(encoding="utf-8")
        case_name = case_path.stem

        for diagram_type in DIAGRAM_TYPES:
            for model_name, model_id in MODELS.items():

                key = (case_name, diagram_type, model_name)
                if SKIP_EXISTING and key in done:
                    print(f"  [skip] {case_name} / {diagram_type} / {model_name}")
                    skipped += 1
                    continue

                print(f"\n[{done_count + skipped + 1}/{total}] {case_name} / {diagram_type} / {model_name}")

                try:
                    t_start = time.time()
                    raw, critique = run_pipeline(requirements, diagram_type, model_id)
                    gen_time = time.time() - t_start

                    puml = clean_output(raw)

                    if not validate_puml(puml):
                        print("  FAIL: невалидный PlantUML-код")
                        row = {
                            "case": case_name, "diagram_type": diagram_type,
                            "model": model_name, "status": "invalid_puml",
                            "generation_time_sec": round(gen_time, 1),
                        }
                        _append_row(results_dir, row)
                        continue

                    metrics = compute_metrics(puml, requirements, diagram_type)
                    png = render_png(puml)

                    history_folder = save_history(
                        requirements, diagram_type, model_id, puml, png,
                        metrics, critique, None, gen_time,
                        experiment_id=EXPERIMENT_ID,
                    )

                    row = {
                        "case":         case_name,
                        "diagram_type": diagram_type,
                        "model":        model_name,
                        "status":       "ok",
                        "generation_time_sec": round(gen_time, 1),
                        "history_folder": history_folder.name,
                        **metrics,
                    }
                    _append_row(results_dir, row)
                    done_count += 1

                    print(f"  OK  | блоков={metrics.get('block_count')} "
                          f"связей={metrics.get('relation_count')} "
                          f"покрытие={metrics.get('entity_coverage_pct')}% "
                          f"время={round(gen_time, 1)}с")

                except KeyboardInterrupt:
                    print("\n\nПрерван пользователем. Прогресс сохранён.")
                    return
                except Exception as e:
                    print(f"  ERROR: {e}")
                    row = {
                        "case": case_name, "diagram_type": diagram_type,
                        "model": model_name, "status": f"error: {e}",
                        "generation_time_sec": None,
                    }
                    _append_row(results_dir, row)

    print(f"\n{'='*60}")
    print(f"Готово: {done_count} успешно, {skipped} пропущено")
    print(f"Сводная таблица: {results_dir / 'summary.csv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
