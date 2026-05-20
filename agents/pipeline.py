import json
import re

from crewai import Agent, Task, Crew, Process, LLM
import config as _cfg
from utils.diagram import clean_output, repair_puml

try:
    from litellm import cost_per_token as litellm_cost_per_token
except ImportError:
    litellm_cost_per_token = None

def _extract_usage(crew, model: str) -> dict:
    try:
        u = crew.usage_metrics
        prompt_tokens     = int(u.prompt_tokens)
        completion_tokens = int(u.completion_tokens)
        total_tokens      = int(u.total_tokens)
    except Exception:
        return {"prompt_tokens": None, "completion_tokens": None,
                "total_tokens": None, "cost_usd": None}
    try:
        if litellm_cost_per_token is None:
            raise ImportError
        p_cost, c_cost = litellm_cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        cost_usd = round(p_cost + c_cost, 6)
    except Exception:
        cost_usd = None
    return {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
        "cost_usd":          cost_usd,
    }


def _get_prompts() -> dict:
    return _cfg.load_prompts()


CODER_PROMPTS = {
    "class": """
На основе полученной архитектуры напиши ТОЛЬКО код PlantUML — диаграмму классов.

Требования:
1. Начни с @startuml, закончи @enduml.
2. Каждый класс содержит атрибуты с типами данных (+ name: Type) и методы.
3. Каждый статус и перечисление — отдельный `enum` с полным списком значений.
4. Используй `interface` для точек взаимодействия между подсистемами.
5. Группируй связанные классы в `package "Название" { }`.
6. Типы связей:
   - `<|--` наследование
   - `*--` композиция (часть не существует без целого)
   - `o--` агрегация (часть может существовать отдельно)
   - `..>` зависимость
   - `-->` направленная ассоциация
7. Добавь `note right of ClassName: текст` для ключевых бизнес-ограничений.
8. НЕ пиши никаких пояснений — только код от @startuml до @enduml.
""",
    "sequence": """
На основе полученной архитектуры напиши ТОЛЬКО код PlantUML — диаграмму последовательности.

Требования:
1. Начни с @startuml, закончи @enduml.
2. Покажи всех участников взаимодействия только через `participant` или `actor`.
   НЕ используй `external`, `component`, `class`, `database` в sequence-диаграмме.
3. Покажи полный основной сценарий и сценарий с ошибкой/исключением через `alt "..." ... else "..." ... end`.
4. Используй `loop "условие"` для повторяющихся действий.
5. Используй `activate` / `deactivate` для участников во время обработки.
6. Добавь `note over participant: текст` для ключевых событий и решений.
7. Синхронные вызовы: `->`, ответы: `-->`, асинхронные: `->>`.
8. НЕ пиши никаких пояснений — только код от @startuml до @enduml.
""",
    "component": """
На основе полученной архитектуры напиши ТОЛЬКО код PlantUML — компонентную диаграмму.

Требования:
1. Начни с @startuml, закончи @enduml.
2. Группируй компоненты в `package` по архитектурным слоям (например: Presentation, Business Logic, Data, External).
3. Используй `interface "Название" as alias` для интерфейсов между компонентами.
4. Используй `database "Название"` для хранилищ данных.
5. Используй `queue "Название"` для очередей и шин сообщений где уместно.
6. Для надёжного рендеринга используй связи только вида `A --> B : label`.
   НЕ используй `--(`, `--)`, `actor`, `class`, `enum` в компонентной диаграмме.
7. Все люди и организации показываются как `component "Name" as Alias <<actor>>`.
8. НЕ пиши атрибуты и методы внутри component. Компонентная диаграмма показывает модули и зависимости, не классы.
9. НЕ используй квадратные компоненты `[Name]`; объявляй каждый компонент явно: `component "Name" as Alias`.
10. НЕ пиши никаких пояснений — только код от @startuml до @enduml.
""",
    "activity": """
На основе полученной архитектуры напиши ТОЛЬКО код PlantUML — диаграмму деятельности.

Требования:
1. Начни с @startuml, закончи @enduml.
2. Используй swimlane для каждого актора/подсистемы: `|Название|` перед каждым действием.
3. Покажи все ветвления: `if "условие" then (да) ... else (нет) ... endif`.
4. Покажи параллельные процессы: `fork ... fork again ... end fork`.
5. Используй современный синтаксис PlantUML activity: `start`, `:Действие;`, `stop`.
   НЕ используй старый синтаксис `(*) -->`.
6. Каждое действие пиши строго как `:Название действия;`.
7. НЕ используй стрелки между именованными действиями вида `Action --> OtherAction`.
8. Заметки пиши в форме `note right ... end note`, не однострочно.
9. НЕ пиши никаких пояснений — только код от @startuml до @enduml.
""",
}


MBSE_CODER_PROMPTS = {
    "class": """
You are generating ONE diagram inside an MBSE UML diagram set.
Return ONLY PlantUML code from @startuml to @enduml.

Generate a UML CLASS diagram.

Hard syntax rules:
1. Use only valid PlantUML class-diagram syntax.
2. Declare domain objects as class, enum, or interface only.
3. Put attributes and methods only inside class blocks.
4. Relationship labels must not end with stray characters such as >.
   Correct: Guest "1" -- "0..*" Room : books
   Wrong:   Guest "1" -- "0..*" Room : books >
5. Use multiplicities only in quotes near relation ends.
6. Business rules may be notes.
7. Do not output markdown or explanation.

Structural coverage rules:
1. A class diagram is a structural model, not a text summary.
2. Facts about domain objects, actors/roles, states, data variants, external systems, interfaces,
   relationships, attributes, and system operations must be represented as class/interface/enum
   declarations, attributes, methods, or UML relationships.
3. Do not satisfy structural facts only with PlantUML notes. A note may clarify a structural element,
   but it must not be the only representation of the fact.
4. External systems and integration points must be modeled as interfaces or boundary classes with
   dependencies/associations to the domain or system core.
5. Scenario steps relevant to the class view should become operations on the responsible class/interface
   or relationships between classes; avoid narrative note dumps.
6. Use notes only for constraints, business rules, non-functional requirements, or genuinely ambiguous
   facts that cannot be expressed structurally.
7. Put trace comments for structural facts immediately above the structural UML element, not above a note.
""",
    "sequence": """
You are generating ONE diagram inside an MBSE UML diagram set.
Return ONLY PlantUML code from @startuml to @enduml.

Generate a UML SEQUENCE diagram.

Hard syntax rules:
1. Declare participants only with actor or participant.
2. Never use class, component, database, external, package, enum in a sequence diagram.
3. Use ->, -->, ->> messages.
4. Use alt/else/end for alternatives and exceptions.
5. Use loop/end only when repeated behavior exists.
6. Do not output markdown or explanation.

Behavioral coverage rules:
1. A sequence diagram is an interaction model, not a text summary.
2. Facts about actors, participants, scenario steps, alternatives, exceptions, external calls,
   requests, responses, approvals, denials, and repeated interactions must be represented as
   actor/participant declarations, messages, alt/else blocks, loop blocks, or activation spans.
3. Do not satisfy interaction facts only with PlantUML notes. A note may clarify a message or branch,
   but it must not be the only representation of the fact.
4. Put trace comments for interaction facts immediately above the participant, message, alt/else,
   loop, or activation element that represents the fact.
""",
    "component": """
You are generating ONE diagram inside an MBSE UML diagram set.
Return ONLY PlantUML code from @startuml to @enduml.

Generate a UML COMPONENT diagram.

Hard syntax rules:
1. Declare components explicitly: component "Name" as Alias
2. Declare interfaces explicitly: interface "Name" as Alias
3. Declare datastores only as database "Name" as Alias.
4. Use packages only for grouping layers.
5. Use relationships only in this form: AliasA --> AliasB : label
6. Never use --(, --), actor, class, enum, methods, attributes, or [Name] component shorthand.
7. Do not create generic infrastructure unless it is present in the requirements or Project Context.
8. Do not output markdown or explanation.

Structural coverage rules:
1. A component diagram is a dependency and integration model, not a text summary.
2. Facts about subsystems, components, external systems, interfaces, APIs, datastores, queues,
   and dependencies must be represented as component/interface/database/queue declarations
   and explicit A --> B : label relationships.
3. Do not satisfy component or integration facts only with PlantUML notes. A note may clarify a
   component or dependency, but it must not be the only representation of the fact.
4. Do not add generic infrastructure unless it is present in the requirements or Project Context.
5. Put trace comments for component facts immediately above the component, interface, datastore,
   queue, or relationship that represents the fact.
""",
    "activity": """
You are generating ONE diagram inside an MBSE UML diagram set.
Return ONLY PlantUML code from @startuml to @enduml.

Generate a UML ACTIVITY diagram.

Hard syntax rules:
1. Use ONLY modern PlantUML activity syntax.
2. Start with start and finish with stop.
3. Every action must be written as :Action;
4. Use swimlanes as |Actor or Role|.
5. Use if (...) then (...) / else (...) / endif for decisions.
6. Use fork / fork again / end fork for parallel work only when needed.
7. Never use legacy syntax: (*), (*) -->, --> (*), Action --> OtherAction.
8. Do not output markdown or explanation.

Process coverage rules:
1. An activity diagram is a workflow model, not a text summary.
2. Facts about actors/roles, process steps, decisions, alternatives, exceptions, cancellation,
   recovery, repeated work, and parallel work must be represented as swimlanes, actions,
   if/else decisions, fork blocks, start/stop, or explicit control flow.
3. Do not satisfy process facts only with PlantUML notes. A note may clarify an action or decision,
   but it must not be the only representation of the fact.
4. Put trace comments for process facts immediately above the swimlane, action, if/else, fork,
   start, or stop element that represents the fact.
""",
}


def _analysis_task_text(requirements: str, diagram_type: str) -> tuple[str, str]:
    if diagram_type == "class":
        return (
            f"Проанализируй текст требований:\n\n{requirements}\n\n"
            "Твой анализ должен содержать:\n"
            "1. Все доменные сущности с полным списком атрибутов и их типами данных\n"
            "2. Все перечисления: для каждого статуса, типа, режима — полный список значений\n"
            "3. Все методы и операции каждой сущности\n"
            "4. Все связи между сущностями с предполагаемым типом связи\n"
            "5. Бизнес-ограничения, которые нужно вынести в notes\n"
            "6. Не добавляй технические слои и сервисы, если их нет в требованиях",
            "Структурированный анализ для class diagram: сущности, атрибуты, enum, методы, связи, бизнес-ограничения.",
        )
    if diagram_type == "sequence":
        return (
            f"Проанализируй текст требований:\n\n{requirements}\n\n"
            "Твой анализ должен содержать только то, что нужно для sequence diagram:\n"
            "1. Акторы и системные участники взаимодействия\n"
            "2. Основной сценарий по шагам в правильном порядке\n"
            "3. Альтернативные потоки и исключения\n"
            "4. Повторяющиеся действия для loop\n"
            "5. Внешние системы, которым отправляются сообщения\n"
            "6. Не перечисляй атрибуты классов и enum, если они не нужны для сообщений",
            "Структурированный анализ для sequence diagram: участники, основной сценарий, alt/loop, исключения, сообщения.",
        )
    if diagram_type == "component":
        return (
            f"Проанализируй текст требований:\n\n{requirements}\n\n"
            "Твой анализ должен содержать только то, что нужно для component diagram:\n"
            "1. Подсистемы, модули и внешние системы\n"
            "2. Ответственность каждого компонента\n"
            "3. Интерфейсы/API между компонентами\n"
            "4. Хранилища данных, если они явно следуют из требований\n"
            "5. Зависимости между компонентами\n"
            "6. Не описывай атрибуты и методы классов; компонентная диаграмма не является class diagram",
            "Структурированный анализ для component diagram: компоненты, интерфейсы, хранилища, зависимости, слои.",
        )
    if diagram_type == "activity":
        return (
            f"Проанализируй текст требований:\n\n{requirements}\n\n"
            "Твой анализ должен содержать только то, что нужно для activity diagram:\n"
            "1. Участники/swimlanes процесса\n"
            "2. Основной бизнес-процесс по действиям\n"
            "3. Точки принятия решений для if/else\n"
            "4. Параллельные действия для fork, если они явно есть\n"
            "5. Альтернативные потоки, отмены и исключения\n"
            "6. Не описывай классы, атрибуты, методы и enum",
            "Структурированный анализ для activity diagram: swimlanes, действия, решения, параллельность, исключения.",
        )
    return (
        f"Проанализируй текст требований для UML-диаграммы типа {diagram_type}:\n\n{requirements}",
        "Структурированный анализ требований.",
    )


def _architecture_task_text(diagram_type: str) -> tuple[str, str]:
    if diagram_type == "class":
        return (
            "На основе анализа спроектируй class-view архитектуру системы.\n\n"
            "Архитектура должна содержать:\n"
            "1. Классы с атрибутами (с типами) и методами\n"
            "2. Для каждой связи — явный тип: композиция / агрегация / зависимость / реализация / ассоциация\n"
            "3. Группировку классов по пакетам\n"
            "4. Отдельные enum для всех статусов и перечислений\n"
            "5. Интерфейсы только для реальных точек взаимодействия\n"
            "6. Бизнес-ограничения для notes",
            "Class-view архитектура: классы, атрибуты, методы, enum, интерфейсы, связи, пакеты.",
        )
    if diagram_type == "sequence":
        return (
            "На основе анализа спроектируй sequence-view взаимодействие.\n\n"
            "Архитектура должна содержать:\n"
            "1. Список участников: actor или participant\n"
            "2. Основной сценарий сообщений в правильном порядке\n"
            "3. alt/else блоки для альтернатив и ошибок\n"
            "4. loop блоки для повторов\n"
            "5. activate/deactivate для ключевых обработчиков\n"
            "6. Никаких классов, атрибутов, enum и компонентных слоёв",
            "Sequence-view: участники, сообщения, alt/else, loop, activate/deactivate.",
        )
    if diagram_type == "component":
        return (
            "На основе анализа спроектируй component-view архитектуру.\n\n"
            "Архитектура должна содержать:\n"
            "1. Компоненты и внешние системы\n"
            "2. Пакеты/слои, если они следуют из требований\n"
            "3. Интерфейсы/API между компонентами\n"
            "4. Базы данных/очереди только если они явно нужны\n"
            "5. Зависимости вида A --> B : label\n"
            "6. Никаких class/enum/actor, атрибутов и методов классов",
            "Component-view: компоненты, интерфейсы, слои, хранилища, зависимости.",
        )
    if diagram_type == "activity":
        return (
            "На основе анализа спроектируй activity-view бизнес-процесс.\n\n"
            "Архитектура должна содержать:\n"
            "1. Swimlane-участников\n"
            "2. Действия процесса в правильном порядке\n"
            "3. if/else решения\n"
            "4. fork/fork again для параллельности\n"
            "5. start и stop\n"
            "6. Никаких классов, атрибутов, enum и компонентных слоёв",
            "Activity-view: swimlanes, действия, решения, параллельность, start/stop.",
        )
    return (
        f"На основе анализа спроектируй UML-view типа {diagram_type}.",
        "Специализированное архитектурное описание для указанного типа диаграммы.",
    )


def run_single_agent(requirements: str, diagram_type: str = "class", model: str = "gpt-4o-mini") -> tuple[str, str, dict, dict]:
    """
    Запускает одиночного агента-кодера без Analyst и Architect (baseline для сравнения).
    Возвращает (PlantUML-код, пустую строку critique).
    """
    llm = LLM(model=model)

    coder = Agent(
        role="PlantUML Expert Engineer",
        goal=(
            f"По описанию системы напиши полный и валидный PlantUML-код для диаграммы типа '{diagram_type}'."
        ),
        backstory=(
            "Ты эксперт по PlantUML. Твои правила абсолютны:\n"
            "- ты пишешь ТОЛЬКО код — никакого текста, пояснений или markdown вокруг\n"
            "- каждый атрибут имеет тип, каждый метод имеет скобки\n"
            "- диаграмма должна отражать все ключевые сущности из описания"
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    task = Task(
        description=(
            f"На основе описания системы напиши PlantUML-код — диаграмму типа '{diagram_type}'.\n\n"
            f"ОПИСАНИЕ СИСТЕМЫ:\n{requirements}\n\n"
            + CODER_PROMPTS[diagram_type]
        ),
        expected_output="Только код PlantUML — от @startuml до @enduml, без какого-либо текста вокруг.",
        agent=coder,
    )

    crew = Crew(
        agents=[coder],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )

    crew.kickoff()
    usage = _extract_usage(crew, model)
    return task.output.raw, "", usage, {}


def run_pipeline(requirements: str, diagram_type: str = "class", model: str = "gpt-4o-mini") -> tuple[str, str, dict, dict]:
    """
    Запускает четырёхагентный пайплайн CrewAI.

    :param requirements: текст требований к системе
    :param diagram_type: тип диаграммы ('class', 'sequence', 'component', 'activity')
    :param model: ID модели для LiteLLM/CrewAI
    :return: кортеж (PlantUML-код диаграммы, замечания критика)
    """
    llm = LLM(model=model)
    p = _get_prompts()

    analyst = Agent(
        role="Senior Business Analyst",
        goal=p.get("analyst", {}).get("goal", "Точно извлечь из требований только то, что явно в них написано: сущности, их атрибуты, статусы, действия и взаимодействия между ними."),
        backstory=p.get("analyst", {}).get("backstory", "Ты опытный бизнес-аналитик. Твой главный принцип: строгое следование тексту требований.\n- ты извлекаешь только то, что явно упомянуто — не додумываешь и не расширяешь\n- ты НЕ добавляешь инфраструктуру, сервисы, события или исключения если они явно не указаны в требованиях\nТвоя задача — точная модель того что написано, не архитектурное видение системы."),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    architect = Agent(
        role="Principal System Architect",
        goal=p.get("architect", {}).get("goal", "Спроектировать архитектуру строго на основе анализа: только те блоки и связи, которые следуют из требований."),
        backstory=p.get("architect", {}).get("backstory", "Ты главный архитектор с экспертизой в UML и SysML. Твои принципы:\n- ты проектируешь ТОЛЬКО то, что есть в требованиях\n- тип каждой связи указывается явно: композиция, агрегация, зависимость, ассоциация\nКоличество блоков определяется требованиями — не больше и не меньше."),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    coder = Agent(
        role="PlantUML Expert Engineer",
        goal=p.get("coder", {}).get("goal", f"Написать полный и валидный код PlantUML для диаграммы типа '{diagram_type}'. Диаграмма должна отражать всю архитектуру без упрощений."),
        backstory=p.get("coder", {}).get("backstory", "Ты эксперт по PlantUML. Твои правила абсолютны:\n- ты пишешь ТОЛЬКО код — никакого текста, пояснений или markdown вокруг\n- ты переносишь в код ВСЕ блоки из архитектуры, ни один не пропускаешь\n- каждый атрибут имеет тип, каждый метод имеет скобки"),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    critic = Agent(
        role="UML Diagram Critic",
        goal=p.get("critic", {}).get("goal", "Критически оценить сгенерированную PlantUML-диаграмму: выявить пропущенные элементы, ошибки нотации и слабые места промптов."),
        backstory=p.get("critic", {}).get("backstory", "Ты строгий эксперт по UML и системному анализу. Твоя задача — не переписывать, а объективно оценить результат работы предыдущих агентов:\n- ты сравниваешь диаграмму с исходными требованиями и находишь пропуски\n- ты проверяешь корректность UML-нотации для данного типа диаграммы\nТы не хвалишь — ты указываешь на проблемы и пути их устранения."),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    analysis_description, analysis_expected = _analysis_task_text(requirements, diagram_type)
    task1 = Task(
        description=analysis_description,
        expected_output=analysis_expected,
        agent=analyst,
    )

    architecture_description, architecture_expected = _architecture_task_text(diagram_type)
    task2 = Task(
        description=architecture_description,
        expected_output=architecture_expected,
        agent=architect,
    )

    _coder_prompt = p.get("coder_prompts", {}).get(diagram_type) or CODER_PROMPTS[diagram_type]
    task3 = Task(
        description=_coder_prompt,
        expected_output="Только код PlantUML — от @startuml до @enduml, без какого-либо текста вокруг.",
        agent=coder,
    )

    task4 = Task(
        description=(
            f"Ты получил исходные требования к системе и сгенерированный PlantUML-код диаграммы типа '{diagram_type}'.\n\n"
            f"ИСХОДНЫЕ ТРЕБОВАНИЯ:\n{requirements}\n\n"
            "Проверь диаграмму по следующим критериям и выдай структурированный отчёт в формате Markdown:\n\n"
            "## 1. Пропущенные элементы\n"
            "Перечисли сущности, связи или атрибуты из требований, которые не попали в диаграмму.\n\n"
            "## 2. Ошибки UML-нотации\n"
            "Укажи конкретные строки или блоки с некорректным использованием нотации PlantUML "
            f"для диаграммы типа '{diagram_type}'.\n\n"
            "## 3. Рекомендации по улучшению\n"
            "Конкретные предложения:\n"
            "- что добавить или уточнить в промптах агентов (Analyst, Architect, Coder)\n"
            "- как улучшить входные требования для более качественного результата\n"
            "- замечания по выбору модели (если применимо)\n\n"
            "## 4. Общая оценка\n"
            "Кратко: насколько диаграмма соответствует требованиям (0–10) и почему."
        ),
        expected_output=(
            "Структурированный Markdown-отчёт с разделами: пропущенные элементы, "
            "ошибки нотации, рекомендации по улучшению промптов, общая оценка."
        ),
        agent=critic,
    )

    task5 = Task(
        description=(
            f"Ты получил оригинальный PlantUML-код диаграммы типа '{diagram_type}' "
            "и список замечаний от агента-критика.\n\n"
            "Исправь диаграмму, устранив ВСЕ замечания критика:\n"
            "- добавь пропущенные сущности, атрибуты и связи\n"
            "- исправь ошибки UML-нотации\n"
            "- не удаляй то, что уже было правильно\n\n"
            "НЕ пиши никаких пояснений — только исправленный код от @startuml до @enduml."
        ),
        expected_output="Только исправленный код PlantUML — от @startuml до @enduml, без какого-либо текста вокруг.",
        agent=coder,
    )

    crew = Crew(
        agents=[analyst, architect, coder, critic],
        tasks=[task1, task2, task3, task4, task5],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )

    crew.kickoff()
    diagram = task5.output.raw
    critique = task4.output.raw
    usage = _extract_usage(crew, model)
    intermediates = {
        "analyst":    task1.output.raw,
        "architect":  task2.output.raw,
        "coder_v1":   task3.output.raw,
    }
    return diagram, critique, usage, intermediates


def run_pipeline_slim(
    requirements: str,
    diagram_type: str = "class",
    model: str = "gpt-4o-mini",
    prompt_profile: str = "default",
) -> tuple[str, str, dict, dict]:
    """
    Пайплайн с ограниченным контекстом — каждый агент видит только вывод предыдущего.
    Экономит токены: Coder не получает вывод Аналитика, Critic — только PlantUML-код.
    """
    llm = LLM(model=model)
    is_mbse_profile = prompt_profile == "mbse"
    p = {} if is_mbse_profile else _get_prompts()

    analyst = Agent(
        role="Senior Business Analyst",
        goal=p.get("analyst", {}).get("goal", "Точно извлечь из требований только то, что явно в них написано: сущности, их атрибуты, статусы, действия и взаимодействия между ними."),
        backstory=p.get("analyst", {}).get("backstory", "Ты опытный бизнес-аналитик. Строгое следование тексту требований без додумывания."),
        verbose=True, allow_delegation=False, llm=llm,
    )

    architect = Agent(
        role="Principal System Architect",
        goal=p.get("architect", {}).get("goal", "Спроектировать архитектуру строго на основе анализа: только те блоки и связи, которые следуют из требований."),
        backstory=p.get("architect", {}).get("backstory", "Ты главный архитектор с экспертизой в UML и SysML. Проектируй только то, что есть в требованиях."),
        verbose=True, allow_delegation=False, llm=llm,
    )

    coder = Agent(
        role="PlantUML Expert Engineer",
        goal=p.get("coder", {}).get("goal", f"Написать полный и валидный код PlantUML для диаграммы типа '{diagram_type}'."),
        backstory=p.get("coder", {}).get("backstory", "Ты эксперт по PlantUML. Пишешь ТОЛЬКО код без пояснений."),
        verbose=True, allow_delegation=False, llm=llm,
    )

    critic = Agent(
        role="UML Diagram Critic",
        goal=p.get("critic", {}).get("goal", "Критически оценить PlantUML-диаграмму: выявить пропущенные элементы и ошибки нотации."),
        backstory=p.get("critic", {}).get("backstory", "Ты строгий эксперт по UML. Сравниваешь диаграмму с требованиями и формулируешь конкретные замечания."),
        verbose=True, allow_delegation=False, llm=llm,
    )

    analysis_description, analysis_expected = _analysis_task_text(requirements, diagram_type)
    task1 = Task(
        description=analysis_description,
        expected_output=analysis_expected,
        agent=analyst,
    )

    architecture_description, architecture_expected = _architecture_task_text(diagram_type)
    task2 = Task(
        description=architecture_description,
        expected_output=architecture_expected,
        agent=architect,
        context=[task1],
    )

    if is_mbse_profile:
        _coder_prompt_slim = MBSE_CODER_PROMPTS.get(diagram_type, CODER_PROMPTS[diagram_type])
    else:
        _coder_prompt_slim = p.get("coder_prompts", {}).get(diagram_type) or CODER_PROMPTS[diagram_type]
    task3 = Task(
        description=_coder_prompt_slim,
        expected_output="Только код PlantUML — от @startuml до @enduml, без какого-либо текста вокруг.",
        agent=coder,
        context=[task2],
    )

    task4 = Task(
        description=(
            f"Ты получил исходные требования и PlantUML-код диаграммы типа '{diagram_type}'.\n\n"
            f"ИСХОДНЫЕ ТРЕБОВАНИЯ:\n{requirements}\n\n"
            "Проверь диаграмму: пропущенные элементы, ошибки нотации, общая оценка 0–10."
        ),
        expected_output="Markdown-отчёт: пропущенные элементы, ошибки нотации, оценка 0–10.",
        agent=critic,
        context=[task3],
    )

    task5 = Task(
        description=(
            f"Исправь PlantUML-диаграмму типа '{diagram_type}' по замечаниям критика.\n"
            "Добавь пропущенное, исправь нотацию, не удаляй правильное.\n"
            "НЕ пиши никаких пояснений — только исправленный код от @startuml до @enduml."
        ),
        expected_output="Только исправленный код PlantUML — от @startuml до @enduml.",
        agent=coder,
        context=[task3, task4],
    )

    crew = Crew(
        agents=[analyst, architect, coder, critic],
        tasks=[task1, task2, task3, task4, task5],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )

    crew.kickoff()
    diagram = task5.output.raw
    critique = task4.output.raw
    usage = _extract_usage(crew, model)
    intermediates = {
        "analyst":   task1.output.raw,
        "architect": task2.output.raw,
        "coder_v1":  task3.output.raw,
    }
    return diagram, critique, usage, intermediates


def run_evaluation(
    requirements: str,
    puml_code: str,
    diagram_type: str,
    metrics: dict,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Запускает отдельного агента-оценщика: анализирует пробелы, ограничения и метрики.

    :param requirements: исходный текст требований
    :param puml_code: финальный PlantUML-код после всех исправлений
    :param diagram_type: тип диаграммы
    :param metrics: вычисленные метрики диаграммы
    :param model: ID модели
    :return: структурированный отчёт (Markdown)
    """
    llm = LLM(model=model)

    metrics_text = "\n".join(f"- {k}: {v}" for k, v in metrics.items())

    evaluator = Agent(
        role="System Design Evaluator",
        goal=(
            "Выявить пробелы, ограничения и проблемы в сгенерированной диаграмме. "
            "Не описывать то, что уже есть — только то, чего не хватает и как это исправить."
        ),
        backstory=(
            "Ты независимый эксперт по системному анализу и UML. "
            "Твой подход — критический и конструктивный:\n"
            "- ты НЕ описываешь что есть в диаграмме — это и так видно\n"
            "- ты фокусируешься на том, чего НЕТ, что неправильно, что можно улучшить\n"
            "- ты анализируешь метрики и интерпретируешь их — хорошие ли значения для данной системы\n"
            "- ты даёшь конкретные рекомендации: как переформулировать требования, "
            "что добавить в промпты агентов, в чём ограничения текущего подхода\n"
            "Ты пишешь кратко и по делу. Никакой воды, никакого восхваления."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    eval_task = Task(
        description=(
            f"Проанализируй результат работы многоагентной системы генерации диаграммы типа '{diagram_type}'.\n\n"
            f"ИСХОДНЫЕ ТРЕБОВАНИЯ:\n{requirements}\n\n"
            f"ВЫЧИСЛЕННЫЕ МЕТРИКИ:\n{metrics_text}\n\n"
            f"ФИНАЛЬНЫЙ PLANTUML-КОД:\n{puml_code}\n\n"
            "Составь отчёт строго по разделам ниже. "
            "В каждом разделе пиши только о проблемах и упущениях — не описывай то, что уже правильно сделано.\n\n"
            "## Что отсутствует в диаграмме\n"
            "Конкретный список сущностей, связей, атрибутов или поведений из требований, "
            "которые не попали в диаграмму. Если всё покрыто — укажи это одной строкой.\n\n"
            "## Ошибки и несоответствия\n"
            f"Конкретные нарушения UML-нотации для типа '{diagram_type}', "
            "смысловые противоречия с требованиями, некорректные связи.\n\n"
            "## Анализ метрик\n"
            "Интерпретируй вычисленные метрики применительно к данной системе:\n"
            "- какие значения вызывают вопросы (слишком мало/много блоков, низкое покрытие и т.д.)\n"
            "- что метрики говорят о качестве генерации\n"
            "- на что стоит обратить внимание при сравнении с другими запусками\n\n"
            "## Ограничения подхода\n"
            "Что принципиально сложно для многоагентной системы в данном случае: "
            "неоднозначности в требованиях, архитектурные решения которые нельзя вывести из текста, "
            "ограничения PlantUML-нотации.\n\n"
            "## Рекомендации\n"
            "Конкретные действия для улучшения результата:\n"
            "- что переформулировать во входных требованиях\n"
            "- что добавить в промпты агентов (Analyst, Architect, Coder)\n"
            "- стоит ли попробовать другую модель или тип диаграммы для этой задачи\n\n"
            "## Итоговая оценка\n"
            "Один абзац: общая оценка качества (0–10) с конкретным обоснованием."
        ),
        expected_output=(
            "Markdown-отчёт: что отсутствует, ошибки и несоответствия, "
            "анализ метрик, ограничения подхода, рекомендации, итоговая оценка 0–10."
        ),
        agent=evaluator,
    )

    crew = Crew(
        agents=[evaluator],
        tasks=[eval_task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )

    result = crew.kickoff()
    return str(result)


def run_render_fixer(
    puml_code: str,
    diagram_type: str,
    render_error: str | None,
    model: str = "gpt-4o-mini",
) -> tuple[str, dict]:
    """
    Fixes PlantUML syntax after renderer failure. Intended for MBSE post-render recovery only.
    Returns (fixed_puml, usage).
    """
    llm = LLM(model=model)
    syntax_rules = MBSE_CODER_PROMPTS.get(diagram_type, CODER_PROMPTS.get(diagram_type, ""))

    fixer = Agent(
        role="PlantUML Render Fixer",
        goal=(
            "Repair a PlantUML diagram that failed to render, changing only syntax and notation "
            "needed to make the diagram render successfully."
        ),
        backstory=(
            "You are a strict PlantUML engineer. You do not redesign the system and do not add new "
            "domain content. You only fix invalid syntax, unsupported notation, mixed diagram syntax, "
            "bad aliases, malformed relationships, and markdown/noise around the code."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    fix_task = Task(
        description=(
            f"Fix this PlantUML diagram of type '{diagram_type}' so that it renders successfully.\n\n"
            f"RENDER ERROR:\n{render_error or 'Renderer returned no PNG.'}\n\n"
            f"STRICT RULES FOR THIS DIAGRAM TYPE:\n{syntax_rules}\n\n"
            f"CURRENT PLANTUML:\n{puml_code}\n\n"
            "Rules:\n"
            "- First fix the RENDER ERROR diagnostics above. Treat them as higher priority than all other guidance.\n"
            "- If diagnostics mention a line or token, change that exact construct first.\n"
            "- Return ONLY PlantUML code from @startuml to @enduml.\n"
            "- Do not wrap the answer in markdown.\n"
            "- Do not remove valid domain elements unless they are syntactically impossible.\n"
            "- Do not add new requirements or new business concepts.\n"
            "- Prefer the smallest syntax-only fix that makes the diagram render.\n"
        ),
        expected_output="Only fixed PlantUML code from @startuml to @enduml.",
        agent=fixer,
    )

    crew = Crew(
        agents=[fixer],
        tasks=[fix_task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    crew.kickoff()
    return fix_task.output.raw, _extract_usage(crew, model)


# ══════════════════════════════════════════════════════════════════════════════
# MBSE Pipeline — генерация полного комплекта диаграмм
# ══════════════════════════════════════════════════════════════════════════════

def _add_usage(total: dict, addition: dict) -> None:
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        if addition.get(key) is not None:
            total[key] = (total.get(key) or 0) + addition[key]
    if addition.get("cost_usd") is not None:
        total["cost_usd"] = round((total.get("cost_usd") or 0.0) + addition["cost_usd"], 6)


_CANONICAL_KEYS = [
    "system_name",
    "actors",
    "entities",
    "relationships",
    "states",
    "scenarios",
    "components",
    "external_systems",
    "business_rules",
    "assumptions",
    "gaps",
]


def _empty_canonical_model() -> dict:
    return {
        "system_name": "Unnamed system",
        "actors": [],
        "entities": [],
        "relationships": [],
        "states": [],
        "scenarios": [],
        "components": [],
        "external_systems": [],
        "business_rules": [],
        "assumptions": [],
        "gaps": [],
    }


def _extract_json_object(raw: str) -> dict:
    """Парсит JSON даже если модель обернула его в markdown-блок."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("canonical_model"), dict):
        data = data["canonical_model"]
    if not isinstance(data, dict):
        raise ValueError("Canonical model must be a JSON object.")
    return data


def _normalise_canonical_model(model: dict) -> dict:
    normalised = _empty_canonical_model()
    if isinstance(model, dict):
        normalised.update(model)

    for key in _CANONICAL_KEYS:
        if key == "system_name":
            if not isinstance(normalised.get(key), str) or not normalised[key].strip():
                normalised[key] = "Unnamed system"
        elif not isinstance(normalised.get(key), list):
            normalised[key] = []

    return normalised


def _item_name(item) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("id") or item.get("title") or "").strip()
    return str(item).strip()


def _validate_canonical_model(model: dict) -> list[str]:
    issues: list[str] = []
    if not isinstance(model, dict):
        return ["Canonical model is not a JSON object."]

    for key in _CANONICAL_KEYS:
        if key not in model:
            issues.append(f"Missing required key: {key}")

    entities = model.get("entities", [])
    if not isinstance(entities, list) or not entities:
        issues.append("No entities extracted. Class diagram will be weak.")

    name_pool: set[str] = set()
    for key in ("actors", "entities", "components", "external_systems"):
        values = model.get(key, [])
        if not isinstance(values, list):
            issues.append(f"{key} must be a list.")
            continue
        for item in values:
            name = _item_name(item)
            if name:
                name_pool.add(name)
            else:
                issues.append(f"Unnamed item in {key}.")

    relationships = model.get("relationships", [])
    if isinstance(relationships, list):
        for i, rel in enumerate(relationships, start=1):
            if not isinstance(rel, dict):
                issues.append(f"Relationship #{i} is not an object.")
                continue
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            if not source or not target:
                issues.append(f"Relationship #{i} has empty source or target.")
            elif name_pool and (source not in name_pool or target not in name_pool):
                issues.append(f"Relationship #{i} references unknown endpoint: {source} -> {target}.")
    else:
        issues.append("relationships must be a list.")

    scenarios = model.get("scenarios", [])
    if isinstance(scenarios, list):
        for i, scenario in enumerate(scenarios, start=1):
            if not isinstance(scenario, dict):
                issues.append(f"Scenario #{i} is not an object.")
                continue
            if not scenario.get("name"):
                issues.append(f"Scenario #{i} has no name.")
            if not scenario.get("steps"):
                issues.append(f"Scenario #{i} has no steps.")
    else:
        issues.append("scenarios must be a list.")

    return issues


def _canonical_json(model: dict) -> str:
    return json.dumps(model, ensure_ascii=False, indent=2)


def _compact_for_evidence(text: str) -> str:
    """Normalize text for evidence checks against PDF-extracted line wraps."""
    return re.sub(r"\s+", "", (text or "").lower())


def _flat_for_evidence(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _evidence_supported(evidence: str, requirements: str) -> bool:
    if not evidence:
        return False
    evidence_flat = _flat_for_evidence(evidence)
    req_flat = _flat_for_evidence(requirements)
    if evidence_flat in req_flat:
        return True
    if _compact_for_evidence(evidence) in _compact_for_evidence(requirements):
        return True
    evidence_words = {
        w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9]{3,}", evidence_flat)
        if w not in _FACT_STOP_WORDS
    }
    if not evidence_words:
        return False
    req_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]{3,}", req_flat))
    matched = len(evidence_words & req_words)
    return matched >= max(3, round(len(evidence_words) * 0.8))


def _merge_views(*view_lists: list[str]) -> list[str]:
    allowed = {"class", "sequence", "component", "activity"}
    result: list[str] = []
    for views in view_lists:
        for view in views or []:
            view = str(view).strip().lower()
            if view in allowed and view not in result:
                result.append(view)
    return result


def _looks_like_interaction(text: str) -> bool:
    return any(w in text for w in (
        "send", "receive", "request", "response", "approval", "denial", "payment",
        "enter", "entered", "presents", "signals", "asks", "tells", "repeats",
        "records", "updates", "fails", "failure", "restart", "recover", "alternate",
        "cancel", "suspend", "remove", "discount", "credit", "coupon", "rebate",
        "receipt", "signature", "scanner", "keyboard",
    ))


def _looks_like_class_operation(text: str) -> bool:
    return any(w in text for w in (
        "operation", "method", "command", "capability", "use case", "scenario step",
        "start", "enter", "record", "apply", "calculate", "generate", "create",
        "update", "remove", "delete", "cancel", "suspend", "recover", "authorize",
        "validate", "print", "save", "load", "select", "search", "scan", "request",
        "submit", "approve", "deny", "restart", "notify", "log",
    ))


def _infer_fact_views(fact_type: str, text: str, evidence: str, current_views: list[str]) -> list[str]:
    """Make view allocation deterministic enough that agents cannot under-assign facts."""
    kind = fact_type.lower().strip()
    haystack = f"{text} {evidence}".lower()
    inferred: list[str] = []
    known_type = True

    if kind in {"frequency", "occurrence"}:
        return []

    if kind in {"actor", "stakeholder", "role"}:
        inferred += ["class", "activity"]
        if any(w in haystack for w in ("cashier", "customer", "system")):
            inferred.append("sequence")
    if kind in {"entity", "attribute", "state"}:
        inferred.append("class")
    if kind in {"relationship", "business_rule", "precondition", "postcondition"}:
        inferred += ["class", "activity"]
        if _looks_like_interaction(haystack) or "external" in haystack or "authorization" in haystack:
            inferred.append("sequence")
    if kind in {"operation", "method", "command", "capability", "system_operation"}:
        inferred.append("class")
        if _looks_like_interaction(haystack):
            inferred += ["sequence", "activity"]
    if kind in {"scenario", "main_flow", "alternative_flow", "exception", "extension", "flow", "use_case_step"}:
        inferred += ["sequence", "activity"]
        if any(w in haystack for w in ("system operation", "service operation", "interface operation", "method", "command", "capability")):
            inferred.append("class")
    if kind in {"component", "interface", "external_system", "datastore"}:
        inferred += ["component"]
        if _looks_like_interaction(haystack) or "authorization" in haystack or "external" in haystack:
            inferred.append("sequence")
    if kind in {"special_requirement"}:
        inferred += ["class", "activity"]
        if any(w in haystack for w in ("ui", "screen", "touch", "remote", "inventory", "authorization", "business rules")):
            inferred.append("component")
        if any(w in haystack for w in ("remote services", "recovery", "authorization response")):
            inferred.append("sequence")
    if kind in {"data_variation", "variation", "variations", "technology_variation"}:
        inferred += ["class", "activity"]
        if any(w in haystack for w in ("scanner", "card reader", "signature", "keyboard", "digital")):
            inferred.append("component")
    if kind in {"requirement", "functional_requirement"} and _looks_like_interaction(haystack):
        inferred += ["sequence", "activity"]
    elif not inferred:
        known_type = False

    if any(w in haystack for w in ("accounting system", "inventory system", "payment authorization service", "external")):
        inferred.append("component")
        inferred.append("sequence")
    if "repeat" in haystack or "until" in haystack:
        inferred.append("sequence")
        inferred.append("activity")

    if known_type and inferred:
        return _merge_views(inferred)
    return _merge_views(current_views, inferred)


def _normalise_requirement_facts(raw: dict, requirements: str) -> tuple[list[dict], list[str]]:
    facts_raw = raw.get("facts", raw) if isinstance(raw, dict) else raw
    if not isinstance(facts_raw, list):
        return [], ["Requirement facts must be a JSON array or an object with a facts array."]

    allowed_views = {"class", "sequence", "component", "activity"}
    facts: list[dict] = []
    issues: list[str] = []
    used_ids: set[str] = set()

    for index, item in enumerate(facts_raw, start=1):
        if not isinstance(item, dict):
            issues.append(f"Fact #{index} is not an object.")
            continue

        fact_id = str(item.get("id") or f"REQ-{index:03d}").strip()
        if fact_id in used_ids:
            fact_id = f"{fact_id}-{index}"
        used_ids.add(fact_id)

        text = str(item.get("text") or "").strip()
        fact_type = str(item.get("type") or "requirement").strip()
        evidence = str(item.get("evidence_quote") or "").strip()
        views = item.get("must_appear_in", [])
        if not isinstance(views, list):
            views = []
        views = [str(v).strip().lower() for v in views if str(v).strip().lower() in allowed_views]

        if not text:
            issues.append(f"{fact_id}: missing fact text.")
            continue
        if not evidence:
            issues.append(f"{fact_id}: missing evidence_quote.")
            continue

        if not _evidence_supported(evidence, requirements):
            issues.append(f"{fact_id}: evidence_quote is not an exact substring of requirements.")
            continue

        facts.append({
            "id": fact_id,
            "type": fact_type,
            "text": text,
            "evidence_quote": evidence,
            "must_appear_in": _infer_fact_views(fact_type, text, evidence, views),
        })

    return facts, issues


def _empty_view_briefs() -> dict:
    return {
        "class": {
            "entities": [],
            "attributes": [],
            "methods": [],
            "states": [],
            "relationships": [],
            "business_rules": [],
        },
        "sequence": {
            "actors": [],
            "participants": [],
            "main_flow": [],
            "alternative_flows": [],
            "exceptions": [],
            "external_calls": [],
        },
        "component": {
            "components": [],
            "interfaces": [],
            "external_systems": [],
            "datastores": [],
            "dependencies": [],
        },
        "activity": {
            "swimlanes": [],
            "actions": [],
            "decisions": [],
            "parallel_flows": [],
            "start_conditions": [],
            "end_conditions": [],
        },
    }


def _normalise_view_briefs(raw: dict) -> dict:
    result = _empty_view_briefs()
    if not isinstance(raw, dict):
        return result
    for dtype, defaults in result.items():
        section = raw.get(dtype, {})
        if not isinstance(section, dict):
            section = {}
        for key in defaults:
            value = section.get(key, [])
            result[dtype][key] = value if isinstance(value, list) else []
    return result


def _append_unique(items: list, value) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    if value not in items:
        items.append(value)


def _fact_text_blob(facts: list[dict], view: str | None = None) -> str:
    selected = []
    for fact in facts or []:
        if view is None or view in fact.get("must_appear_in", []):
            selected.append(str(fact.get("text", "")))
            selected.append(str(fact.get("evidence_quote", "")))
    return " ".join(selected).lower()


def _augment_view_briefs(view_briefs: dict, canonical_model: dict, requirement_facts: list[dict]) -> dict:
    """Programmatic backfill for view allocations that LLMs often leave too sparse."""
    result = _normalise_view_briefs(view_briefs)
    all_fact_blob = _fact_text_blob(requirement_facts)
    sequence_blob = _fact_text_blob(requirement_facts, "sequence")
    activity_blob = _fact_text_blob(requirement_facts, "activity")
    sequence_step_blob = " ".join(
        f"{fact.get('text', '')} {fact.get('evidence_quote', '')}"
        for fact in requirement_facts or []
        if str(fact.get("type", "")).lower() in {"scenario", "main_flow", "alternative_flow", "extension", "exception", "precondition"}
    ).lower()
    external_names = {
        _item_name(item).lower()
        for item in canonical_model.get("external_systems", []) or []
        if _item_name(item)
    }

    for actor in canonical_model.get("actors", []) or []:
        name = _item_name(actor)
        if not name:
            continue
        name_l = name.lower()
        if name_l not in external_names and (
            name_l in sequence_step_blob
            or name_l in all_fact_blob and name_l in {"cashier", "customer", "system"}
        ):
            _append_unique(result["sequence"]["actors"], name)
        if name_l in activity_blob or name_l in all_fact_blob and name_l in {"cashier", "customer", "system"}:
            _append_unique(result["activity"]["swimlanes"], name)

    if "system" in all_fact_blob:
        _append_unique(result["sequence"]["participants"], "System")
        _append_unique(result["activity"]["swimlanes"], "System")

    for entity in canonical_model.get("entities", []) or []:
        name = _item_name(entity)
        if name and name.lower() in sequence_blob:
            _append_unique(result["sequence"]["participants"], name)

    for ext in canonical_model.get("external_systems", []) or []:
        name = _item_name(ext)
        if not name:
            continue
        if name.lower() in sequence_blob:
            _append_unique(result["sequence"]["participants"], name)
            _append_unique(result["sequence"]["external_calls"], name)
        if name.lower() in all_fact_blob:
            _append_unique(result["component"]["external_systems"], ext)

    for comp in canonical_model.get("components", []) or []:
        name = _item_name(comp)
        if not name:
            continue
        if name.lower() in sequence_blob:
            _append_unique(result["sequence"]["participants"], name)
        if name.lower() in activity_blob:
            _append_unique(result["activity"]["swimlanes"], name)
        _append_unique(result["component"]["components"], comp)

    for fact in requirement_facts or []:
        text = str(fact.get("text") or "").strip()
        if not text:
            continue
        kind = str(fact.get("type") or "").lower()
        views = fact.get("must_appear_in", [])
        if "sequence" in views:
            if kind in {"scenario", "main_flow"}:
                _append_unique(result["sequence"]["main_flow"], text)
            elif kind in {"alternative_flow", "extension"}:
                _append_unique(result["sequence"]["alternative_flows"], text)
            elif kind == "exception":
                _append_unique(result["sequence"]["exceptions"], text)
        if "activity" in views:
            if kind in {"scenario", "main_flow", "alternative_flow", "extension", "exception"}:
                _append_unique(result["activity"]["actions"], text)
            elif kind == "precondition":
                _append_unique(result["activity"]["start_conditions"], text)
            elif kind == "postcondition":
                _append_unique(result["activity"]["end_conditions"], text)

    return result


def _validate_view_briefs(view_briefs: dict) -> list[str]:
    issues: list[str] = []
    expected = _empty_view_briefs()
    if not isinstance(view_briefs, dict):
        return ["View briefs must be a JSON object."]
    for dtype, fields in expected.items():
        section = view_briefs.get(dtype)
        if not isinstance(section, dict):
            issues.append(f"Missing or invalid view brief: {dtype}")
            continue
        non_empty = 0
        for key in fields:
            value = section.get(key)
            if not isinstance(value, list):
                issues.append(f"{dtype}.{key} must be a list.")
            elif value:
                non_empty += 1
        if non_empty == 0:
            issues.append(f"{dtype} view brief is empty.")
    return issues


def _build_view_spec(canonical_model: dict, diagram_type: str) -> str:
    base = (
        "Use the canonical system model below as the single source of truth. "
        "Preserve exact names from the model. Do not introduce entities, actors, components, "
        "states, or external systems that are absent from the model. If something is missing, "
        "represent it as a note or omit it instead of inventing details.\n\n"
        f"CANONICAL SYSTEM MODEL:\n{_canonical_json(canonical_model)}\n\n"
    )

    if diagram_type == "class":
        focus = (
            "CLASS VIEW SPEC:\n"
            "- Build classes from entities.\n"
            "- Build actor/role boundary concepts, external systems, and integration points as interfaces or boundary classes when they participate in the system contract.\n"
            "- Add attributes from each entity.attributes.\n"
            "- Add operations from entity.operations and from scenario steps that describe visible system capabilities.\n"
            "- Add enums from states.\n"
            "- Add relationships from relationships with explicit UML relation types.\n"
            "- Represent structural requirement facts as classes/interfaces/enums, attributes, methods, or relationships; do not cover them only with notes.\n"
            "- Add business_rules, constraints, and ambiguous details as concise PlantUML notes attached to the relevant structural element.\n"
        )
    elif diagram_type == "sequence":
        focus = (
            "SEQUENCE VIEW SPEC:\n"
            "- Build one main interaction from the most important scenario.\n"
            "- Use actors, components, entities, and external_systems as participants only when they appear in scenario steps.\n"
            "- Declare every participant only as actor or participant. Never use external/component/class/database in a sequence diagram.\n"
            "- Use alt/else for alternative flows and exceptions from scenarios and business_rules.\n"
            "- Use loop when the scenario explicitly repeats item processing or similar repeated actions.\n"
            "- Keep message names close to the wording in scenarios.steps.\n"
        )
    elif diagram_type == "component":
        focus = (
            "COMPONENT VIEW SPEC:\n"
            "- Build components from components and external_systems.\n"
            "- Group components into packages/layers using each component.layer if available.\n"
            "- Show entities as databases or repositories only when storage is explicitly implied.\n"
            "- Show interfaces between components and external systems from relationships and scenarios.\n"
            "- Use only component, interface, database, queue, package, and A --> B : label relationships.\n"
            "- Never use actor, class, enum, --(, or --) in the component diagram.\n"
            "- Represent people/organizations as component \"Name\" as Alias <<actor>> if needed.\n"
            "- Do not create generic infrastructure unless it is present in the model.\n"
        )
    elif diagram_type == "activity":
        focus = (
            "ACTIVITY VIEW SPEC:\n"
            "- Build the end-to-end business process from scenarios.\n"
            "- Use swimlanes from actors and major components.\n"
            "- Represent branches from alternative flows, exceptions, and business_rules.\n"
            "- Preserve the order of scenario steps.\n"
            "- Use modern PlantUML activity syntax only: start, :Action;, if/else/endif, fork/fork again/end fork, stop.\n"
            "- Do not use legacy (*) syntax.\n"
            "- Do not add implementation details absent from the canonical model.\n"
        )
    else:
        focus = "VIEW SPEC: Generate the requested UML view strictly from the canonical model.\n"

    return base + focus


def _facts_for_view(requirement_facts: list[dict], diagram_type: str) -> list[dict]:
    result = []
    for fact in requirement_facts or []:
        views = fact.get("must_appear_in", [])
        if diagram_type in views:
            result.append(fact)
    return result


def _fact_id(fact: dict) -> str:
    return str(fact.get("id") or "").strip()


def _trace_ids_from_puml(puml_code: str) -> set[str]:
    ids: set[str] = set()
    for line in (puml_code or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("'"):
            continue
        if "trace" not in stripped.lower():
            continue
        ids.update(re.findall(r"\bREQ-\d+(?:-\d+)?\b", stripped, flags=re.IGNORECASE))
    return {item.upper() for item in ids}


def _assigned_fact_ids(requirement_facts: list[dict], diagram_type: str) -> set[str]:
    return {
        _fact_id(fact).upper()
        for fact in requirement_facts or []
        if diagram_type in fact.get("must_appear_in", []) and _fact_id(fact)
    }


def _facts_by_id(requirement_facts: list[dict]) -> dict[str, dict]:
    return {
        _fact_id(fact).upper(): fact
        for fact in requirement_facts or []
        if _fact_id(fact)
    }


_CLASS_STRUCTURAL_FACT_TYPES = {
    "actor",
    "stakeholder",
    "role",
    "entity",
    "attribute",
    "relationship",
    "state",
    "component",
    "interface",
    "external_system",
    "datastore",
    "data_variation",
    "variation",
    "variations",
    "technology_variation",
    "operation",
    "method",
    "command",
    "capability",
    "system_operation",
}

_CLASS_NOTE_ELIGIBLE_FACT_TYPES = {
    "business_rule",
    "constraint",
    "nonfunctional_requirement",
    "quality_attribute",
    "frequency",
    "occurrence",
    "assumption",
    "gap",
}

_CLASS_STRUCTURAL_HINTS = (
    "actor", "role", "user", "class", "entity", "object", "attribute", "field",
    "state", "status", "enum", "type", "mode", "relationship", "association",
    "composition", "aggregation", "interface", "external system", "service",
    "integration", "api", "datastore", "database", "repository", "operation",
    "method", "command", "capability",
)

_CLASS_RELATION_RE = re.compile(
    r"(<\|--|--\|>|\*--|--\*|o--|--o|\.\.>|<\.\.|-->|<--|\.\.|--)"
)

_SEQUENCE_MESSAGE_RE = re.compile(r"[-.]+[<>]+|[<>]+[-.]+")
_COMPONENT_RELATION_RE = re.compile(r"(-->|<--|\.\.>|<\.\.)")


def _is_structural_class_fact(fact: dict) -> bool:
    kind = str(fact.get("type") or "").strip().lower()
    haystack = f"{fact.get('text', '')} {fact.get('evidence_quote', '')}".lower()

    if kind in _CLASS_STRUCTURAL_FACT_TYPES:
        return True
    if kind in _CLASS_NOTE_ELIGIBLE_FACT_TYPES and not any(h in haystack for h in _CLASS_STRUCTURAL_HINTS):
        return False
    if kind in {"scenario", "main_flow", "alternative_flow", "exception", "extension", "flow", "use_case_step"}:
        return _looks_like_class_operation(haystack)
    if kind in {"requirement", "functional_requirement", "special_requirement", "precondition", "postcondition"}:
        return any(h in haystack for h in _CLASS_STRUCTURAL_HINTS) or _looks_like_class_operation(haystack)
    return any(h in haystack for h in _CLASS_STRUCTURAL_HINTS)


def _is_sequence_interaction_fact(fact: dict) -> bool:
    kind = str(fact.get("type") or "").strip().lower()
    haystack = f"{fact.get('text', '')} {fact.get('evidence_quote', '')}".lower()
    if kind in {
        "actor", "stakeholder", "role", "scenario", "main_flow", "alternative_flow",
        "exception", "extension", "flow", "use_case_step", "operation", "method",
        "command", "capability", "system_operation",
    }:
        return True
    if kind in {"component", "interface", "external_system"}:
        return _looks_like_interaction(haystack) or any(w in haystack for w in ("api", "service", "request", "response"))
    if kind in {"relationship", "business_rule", "precondition", "postcondition", "requirement", "functional_requirement", "special_requirement"}:
        return _looks_like_interaction(haystack) or any(w in haystack for w in ("alternative", "exception", "else", "repeat", "until"))
    return _looks_like_interaction(haystack)


def _is_activity_process_fact(fact: dict) -> bool:
    kind = str(fact.get("type") or "").strip().lower()
    haystack = f"{fact.get('text', '')} {fact.get('evidence_quote', '')}".lower()
    if kind in {
        "actor", "stakeholder", "role", "scenario", "main_flow", "alternative_flow",
        "exception", "extension", "flow", "use_case_step", "operation", "method",
        "command", "capability", "system_operation", "precondition", "postcondition",
    }:
        return True
    if kind in {"business_rule", "requirement", "functional_requirement", "special_requirement"}:
        return _looks_like_interaction(haystack) or any(
            w in haystack for w in ("if", "when", "then", "else", "decision", "cancel", "recover", "repeat", "parallel")
        )
    return False


def _is_component_structural_fact(fact: dict) -> bool:
    kind = str(fact.get("type") or "").strip().lower()
    haystack = f"{fact.get('text', '')} {fact.get('evidence_quote', '')}".lower()
    if kind in {"component", "interface", "external_system", "datastore", "queue"}:
        return True
    if kind in {"relationship", "dependency", "integration", "api"}:
        return any(w in haystack for w in ("component", "interface", "external", "service", "api", "database", "datastore", "queue"))
    if kind in {"special_requirement", "requirement", "functional_requirement"}:
        return any(
            w in haystack
            for w in ("component", "subsystem", "module", "interface", "api", "external system", "service", "database", "datastore", "queue", "integration")
        )
    return False


def _required_trace_contexts_for_fact(diagram_type: str, fact: dict) -> set[str]:
    if diagram_type == "class" and _is_structural_class_fact(fact):
        return {"classifier", "member", "relationship"}
    if diagram_type == "sequence" and _is_sequence_interaction_fact(fact):
        return {"participant", "message", "control", "activation"}
    if diagram_type == "activity" and _is_activity_process_fact(fact):
        return {"swimlane", "action", "decision", "fork", "terminal"}
    if diagram_type == "component" and _is_component_structural_fact(fact):
        return {"component", "interface", "datastore", "queue", "relationship"}
    return set()


def _class_trace_target_context(stripped_line: str) -> str:
    stripped = stripped_line.strip()
    if not stripped:
        return "other"
    lowered = stripped.lower()
    if lowered.startswith("note "):
        return "note"
    if re.match(r"(abstract\s+class|class|interface|enum)\b", lowered):
        return "classifier"
    if re.match(r"^[+\-#~]\s*[\w\"']", stripped):
        return "member"
    if _CLASS_RELATION_RE.search(stripped):
        return "relationship"
    if lowered.startswith("package "):
        return "package"
    return "other"


def _class_line_contexts(lines: list[str]) -> list[str]:
    contexts: list[str] = []
    in_note = False
    classifier_depth = 0

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()

        if in_note or lowered.startswith("note "):
            context = "note"
        elif classifier_depth > 0:
            context = "member" if stripped and not stripped.startswith(("'", "}")) else "other"
        else:
            context = _class_trace_target_context(stripped)
        contexts.append(context)

        if in_note:
            if lowered.startswith("end note"):
                in_note = False
            continue

        if lowered.startswith("note ") and ":" not in stripped:
            in_note = True
            continue

        if classifier_depth > 0:
            classifier_depth += stripped.count("{") - stripped.count("}")
            classifier_depth = max(0, classifier_depth)
            continue

        if re.match(r"(abstract\s+class|class|interface|enum)\b", lowered) and "{" in stripped:
            classifier_depth = max(0, stripped.count("{") - stripped.count("}"))

    return contexts


def _note_aware_contexts(lines: list[str], classifier) -> list[str]:
    contexts: list[str] = []
    in_note = False
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if in_note or lowered.startswith("note "):
            context = "note"
        else:
            context = classifier(stripped)
        contexts.append(context)

        if in_note:
            if lowered.startswith("end note"):
                in_note = False
            continue
        if lowered.startswith("note ") and ":" not in stripped:
            in_note = True
    return contexts


def _sequence_trace_target_context(stripped_line: str) -> str:
    stripped = stripped_line.strip()
    lowered = stripped.lower()
    if not stripped:
        return "other"
    if lowered.startswith("note "):
        return "note"
    if re.match(r"(actor|participant)\b", lowered):
        return "participant"
    if re.match(r"(alt|else|opt|loop|par|break|critical|group|end)\b", lowered):
        return "control"
    if re.match(r"(activate|deactivate)\b", lowered):
        return "activation"
    if _SEQUENCE_MESSAGE_RE.search(stripped):
        return "message"
    return "other"


def _activity_trace_target_context(stripped_line: str) -> str:
    stripped = stripped_line.strip()
    lowered = stripped.lower()
    if not stripped:
        return "other"
    if lowered.startswith("note "):
        return "note"
    if re.match(r"^\|[^|]+\|$", stripped):
        return "swimlane"
    if lowered in {"start", "stop", "end"}:
        return "terminal"
    if re.match(r"^:.*;\s*$", stripped):
        return "action"
    if re.match(r"(if|elseif|else|endif)\b", lowered):
        return "decision"
    if re.match(r"(fork|fork again|end fork)\b", lowered):
        return "fork"
    return "other"


def _component_trace_target_context(stripped_line: str) -> str:
    stripped = stripped_line.strip()
    lowered = stripped.lower()
    if not stripped:
        return "other"
    if lowered.startswith("note "):
        return "note"
    if re.match(r"component\b", lowered):
        return "component"
    if re.match(r"interface\b", lowered):
        return "interface"
    if re.match(r"database\b", lowered):
        return "datastore"
    if re.match(r"queue\b", lowered):
        return "queue"
    if _COMPONENT_RELATION_RE.search(stripped):
        return "relationship"
    return "other"


def _line_contexts_for_diagram(lines: list[str], diagram_type: str) -> list[str]:
    if diagram_type == "class":
        return _class_line_contexts(lines)
    if diagram_type == "sequence":
        return _note_aware_contexts(lines, _sequence_trace_target_context)
    if diagram_type == "activity":
        return _note_aware_contexts(lines, _activity_trace_target_context)
    if diagram_type == "component":
        return _note_aware_contexts(lines, _component_trace_target_context)
    return ["trace"] * len(lines)


def _nearby_non_comment_context(
    lines: list[str],
    contexts: list[str],
    start: int,
    step: int,
) -> str:
    i = start + step
    while 0 <= i < len(lines):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("'"):
            return contexts[i]
        i += step
    return "other"


def _trace_id_contexts_from_puml(puml_code: str, diagram_type: str) -> dict[str, set[str]]:
    contexts_by_id: dict[str, set[str]] = {}
    lines = (puml_code or "").splitlines()
    line_contexts = _line_contexts_for_diagram(lines, diagram_type)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("'") or "trace" not in stripped.lower():
            continue
        ids = re.findall(r"\bREQ-\d+(?:-\d+)?\b", stripped, flags=re.IGNORECASE)
        if not ids:
            continue

        if diagram_type in {"class", "sequence", "activity", "component"}:
            context = line_contexts[i]
            if context != "note":
                next_context = _nearby_non_comment_context(lines, line_contexts, i, 1)
                context = next_context if next_context != "other" else context
        else:
            context = "trace"

        for fact_id in ids:
            contexts_by_id.setdefault(fact_id.upper(), set()).add(context)

    return contexts_by_id


def _trace_contract(requirement_facts: list[dict], diagram_type: str) -> str:
    facts = _facts_for_view(requirement_facts or [], diagram_type)
    ids = ", ".join(_fact_id(fact) for fact in facts if _fact_id(fact))
    representation_rule = ""
    if diagram_type == "class":
        representation_rule = (
            "- In class diagrams, trace comments for structural facts must be placed above "
            "a class/interface/enum, attribute, method, or relationship; a note-only trace is incomplete.\n"
        )
    elif diagram_type == "sequence":
        representation_rule = (
            "- In sequence diagrams, trace comments for interaction facts must be placed above "
            "a participant, message, alt/else, loop, or activation element; a note-only trace is incomplete.\n"
        )
    elif diagram_type == "activity":
        representation_rule = (
            "- In activity diagrams, trace comments for process facts must be placed above "
            "a swimlane, action, decision, fork, start, or stop element; a note-only trace is incomplete.\n"
        )
    elif diagram_type == "component":
        representation_rule = (
            "- In component diagrams, trace comments for structural integration facts must be placed above "
            "a component, interface, datastore, queue, or dependency relationship; a note-only trace is incomplete.\n"
        )
    return (
        "TRACEABILITY CONTRACT:\n"
        f"- Required fact IDs for this view: {ids or '(none)'}.\n"
        "- Every required fact ID must appear in the PlantUML output in a trace comment.\n"
        "- Put trace comments immediately above the UML element that represents the fact.\n"
        "- Trace comment format is exactly: ' trace: REQ-001 REQ-002\n"
        "- One UML element may cover multiple facts; list all relevant IDs on that element's trace comment.\n"
        f"{representation_rule}"
        "- Do not use trace IDs that are not listed as required for this view.\n"
        "- PlantUML comments beginning with ' are valid and must be preserved.\n"
    )


def _semantic_check_puml(
    puml_code: str,
    diagram_type: str,
    requirement_facts: list[dict],
    allow_keyword_fallback: bool = False,
) -> dict:
    assigned_ids = _assigned_fact_ids(requirement_facts, diagram_type)
    facts_by_id = _facts_by_id(requirement_facts)
    trace_ids = _trace_ids_from_puml(puml_code)
    trace_contexts = _trace_id_contexts_from_puml(puml_code, diagram_type)
    allowed_present = trace_ids & assigned_ids
    unsupported = sorted(trace_ids - assigned_ids)
    note_only: list[str] = []

    for fact_id in sorted(list(allowed_present)):
        fact = facts_by_id.get(fact_id, {})
        contexts = trace_contexts.get(fact_id, set())
        required_contexts = _required_trace_contexts_for_fact(diagram_type, fact)
        if required_contexts and not (contexts & required_contexts):
            allowed_present.remove(fact_id)
            note_only.append(fact_id)

    if allow_keyword_fallback and not trace_ids:
        puml_lower = (puml_code or "").lower()
        allowed_present = {
            fact_id for fact_id in assigned_ids
            if _fact_covered_by_puml(facts_by_id.get(fact_id, {}), puml_lower)
        }

    missing = sorted(assigned_ids - allowed_present)
    expected_count = len(assigned_ids)
    present_count = len(allowed_present)
    return {
        "expected_count": expected_count,
        "present_count": present_count,
        "missing_count": len(missing),
        "coverage_pct": round(present_count / expected_count * 100, 1) if expected_count else 100.0,
        "present_trace_ids": sorted(allowed_present),
        "missing_fact_ids": missing,
        "missing_facts": [
            {
                "id": fact_id,
                "type": facts_by_id.get(fact_id, {}).get("type"),
                "text": facts_by_id.get(fact_id, {}).get("text"),
                "evidence_quote": facts_by_id.get(fact_id, {}).get("evidence_quote"),
            }
            for fact_id in missing
        ],
        "unsupported_trace_ids": unsupported,
        "note_only_fact_ids": note_only,
        "trace_contexts": {
            fact_id: sorted(contexts)
            for fact_id, contexts in sorted(trace_contexts.items())
            if fact_id in assigned_ids or fact_id in trace_ids
        },
    }


def _line_fact_match_score(line: str, fact: dict) -> int:
    stripped = line.strip()
    if not stripped or stripped.startswith("'") or stripped.startswith("@"):
        return 0
    keywords = _fact_keywords(fact)
    if not keywords:
        return 0
    line_lower = stripped.lower()
    return sum(1 for word in keywords if word in line_lower)


def _append_trace_id_to_line(line: str, fact_id: str) -> str:
    existing = re.findall(r"\bREQ-\d+(?:-\d+)?\b", line, flags=re.IGNORECASE)
    existing_upper = {item.upper() for item in existing}
    if fact_id.upper() in existing_upper:
        return line
    return line.rstrip() + f" {fact_id.upper()}"


def _repair_trace_comments_for_covered_facts(
    puml_code: str,
    diagram_type: str,
    requirement_facts: list[dict],
) -> str:
    """Add missing trace IDs when the fact is already visibly represented."""
    semantic = _semantic_check_puml(puml_code, diagram_type, requirement_facts)
    missing_ids = semantic.get("missing_fact_ids", [])
    if not missing_ids:
        return puml_code

    facts_by_id = _facts_by_id(requirement_facts)
    lines = (puml_code or "").splitlines()
    line_contexts = _line_contexts_for_diagram(lines, diagram_type)

    for fact_id in missing_ids:
        fact = facts_by_id.get(fact_id)
        if not fact:
            continue

        best_index = None
        best_score = 0
        required_contexts = _required_trace_contexts_for_fact(diagram_type, fact)
        for i, line in enumerate(lines):
            if required_contexts and i < len(line_contexts) and line_contexts[i] not in required_contexts:
                continue
            score = _line_fact_match_score(line, fact)
            if score > best_score:
                best_score = score
                best_index = i

        threshold = max(2, min(4, round(len(_fact_keywords(fact)) * 0.35)))
        if best_index is None or best_score < threshold:
            continue

        trace_index = None
        j = best_index - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0 and lines[j].strip().lower().startswith("' trace:"):
            trace_index = j

        indent = lines[best_index][:len(lines[best_index]) - len(lines[best_index].lstrip())]
        if trace_index is not None:
            lines[trace_index] = _append_trace_id_to_line(lines[trace_index], fact_id)
        else:
            lines.insert(best_index, f"{indent}' trace: {fact_id.upper()}")

    return "\n".join(lines)


def _build_mbse_input(
    requirements: str,
    project_context: dict,
    view_brief: dict,
    diagram_type: str,
    requirement_facts: list[dict] | None = None,
) -> str:
    if diagram_type == "class":
        focus = (
            "Сгенерируй UML class diagram. Удели максимум внимания сущностям, атрибутам, методам, enum, "
            "кратностям и типам связей. Доменные объекты, роли, состояния, внешние точки взаимодействия "
            "и системные операции показывай структурой UML, а не только note. Не теряй детали исходных требований."
        )
    elif diagram_type == "sequence":
        focus = (
            "Сгенерируй UML sequence diagram. Удели максимум внимания основному сценарию, альтернативным "
            "веткам, исключениям, внешним системам и порядку сообщений. Участники объявляются только через "
            "actor или participant."
        )
    elif diagram_type == "component":
        focus = (
            "Сгенерируй UML component diagram. Удели максимум внимания подсистемам, внешним сервисам, "
            "интерфейсам, хранилищам и зависимостям. Используй только component/interface/database/queue/package "
            "и связи вида A --> B : label."
        )
    elif diagram_type == "activity":
        focus = (
            "Сгенерируй UML activity diagram. Удели максимум внимания бизнес-процессу, swimlane-участникам, "
            "ветвлениям, исключениям и параллельным действиям. Используй современный синтаксис PlantUML: "
            "start, :Action;, if/else/endif, fork/fork again/end fork, stop."
        )
    else:
        focus = f"Сгенерируй UML-диаграмму типа {diagram_type}."

    view_semantic_rule = {
        "class": "- Для class diagram структурные факты нельзя закрывать только note: используй class/interface/enum, атрибуты, методы или связи; note только дополняет структуру.\n",
        "sequence": "- Для sequence diagram сценарные и интеракционные факты нельзя закрывать только note: используй actor/participant, сообщения, alt/else, loop или activation.\n",
        "activity": "- Для activity diagram процессные факты нельзя закрывать только note: используй swimlane, действия, if/else, fork, start/stop.\n",
        "component": "- Для component diagram факты о компонентах, внешних системах, интерфейсах, хранилищах и зависимостях нельзя закрывать только note: используй component/interface/database/queue и связи A --> B : label.\n",
    }.get(diagram_type, "")

    return (
        "MBSE MODE: GENERATE ONE VIEW OF A CONSISTENT UML SET.\n\n"
        "ГЛАВНОЕ ПРАВИЛО:\n"
        "Исходные требования ниже являются главным источником истины. Project context нужен только для "
        "единых имён, терминов и согласованности между диаграммами. Если между контекстом и требованиями "
        "есть расхождение, следуй исходным требованиям.\n\n"
        f"ИСХОДНЫЕ ТРЕБОВАНИЯ:\n{requirements}\n\n"
        f"TRACEABLE REQUIREMENT FACTS FOR THIS VIEW:\n{_canonical_json({'facts': _facts_for_view(requirement_facts or [], diagram_type)})}\n\n"
        f"{_trace_contract(requirement_facts or [], diagram_type)}\n"
        f"PROJECT CONTEXT / ГЛОССАРИЙ:\n{_canonical_json(project_context)}\n\n"
        f"VIEW BRIEF / ОБЯЗАТЕЛЬНЫЕ ЭЛЕМЕНТЫ ДЛЯ ЭТОЙ ДИАГРАММЫ:\n{_canonical_json(view_brief)}\n\n"
        f"ЗАДАНИЕ ДЛЯ ЭТОГО VIEW:\n{focus}\n\n"
        "ОБЩИЕ ПРАВИЛА:\n"
        "- Используй имена из Project context, если они соответствуют требованиям.\n"
        "- View Brief задаёт обязательные элементы, которые нужно покрыть в этой диаграмме.\n"
        "- Каждый TRACEABLE REQUIREMENT FACT для этого view должен быть представлен в диаграмме как элемент, связь, сообщение, действие, enum или note.\n"
        f"{view_semantic_rule}"
        "- У каждого такого элемента должен быть trace-comment с соответствующим REQ-ID.\n"
        "- Не выдумывай доменные сущности, которых нет в требованиях.\n"
        "- Не сокращай требования до Project context: извлекай детали из полного текста требований.\n"
        "- Если View Brief и исходные требования расходятся, следуй исходным требованиям и объясни расхождение в critique.\n"
        "- Верни только PlantUML-код от @startuml до @enduml.\n"
    )


def _diagram_semantic_representation_rules(diagram_type: str) -> str:
    if diagram_type == "class":
        return (
            "CLASS-VIEW SEMANTIC RULES:\n"
            "- Structural facts must be represented structurally: class/interface/enum declarations, attributes, methods, or relationships.\n"
            "- Do not cover a structural fact only by putting its trace ID on a note.\n"
            "- Notes are allowed for constraints, business rules, non-functional requirements, and ambiguous details, but they must attach to existing structure when possible.\n"
            "- If a missing fact describes a system capability or scenario operation, add or update a method on the responsible class/interface.\n"
            "- If a missing fact describes an external system or integration point, add an interface or boundary class and a dependency/association.\n"
        )
    if diagram_type == "sequence":
        return (
            "SEQUENCE-VIEW SEMANTIC RULES:\n"
            "- Interaction facts must be represented as participants, messages, alt/else branches, loop blocks, or activation spans.\n"
            "- Do not cover a scenario step, alternative, exception, request, response, or external call only by putting its trace ID on a note.\n"
            "- If a missing fact describes a branch or exception, add an alt/else block with concrete messages.\n"
            "- If a missing fact describes an external call, add the external participant and the request/response messages.\n"
        )
    if diagram_type == "activity":
        return (
            "ACTIVITY-VIEW SEMANTIC RULES:\n"
            "- Process facts must be represented as swimlanes, actions, decisions, fork blocks, start, or stop.\n"
            "- Do not cover a workflow step, decision, alternative, exception, cancellation, recovery, or parallel flow only by putting its trace ID on a note.\n"
            "- If a missing fact describes a branch, add if/else/endif with concrete actions.\n"
            "- If a missing fact describes a responsible actor or subsystem, add/use a swimlane.\n"
        )
    if diagram_type == "component":
        return (
            "COMPONENT-VIEW SEMANTIC RULES:\n"
            "- Component facts must be represented as component/interface/database/queue declarations or A --> B : label dependencies.\n"
            "- Do not cover a component, external system, interface, datastore, queue, API, or dependency only by putting its trace ID on a note.\n"
            "- If a missing fact describes an integration point, add the interface/external component and the dependency relationship.\n"
            "- Do not add generic infrastructure unless it is explicitly present in SOURCE VIEW INPUT.\n"
        )
    return ""


def _compact_semantic_report(semantic_report: dict, max_missing: int = 30) -> dict:
    """Keep the semantic fixer prompt small enough to avoid long stalled LLM calls."""
    missing_facts = semantic_report.get("missing_facts", [])
    if not isinstance(missing_facts, list):
        missing_facts = []
    missing_ids = [
        str(item).upper()
        for item in semantic_report.get("missing_fact_ids", [])
        if str(item).strip()
    ]
    note_only_ids = [
        str(item).upper()
        for item in semantic_report.get("note_only_fact_ids", [])
        if str(item).strip()
    ]

    selected_ids = []
    for fact_id in missing_ids + note_only_ids:
        if fact_id not in selected_ids:
            selected_ids.append(fact_id)
    selected_ids = selected_ids[:max_missing]

    by_id = {
        str(item.get("id") or "").upper(): item
        for item in missing_facts
        if isinstance(item, dict)
    }
    selected_facts = [by_id[fact_id] for fact_id in selected_ids if fact_id in by_id]

    return {
        "expected_count": semantic_report.get("expected_count"),
        "present_count": semantic_report.get("present_count"),
        "missing_count": semantic_report.get("missing_count"),
        "coverage_pct": semantic_report.get("coverage_pct"),
        "missing_fact_ids": selected_ids,
        "missing_facts": selected_facts,
        "note_only_fact_ids": [fact_id for fact_id in note_only_ids if fact_id in selected_ids],
        "unsupported_trace_ids": semantic_report.get("unsupported_trace_ids", []),
        "truncated": len(missing_ids) > len(selected_ids),
    }


def _compact_semantic_source(view_input: str, semantic_report: dict) -> str:
    missing_facts = semantic_report.get("missing_facts", [])
    return (
        "COMPACT SOURCE FOR TARGETED SEMANTIC FIXER.\n"
        "Use only the missing facts below and the current PlantUML. Do not infer new requirements.\n\n"
        f"MISSING / NOTE-ONLY FACTS:\n{_canonical_json({'facts': missing_facts})}\n\n"
        "If a fact is unclear, represent only the exact wording in its text/evidence_quote.\n"
    )


def run_mbse_view_generator(
    view_input: str,
    diagram_type: str,
    model: str = "gpt-4o-mini",
) -> tuple[str, str, dict, dict]:
    """MBSE-only direct PlantUML generator for one view.

    This intentionally does not reuse run_pipeline_slim: in complex MBSE mode
    the system design and view allocation are already done globally, so a
    second Analyst/Architect/Critic cycle per diagram tends to mix context and
    add unsupported assumptions.
    """
    llm = LLM(model=model)
    prompt = MBSE_CODER_PROMPTS.get(diagram_type, "")

    generator = Agent(
        role=f"MBSE {diagram_type.title()} View Generator",
        goal=f"Generate valid PlantUML for the {diagram_type} view from the allocated MBSE view input.",
        backstory=(
            "You are a precise PlantUML engineer working inside an MBSE pipeline. "
            "The system model and view allocation are already prepared. "
            "Your job is to render only this view without adding new domain facts."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    task = Task(
        description=(
            f"{prompt}\n\n"
            "SOURCE VIEW INPUT:\n"
            f"{view_input}\n\n"
            "Additional hard rules:\n"
            "- Use the requirements and view brief as the source of truth.\n"
            "- Do not invent scenarios, actors, components, data stores, reports, or exceptions.\n"
            "- Do not shrink the view to a summary: represent every traceable fact assigned to this view.\n"
            "- Add PlantUML trace comments for every represented fact using exactly: ' trace: REQ-001 REQ-002\n"
            "- If one UML element covers several facts, put all covered IDs in the trace comment above it.\n"
            "- If a requirement is ambiguous, add a short PlantUML note instead of dropping it.\n"
            f"{_diagram_semantic_representation_rules(diagram_type)}"
            "- Return only PlantUML code from @startuml to @enduml."
        ),
        expected_output="Only valid PlantUML code from @startuml to @enduml.",
        agent=generator,
    )

    crew = Crew(
        agents=[generator],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    crew.kickoff()

    usage = _extract_usage(crew, model)
    raw = task.output.raw
    return raw, "", usage, {"mbse_view_generator": raw}


def run_mbse_view_completeness_fixer(
    view_input: str,
    puml_code: str,
    diagram_type: str,
    model: str = "gpt-4o-mini",
) -> tuple[str, str, dict, dict]:
    """MBSE-only semantic rewrite: cover traceable facts without adding new ones."""
    llm = LLM(model=model)
    syntax_rules = MBSE_CODER_PROMPTS.get(diagram_type, "")

    fixer = Agent(
        role=f"MBSE {diagram_type.title()} Completeness Fixer",
        goal=f"Revise a {diagram_type} PlantUML view so it covers all assigned traceable requirement facts.",
        backstory=(
            "You are a strict UML/MBSE reviewer and PlantUML engineer. "
            "You preserve syntax validity, but your main job is semantic coverage: "
            "every traceable fact assigned to this view must be represented. "
            "You do not invent new facts."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    task = Task(
        description=(
            f"Revise this PlantUML {diagram_type} diagram for semantic completeness.\n\n"
            f"STRICT SYNTAX RULES:\n{syntax_rules}\n\n"
            f"SOURCE VIEW INPUT:\n{view_input}\n\n"
            f"CURRENT PLANTUML:\n{puml_code}\n\n"
            "Rules:\n"
            "- Return ONLY PlantUML code from @startuml to @enduml.\n"
            "- Represent every TRACEABLE REQUIREMENT FACT assigned to this view.\n"
            "- Add or preserve PlantUML trace comments for every represented fact using exactly: ' trace: REQ-001 REQ-002\n"
            "- If one UML element covers several facts, put all covered IDs in the trace comment above it.\n"
            "- Remove trace IDs that are not assigned to this view.\n"
            "- Do not summarize several requirement facts into one vague element when UML can show them explicitly.\n"
            "- Preserve enumerated values explicitly as enum values, labels, actions, messages, or notes depending on the view.\n"
            "- Remove elements that are not supported by the requirements or traceable facts.\n"
            "- Do not add new exceptions, integrations, UI, reports, authentication, payment gateways, or failure cases unless present in traceable facts.\n"
            f"{_diagram_semantic_representation_rules(diagram_type)}"
            "- Keep syntax valid for the requested diagram type."
        ),
        expected_output="Only revised PlantUML code from @startuml to @enduml.",
        agent=fixer,
    )

    crew = Crew(
        agents=[fixer],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    crew.kickoff()

    raw = task.output.raw
    usage = _extract_usage(crew, model)
    return raw, "", usage, {"mbse_completeness_fixer": raw}


def run_mbse_semantic_fixer(
    view_input: str,
    puml_code: str,
    diagram_type: str,
    semantic_report: dict,
    model: str = "gpt-4o-mini",
) -> tuple[str, str, dict, dict]:
    """Targeted semantic repair using concrete missing REQ ids."""
    llm = LLM(model=model)
    syntax_rules = MBSE_CODER_PROMPTS.get(diagram_type, "")
    compact_report = _compact_semantic_report(semantic_report)
    if compact_report.get("truncated"):
        return (
            puml_code,
            (
                "Semantic fixer skipped: too many missing/note-only facts for one targeted patch. "
                "The diagram is returned unchanged and coverage metrics should be used to diagnose the weak view."
            ),
            {},
            {"mbse_semantic_fixer": "skipped_too_many_missing_facts"},
        )
    compact_view_input = _compact_semantic_source(view_input, compact_report)

    fixer = Agent(
        role=f"MBSE {diagram_type.title()} Trace Coverage Fixer",
        goal=f"Patch a {diagram_type} PlantUML view so all missing traceable requirement facts are represented.",
        backstory=(
            "You are a strict MBSE traceability fixer. You do not redesign the diagram broadly. "
            "You add only the UML elements, messages, actions, notes, or trace comments needed to cover "
            "the listed missing REQ ids, while preserving valid existing content."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    task = Task(
        description=(
            f"Patch this PlantUML {diagram_type} diagram for trace coverage.\n\n"
            f"STRICT SYNTAX RULES:\n{syntax_rules}\n\n"
            f"SOURCE VIEW INPUT:\n{compact_view_input}\n\n"
            f"SEMANTIC REPORT WITH MISSING IDS:\n{_canonical_json(compact_report)}\n\n"
            f"CURRENT PLANTUML:\n{puml_code}\n\n"
            "Rules:\n"
            "- Return ONLY PlantUML code from @startuml to @enduml.\n"
            "- Cover every missing_fact_id in the semantic report.\n"
            "- Keep the patch narrow: add or adjust only elements needed for the listed missing IDs.\n"
            "- Each covered missing fact must have a trace comment using exactly: ' trace: REQ-001 REQ-002\n"
            "- Put the trace comment immediately above the element/message/action/note that represents the fact.\n"
            "- Preserve existing valid elements and valid trace comments.\n"
            "- Remove unsupported_trace_ids from comments unless the fact is explicitly assigned in SOURCE VIEW INPUT.\n"
            "- Treat note_only_fact_ids as still missing: move their trace IDs to native UML elements for this diagram type or add the missing element.\n"
            "- Do not invent domain facts beyond SOURCE VIEW INPUT.\n"
            f"{_diagram_semantic_representation_rules(diagram_type)}"
            "- Keep syntax valid for the requested diagram type.\n"
        ),
        expected_output="Only patched PlantUML code from @startuml to @enduml.",
        agent=fixer,
    )

    crew = Crew(
        agents=[fixer],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    crew.kickoff()

    raw = task.output.raw
    usage = _extract_usage(crew, model)
    return raw, "", usage, {"mbse_semantic_fixer": raw}


def _canonical_names(model: dict, keys: list[str]) -> list[str]:
    names: list[str] = []
    for key in keys:
        for item in model.get(key, []):
            name = _item_name(item)
            if name and name not in names:
                names.append(name)
    return names


_FACT_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "must", "will",
    "are", "can", "when", "then", "system", "requirements", "requirement",
}


def _fact_keywords(fact: dict) -> list[str]:
    text = f"{fact.get('text', '')} {fact.get('evidence_quote', '')}".lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]{3,}", text)
    result: list[str] = []
    for word in words:
        if word not in _FACT_STOP_WORDS and word not in result:
            result.append(word)
    return result[:12]


def _fact_covered_by_puml(fact: dict, puml_lower: str) -> bool:
    keywords = _fact_keywords(fact)
    if not keywords:
        return True
    matched = sum(1 for word in keywords if word in puml_lower)
    return matched >= max(1, min(3, round(len(keywords) * 0.35)))


def _compute_view_coverage(
    canonical_model: dict,
    diagrams: dict,
    requirement_facts: list[dict] | None = None,
) -> dict:
    if requirement_facts:
        by_type: dict = {}
        coverage_values: list[float] = []
        for dtype in ["class", "sequence", "component", "activity"]:
            semantic = _semantic_check_puml(
                str((diagrams.get(dtype) or {}).get("puml", "")),
                dtype,
                requirement_facts,
                allow_keyword_fallback=False,
            )
            coverage = semantic["coverage_pct"]
            coverage_values.append(coverage)
            by_type[dtype] = {
                "expected_count": semantic["expected_count"],
                "present_count": semantic["present_count"],
                "missing_count": semantic["missing_count"],
                "coverage_pct": coverage,
                "missing": [
                    f"{item.get('id')}: {item.get('text')}"
                    for item in semantic["missing_facts"][:30]
                ],
                "unsupported_trace_ids": semantic["unsupported_trace_ids"],
                "note_only_fact_ids": semantic.get("note_only_fact_ids", []),
            }
        return {
            "by_type": by_type,
            "avg_view_coverage_pct": round(sum(coverage_values) / len(coverage_values), 1) if coverage_values else 0.0,
        }

    expected = {
        "class": _canonical_names(canonical_model, ["entities", "external_systems"]),
        "sequence": _canonical_names(canonical_model, ["actors", "components", "external_systems"]),
        "component": _canonical_names(canonical_model, ["components", "external_systems"]),
        "activity": _canonical_names(canonical_model, ["actors", "components"]),
    }

    by_type: dict = {}
    coverage_values: list[float] = []
    for dtype, names in expected.items():
        puml = str((diagrams.get(dtype) or {}).get("puml", "")).lower()
        present = [name for name in names if name.lower() in puml]
        missing = [name for name in names if name.lower() not in puml]
        coverage = round(len(present) / len(names) * 100, 1) if names else 100.0
        coverage_values.append(coverage)
        by_type[dtype] = {
            "expected_count": len(names),
            "present_count": len(present),
            "missing_count": len(missing),
            "coverage_pct": coverage,
            "missing": missing,
        }

    return {
        "by_type": by_type,
        "avg_view_coverage_pct": round(sum(coverage_values) / len(coverage_values), 1) if coverage_values else 0.0,
    }


def _diagram_quality_issues(diagrams: dict) -> dict[str, list[str]]:
    """Small deterministic hygiene checks that the LLM consistency reviewer must not ignore."""
    issues: dict[str, list[str]] = {}

    for dtype in ["class", "sequence", "component", "activity"]:
        puml = str((diagrams.get(dtype) or {}).get("puml", ""))
        dtype_issues: list[str] = []
        start_count = len(re.findall(r"^\s*@startuml\s*$", puml, re.MULTILINE | re.IGNORECASE))
        end_count = len(re.findall(r"^\s*@enduml\s*$", puml, re.MULTILINE | re.IGNORECASE))
        malformed = sorted(set(re.findall(r"\bRE[_\s]+Q[-_\s]*\d+(?:-\d+)?\b", puml, re.IGNORECASE)))

        if start_count != 1 or end_count != 1:
            dtype_issues.append(f"Invalid PlantUML envelope count: @startuml={start_count}, @enduml={end_count}.")
        if malformed:
            dtype_issues.append(f"Malformed trace IDs: {', '.join(malformed[:10])}.")

        relation_count = len(re.findall(r"(-->|<--|\.\.>|\*--|o--|<\|--|->>|->|--|\.\.)", puml))
        block_count = len(re.findall(r"^\s*(?:abstract\s+class|class|interface|enum|component|database|queue|participant|actor)\b", puml, re.MULTILINE | re.IGNORECASE))

        if dtype == "class":
            enum_count = len(re.findall(r"^\s*enum\b", puml, re.MULTILINE | re.IGNORECASE))
            if enum_count == 0:
                dtype_issues.append("Class diagram has no enum declarations.")
            if block_count and relation_count < max(3, block_count // 2):
                dtype_issues.append("Class diagram has low relation density for the number of classifiers.")
        elif dtype == "sequence":
            alt_count = len(re.findall(r"^\s*alt\b", puml, re.MULTILINE | re.IGNORECASE))
            loop_count = len(re.findall(r"^\s*loop\b", puml, re.MULTILINE | re.IGNORECASE))
            activation_count = len(re.findall(r"^\s*activate\b", puml, re.MULTILINE | re.IGNORECASE))
            if alt_count > 15 and loop_count == 0:
                dtype_issues.append("Sequence diagram has many alt blocks and no loops.")
            if activation_count == 0:
                dtype_issues.append("Sequence diagram has no activation spans.")
        elif dtype == "component":
            interface_count = len(re.findall(r"^\s*interface\b", puml, re.MULTILINE | re.IGNORECASE))
            if interface_count == 0:
                dtype_issues.append("Component diagram has no explicit interfaces.")
            if block_count and relation_count < max(2, block_count // 2):
                dtype_issues.append("Component diagram has low dependency density.")
        elif dtype == "activity":
            stop_count = len(re.findall(r"^\s*stop\s*$", puml, re.MULTILINE | re.IGNORECASE))
            if stop_count > 1:
                dtype_issues.append("Activity diagram has multiple stop nodes; workflow may terminate inside alternatives.")

        issues[dtype] = dtype_issues

    return issues


def run_mbse_pipeline(requirements: str, model: str = "gpt-4o-mini") -> dict:
    """
    MBSE-пайплайн: генерирует комплект из 4 UML-диаграмм.
    Исходные требования остаются главным источником истины для каждой диаграммы,
    а Project Context используется как общий глоссарий для согласованности.

    Шаги:
      1. Requirement Fact Extractor — извлекает трассируемые атомарные факты требований
      2. Project Context Builder — извлекает общий JSON-глоссарий и сценарии
      3. Programmatic Context Validator — проверяет структуру и ссылки JSON
      4. Context Fixer — исправляет JSON, если валидация нашла проблемы
      5. MBSE System Designer — уточняет общий проектный контекст без PlantUML
      6. View Allocation Agent — распределяет требования по четырём view briefs
      7. Для каждого типа диаграммы — MBSE View Generator + Completeness Fixer
      8. Consistency Checker — сверяет диаграммы с Project Context, View Briefs и исходными требованиями

    Возвращает dict:
      {
        "canonical_model": dict,  # backward-compatible alias for project_context
        "model_issues": list[str],
        "view_briefs": {"class": dict, "sequence": dict, "component": dict, "activity": dict},
        "view_specs": {"class": str, "sequence": str, "component": str, "activity": str},  # фактические inputs
        "diagrams":   {"class": {"puml": str, "critique": str, "usage": dict}, ...},
        "consistency_report": str,
        "total_usage": dict,
      }
    """
    llm = LLM(model=model)
    total_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}

    # MBSE traceability layer: preserve requirement facts before any design compression.
    fact_extractor = Agent(
        role="Requirement Fact Extractor",
        goal="Extract traceable atomic requirement facts for a multi-view UML/MBSE generation pipeline.",
        backstory=(
            "You are a strict requirements analyst. You do not design the system and do not infer "
            "features beyond the text. Your task is to preserve all meaningful facts from the "
            "requirements so later agents cannot silently compress or invent the system."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    fact_task = Task(
        description=(
            "Extract atomic requirement facts. Return ONLY valid JSON without markdown.\n\n"
            f"REQUIREMENTS:\n{requirements}\n\n"
            "Return this exact structure:\n"
            "{\n"
            '  "facts": [\n'
            '    {"id": "REQ-001", "type": "actor|entity|attribute|relationship|operation|scenario|business_rule|state|component|interface|external_system", '
            '"text": "one atomic requirement fact", "evidence_quote": "exact substring copied from requirements", '
            '"must_appear_in": ["class", "sequence", "component", "activity"]}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Split the requirements into all meaningful atomic facts, but do not invent facts.\n"
            "- evidence_quote must be copied from the requirements text. Preserve wording, but line wraps from PDFs are acceptable.\n"
            "- Preserve enumerated values as facts, for example room types or service types.\n"
            "- Preserve lifecycle/process facts as separate facts: booking, check-in, stay services, checkout, payment, cleaning, cancellation policy, staff roles.\n"
            "- For use-case text, extract every Main Success Scenario step and every Extension/Alternative Flow step as a separate fact.\n"
            "- Assign main scenario, alternative flow, exception, and interaction facts to BOTH sequence and activity.\n"
            "- Assign system capabilities, commands, visible use-case operations, and externally visible service operations to class as operation/method facts; also to sequence/activity when they are interaction steps.\n"
            "- Assign external systems, interfaces, UI, datastores, and integration facts to component; also to sequence when messages are exchanged.\n"
            "- Assign domain entities, attributes, states, business rules, preconditions, and postconditions to class; also to activity when they affect the process.\n"
            "- must_appear_in is a traceability hint, not a size limit. It is better to assign a fact to two relevant views than to lose it.\n"
            "- Do not add gaps, assumptions, payment gateways, UI, dashboards, reports, authentication, identification, or failure cases unless explicitly present.\n"
        ),
        expected_output="Valid JSON object with a facts array, without markdown.",
        agent=fact_extractor,
    )

    fact_crew = Crew(
        agents=[fact_extractor],
        tasks=[fact_task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    fact_crew.kickoff()
    _add_usage(total_usage, _extract_usage(fact_crew, model))

    requirement_facts_raw = fact_task.output.raw
    try:
        requirement_facts, requirement_fact_issues = _normalise_requirement_facts(
            _extract_json_object(requirement_facts_raw),
            requirements,
        )
    except Exception as e:
        requirement_facts = []
        requirement_fact_issues = [f"Requirement fact extraction returned invalid JSON: {e}"]

    # ── 1. Project Context Builder ────────────────────────────────────────────
    model_builder = Agent(
        role="Project Context Builder",
        goal="Извлечь из требований общий JSON-глоссарий, сценарии и элементы согласованности для комплекта UML-диаграмм.",
        backstory=(
            "Ты системный аналитик и архитектор MBSE. Твоя задача — подготовить общую память проекта: "
            "единые имена, акторов, сущности, внешние системы, сценарии и бизнес-правила. "
            "Этот контекст не заменяет исходные требования, а помогает всем diagram-view использовать "
            "одни и те же термины."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    model_task = Task(
        description=(
            "Проанализируй требования и верни ТОЛЬКО валидный JSON без markdown и пояснений.\n"
            "Не ограничивай количество сущностей, сценариев или правил: включи все значимые элементы требований.\n\n"
            f"ТРЕБОВАНИЯ:\n{requirements}\n\n"
            f"TRACEABLE REQUIREMENT FACTS:\n{_canonical_json({'facts': requirement_facts})}\n\n"
            "JSON должен иметь ровно такую верхнеуровневую структуру:\n"
            "{\n"
            '  "system_name": "string",\n'
            '  "actors": [{"name": "string", "responsibilities": ["string"]}],\n'
            '  "entities": [{"name": "string", "description": "string", "attributes": [{"name": "string", "type": "string"}], "operations": ["string"]}],\n'
            '  "relationships": [{"source": "string", "target": "string", "type": "association|aggregation|composition|inheritance|dependency|realization", "description": "string", "multiplicity": "string"}],\n'
            '  "states": [{"name": "string", "owner": "string", "values": ["string"]}],\n'
            '  "scenarios": [{"name": "string", "primary_actor": "string", "steps": ["string"], "alternative_flows": ["string"], "exceptions": ["string"]}],\n'
            '  "components": [{"name": "string", "layer": "presentation|business|data|external|unspecified", "responsibilities": ["string"], "interfaces": ["string"]}],\n'
            '  "external_systems": [{"name": "string", "protocol_or_interface": "string", "responsibilities": ["string"]}],\n'
            '  "business_rules": ["string"],\n'
            '  "assumptions": ["string"],\n'
            '  "gaps": ["string"]\n'
            "}\n\n"
            "Правила:\n"
            "- Используй одинаковые имена объектов во всех секциях.\n"
            "- relationships.source и relationships.target должны ссылаться на имена из actors/entities/components/external_systems.\n"
            "- Системные возможности и команды из сценариев отражай в entities.operations или interfaces, если они являются видимыми операциями системы.\n"
            "- Не добавляй инфраструктуру, если она не следует из требований.\n"
            "- Все TRACEABLE REQUIREMENT FACTS должны быть отражены в actors/entities/relationships/states/scenarios/components/business_rules или gaps.\n"
            "- Не добавляй assumptions/gaps/exceptions, если для них нет evidence_quote в TRACEABLE REQUIREMENT FACTS.\n"
            "- Этот JSON является глоссарием и картой трассировки, а не заменой исходного текста требований.\n"
            "- Не используй null: если данных нет, ставь пустую строку или пустой список.\n"
        ),
        expected_output="Валидный JSON-объект Project Context без markdown.",
        agent=model_builder,
    )

    model_crew = Crew(
        agents=[model_builder],
        tasks=[model_task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    model_crew.kickoff()
    _add_usage(total_usage, _extract_usage(model_crew, model))

    canonical_model_raw = model_task.output.raw
    parse_error: str | None = None
    try:
        canonical_model = _normalise_canonical_model(_extract_json_object(canonical_model_raw))
    except Exception as e:
        parse_error = str(e)
        canonical_model = _empty_canonical_model()
        canonical_model["gaps"].append(f"Failed to parse model-builder JSON: {parse_error}")

    model_issues = _validate_canonical_model(canonical_model)
    if parse_error:
        model_issues.insert(0, f"JSON parse error: {parse_error}")

    # ── 2. Context Fixer при проблемах JSON ──────────────────────────────────
    if model_issues:
        fixer = Agent(
            role="Project Context Fixer",
            goal="Исправить JSON Project Context так, чтобы он был валидным и согласованным.",
            backstory=(
                "Ты исправляешь структуру глоссария, не добавляя неподтверждённые требования. "
                "Если проблему нельзя исправить из текста, фиксируй её в gaps."
            ),
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        fix_task = Task(
            description=(
                "Исправь Project Context. Верни ТОЛЬКО валидный JSON без markdown.\n\n"
                f"ИСХОДНЫЕ ТРЕБОВАНИЯ:\n{requirements}\n\n"
                f"TRACEABLE REQUIREMENT FACTS:\n{_canonical_json({'facts': requirement_facts})}\n\n"
                f"ТЕКУЩИЙ PROJECT CONTEXT:\n{_canonical_json(canonical_model)}\n\n"
                "ПРОБЛЕМЫ ВАЛИДАЦИИ:\n"
                + "\n".join(f"- {issue}" for issue in model_issues)
            ),
            expected_output="Валидный JSON-объект Project Context без markdown.",
            agent=fixer,
        )

        fix_crew = Crew(
            agents=[fixer],
            tasks=[fix_task],
            verbose=True,
            process=Process.sequential,
            output_log_file="agent_thoughts.log",
        )
        fix_crew.kickoff()
        _add_usage(total_usage, _extract_usage(fix_crew, model))

        try:
            fixed_model = _normalise_canonical_model(_extract_json_object(fix_task.output.raw))
            fixed_issues = _validate_canonical_model(fixed_model)
            canonical_model = fixed_model
            model_issues = fixed_issues
            canonical_model_raw = fix_task.output.raw
        except Exception as e:
            model_issues.append(f"Model fixer returned invalid JSON: {e}")

    # ── 3. View Allocation Agent ──────────────────────────────────────────────
    # MBSE System Designer: one global design pass before splitting views.
    system_design_raw = ""
    system_design_issues: list[str] = []
    designer = Agent(
        role="MBSE System Designer",
        goal="Refine the extracted Project Context into a coherent MBSE design model for a four-view UML set.",
        backstory=(
            "You are a strict MBSE system designer. You do not draw diagrams and do not write PlantUML. "
            "You align domain entities, roles, scenarios, rules, states, relationships, and logical components "
            "so that later UML views can be generated independently without mixing responsibilities."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    design_task = Task(
        description=(
            "Refine the current Project Context into a coherent MBSE system design JSON. "
            "Return ONLY valid JSON with exactly the same top-level schema as the current Project Context.\n\n"
            f"ORIGINAL REQUIREMENTS:\n{requirements}\n\n"
            f"TRACEABLE REQUIREMENT FACTS:\n{_canonical_json({'facts': requirement_facts})}\n\n"
            f"CURRENT PROJECT CONTEXT:\n{_canonical_json(canonical_model)}\n\n"
            "Rules:\n"
            "- The original requirements are the source of truth.\n"
            "- TRACEABLE REQUIREMENT FACTS are mandatory coverage items. Do not drop them during design refinement.\n"
            "- Preserve requirement-supported domain facts and remove unsupported assumptions.\n"
            "- Do not generate PlantUML, diagrams, markdown, or commentary.\n"
            "- Do not invent dispute flows, no-reservation flows, reports, dashboards, mobile apps, web apps, "
            "generic UI, generic databases, or technical layers unless the requirements explicitly imply them.\n"
            "- It is acceptable to introduce domain concepts directly implied by requirements, such as Reservation "
            "for booking rooms by dates or Bill for checkout payment.\n"
            "- Components must be logical capabilities from the requirements, not generic technical infrastructure.\n"
            "- Scenarios must be factual, ordered, and cover the main lifecycle described by the requirements.\n"
            "- For every exception, gap, assumption, component, business rule, and scenario, there must be support in TRACEABLE REQUIREMENT FACTS.\n"
            "- Keep names stable across actors, entities, relationships, scenarios, components, and states.\n"
            "- Use empty lists or empty strings instead of null.\n"
        ),
        expected_output="Valid JSON Project Context with the same schema, without markdown.",
        agent=designer,
    )

    design_crew = Crew(
        agents=[designer],
        tasks=[design_task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    design_crew.kickoff()
    _add_usage(total_usage, _extract_usage(design_crew, model))

    system_design_raw = design_task.output.raw
    try:
        designed_model = _normalise_canonical_model(_extract_json_object(system_design_raw))
        system_design_issues = _validate_canonical_model(designed_model)
        if not system_design_issues:
            canonical_model = designed_model
            canonical_model_raw = system_design_raw
            model_issues = []
    except Exception as e:
        system_design_issues = [f"System designer returned invalid JSON: {e}"]

    allocator = Agent(
        role="UML View Allocation Agent",
        goal="Распределить элементы требований по четырём UML-представлениям: class, sequence, component, activity.",
        backstory=(
            "Ты архитектор, который готовит чек-листы для разных UML-view. "
            "Ты не генерируешь диаграммы, а определяешь, какие элементы обязательно должны попасть "
            "в каждую диаграмму. Исходные требования остаются главным источником истины."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    allocation_task = Task(
        description=(
            "Распредели элементы требований по UML-представлениям. Верни ТОЛЬКО валидный JSON без markdown.\n\n"
            f"ИСХОДНЫЕ ТРЕБОВАНИЯ:\n{requirements}\n\n"
            f"TRACEABLE REQUIREMENT FACTS:\n{_canonical_json({'facts': requirement_facts})}\n\n"
            f"PROJECT CONTEXT:\n{_canonical_json(canonical_model)}\n\n"
            "JSON должен иметь ровно такую структуру:\n"
            "{\n"
            '  "class": {\n'
            '    "entities": [], "attributes": [], "methods": [], "states": [], "relationships": [], "business_rules": []\n'
            "  },\n"
            '  "sequence": {\n'
            '    "actors": [], "participants": [], "main_flow": [], "alternative_flows": [], "exceptions": [], "external_calls": []\n'
            "  },\n"
            '  "component": {\n'
            '    "components": [], "interfaces": [], "external_systems": [], "datastores": [], "dependencies": []\n'
            "  },\n"
            '  "activity": {\n'
            '    "swimlanes": [], "actions": [], "decisions": [], "parallel_flows": [], "start_conditions": [], "end_conditions": []\n'
            "  }\n"
            "}\n\n"
            "Правила:\n"
            "- Не ограничивай количество элементов: включи всё существенное из требований.\n"
            "- Распредели каждый TRACEABLE REQUIREMENT FACT в те view, где он указан в must_appear_in.\n"
            "- Не добавляй элементы, которых нет в требованиях или Project Context.\n"
            "- Используй одинаковые имена с Project Context.\n"
            "- Каждый список должен содержать строки или простые объекты с name/description.\n"
        ),
        expected_output="Валидный JSON-объект View Briefs без markdown.",
        agent=allocator,
    )

    allocation_crew = Crew(
        agents=[allocator],
        tasks=[allocation_task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    allocation_crew.kickoff()
    _add_usage(total_usage, _extract_usage(allocation_crew, model))

    try:
        view_briefs = _normalise_view_briefs(_extract_json_object(allocation_task.output.raw))
        view_briefs = _augment_view_briefs(view_briefs, canonical_model, requirement_facts)
        view_brief_issues = _validate_view_briefs(view_briefs)
    except Exception as e:
        view_briefs = _augment_view_briefs(_empty_view_briefs(), canonical_model, requirement_facts)
        view_brief_issues = [f"View allocation JSON parse error: {e}"]

    # ── 4. Генерация каждой диаграммы из требований + контекста + view brief ─
    view_specs = {
        dtype: _build_mbse_input(requirements, canonical_model, view_briefs.get(dtype, {}), dtype, requirement_facts)
        for dtype in ["class", "sequence", "component", "activity"]
    }

    diagrams: dict = {}
    for dtype in ["class", "sequence", "component", "activity"]:
        specialized_req = view_specs[dtype]
        puml, critique, usage, intermediates = run_mbse_view_generator(
            specialized_req,
            dtype,
            model,
        )
        _add_usage(total_usage, usage)
        fixed_puml, fixed_critique, fixed_usage, fixed_intermediates = run_mbse_view_completeness_fixer(
            specialized_req,
            puml,
            dtype,
            model,
        )
        _add_usage(total_usage, fixed_usage)
        intermediates.update(fixed_intermediates)
        puml = fixed_puml
        critique = fixed_critique or critique
        puml = repair_puml(clean_output(puml), dtype, strict=True)
        puml = _repair_trace_comments_for_covered_facts(puml, dtype, requirement_facts)
        semantic_coverage = _semantic_check_puml(puml, dtype, requirement_facts)
        semantic_usage = {}
        if semantic_coverage["missing_fact_ids"]:
            semantic_puml, semantic_critique, semantic_usage, semantic_intermediates = run_mbse_semantic_fixer(
                specialized_req,
                puml,
                dtype,
                semantic_coverage,
                model,
            )
            _add_usage(total_usage, semantic_usage)
            intermediates.update(semantic_intermediates)
            puml = repair_puml(clean_output(semantic_puml), dtype, strict=True)
            puml = _repair_trace_comments_for_covered_facts(puml, dtype, requirement_facts)
            critique = semantic_critique or critique
            semantic_coverage = _semantic_check_puml(puml, dtype, requirement_facts)

        diagram_usage = {
            "generation": usage,
            "completeness_fixer": fixed_usage,
            "semantic_fixer": semantic_usage,
        }
        diagrams[dtype] = {
            "puml": puml,
            "critique": critique,
            "usage": diagram_usage,
            "intermediates": intermediates,
            "semantic_coverage": semantic_coverage,
        }

    view_coverage = _compute_view_coverage(canonical_model, diagrams, requirement_facts)
    diagram_quality_issues = _diagram_quality_issues(diagrams)

    # ── 4. Consistency Checker ────────────────────────────────────────────────
    checker = Agent(
        role="Consistency Checker",
        goal="Проверить согласованность комплекта UML-диаграмм с Project Context и исходными требованиями.",
        backstory=(
            "Ты эксперт по UML, MBSE и системной согласованности. Ты сверяешь разные представления "
            "с единым источником истины и отделяешь реальные проблемы от допустимых различий между видами."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    all_puml = "\n\n".join(
        f"=== {dtype.upper()} DIAGRAM ===\n{diagrams[dtype]['puml']}"
        for dtype in ["class", "sequence", "component", "activity"]
    )

    check_task = Task(
        description=(
            "Проверь комплект UML-диаграмм относительно Project Context и исходных требований.\n\n"
            f"ИСХОДНЫЕ ТРЕБОВАНИЯ:\n{requirements}\n\n"
            f"TRACEABLE REQUIREMENT FACTS:\n{_canonical_json({'facts': requirement_facts})}\n\n"
            f"PROJECT CONTEXT:\n{_canonical_json(canonical_model)}\n\n"
            f"VIEW BRIEFS:\n{_canonical_json(view_briefs)}\n\n"
            "ПРОБЛЕМЫ ПРОГРАММНОЙ ВАЛИДАЦИИ PROJECT CONTEXT:\n"
            + ("\n".join(f"- {issue}" for issue in model_issues) if model_issues else "Проблем не найдено.")
            + "\n\n"
            "ПРОБЛЕМЫ ПРОГРАММНОЙ ВАЛИДАЦИИ VIEW BRIEFS:\n"
            + ("\n".join(f"- {issue}" for issue in view_brief_issues) if view_brief_issues else "Проблем не найдено.")
            + "\n\n"
            "ПРОБЛЕМЫ ПРОВЕРКИ TRACEABLE REQUIREMENT FACTS:\n"
            + ("\n".join(f"- {issue}" for issue in requirement_fact_issues) if requirement_fact_issues else "Проблем не найдено.")
            + "\n\n"
            f"ПРОГРАММНЫЕ МЕТРИКИ ПОКРЫТИЯ VIEW-SPEC:\n{_canonical_json(view_coverage)}\n\n"
            f"ПРОГРАММНЫЕ QUALITY/HYGIENE WARNINGS:\n{_canonical_json(diagram_quality_issues)}\n\n"
            f"{all_puml}\n\n"
            "Выдай отчёт строго в формате Markdown:\n\n"
            "## Покрытие требований и Project Context\n"
            "Какие ключевые сущности, сценарии, компоненты или внешние системы из требований/контекста отсутствуют в соответствующих диаграммах.\n\n"
            "Важно: если PROGRAMMATIC QUALITY/HYGIENE WARNINGS содержит непустые списки, нельзя писать, что проблем нет, и нельзя ставить 10/10.\n\n"
            "## Междиаграммные несогласованности\n"
            "Разные имена для одной сущности, конфликтующие ответственности, несовместимые связи или сценарии.\n\n"
            "## Ограничения входных требований\n"
            "Какие gaps/assumptions из Project Context влияют на качество диаграмм.\n\n"
            "## Общая согласованность (0–10)\n"
            "Оценка и краткий вывод с указанием главной причины снижения оценки."
        ),
        expected_output="Markdown-отчёт: покрытие требований/контекста, междиаграммные несогласованности, ограничения, оценка 0–10.",
        agent=checker,
    )

    check_crew = Crew(
        agents=[checker],
        tasks=[check_task],
        verbose=True,
        process=Process.sequential,
        output_log_file="agent_thoughts.log",
    )
    check_crew.kickoff()
    _add_usage(total_usage, _extract_usage(check_crew, model))

    return {
        "canonical_model": canonical_model,
        "canonical_model_raw": canonical_model_raw,
        "model_issues": model_issues,
        "requirement_facts": requirement_facts,
        "requirement_facts_raw": requirement_facts_raw,
        "requirement_fact_issues": requirement_fact_issues,
        "system_design_raw": system_design_raw,
        "system_design_issues": system_design_issues,
        "view_briefs": view_briefs,
        "view_brief_issues": view_brief_issues,
        "view_specs": view_specs,
        "view_coverage": view_coverage,
        "diagram_quality_issues": diagram_quality_issues,
        # Backward-compatible aliases for the existing UI/history code.
        "decomposed": view_specs,
        "decomposed_raw": _canonical_json(canonical_model),
        "diagrams": diagrams,
        "consistency_report": check_task.output.raw,
        "total_usage": total_usage,
    }
