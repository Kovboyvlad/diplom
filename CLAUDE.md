# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Activate the virtual environment before running anything:
```bash
source venv/Scripts/activate  # Windows bash
# or
venv\Scripts\activate.bat     # Windows cmd
```

Run the application:
```bash
python main.py
```

Install dependencies (no requirements.txt — install manually):
```bash
pip install crewai openai
```

## Architecture

Single-file project (`main.py`) implementing a **multi-agent AI pipeline** using [CrewAI](https://docs.crewai.com) to automatically generate a PlantUML architecture diagram from a natural-language system specification.

**Pipeline (sequential):**
1. `analyst` (Business Analyst) — parses the requirements text (`requirements_text`), extracts entities and interactions
2. `architect` (System Architect) — designs blocks and relationships from the analyst's output
3. `coder` (PlantUML Engineer) — writes valid PlantUML code based on the architect's output

Each agent's output is automatically passed as context to the next task. The final output is written to `diagram.puml`. Execution logs are written to `agent_thoughts.log`.

**Subject domain:** Smart Warehouse WMS — WMS Server, AGV robots, Inbound Zone, Storage Racks, Charging Station.

## Configuration

Credentials and model selection are set via `os.environ` at the top of `main.py`:
- `OPENAI_API_KEY` — OpenAI API key
- `OPENAI_MODEL_NAME` — model used by all agents (currently `gpt-4o-mini`)
- `HTTP_PROXY` / `HTTPS_PROXY` — proxy for outbound requests

These are currently hardcoded in source. Move them to a `.env` file and load with `python-dotenv` (already installed in venv) to avoid exposing credentials.

## Output Files

| File | Description |
|------|-------------|
| `diagram.puml` | Generated PlantUML diagram code |
| `agent_thoughts.log` | Full CrewAI execution log |
