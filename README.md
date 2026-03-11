# interview-prep-agent

A small stateful LLM agent that turns a job description and a resume into a structured interview-prep report.

## What it does

Given:

- a plain-text job description
- a plain-text resume

the agent produces a markdown report with:

- estimated fit score
- extracted job requirements
- resume evidence matched to those requirements
- weak areas / gaps
- suggested talking points

## Why this project exists

This project is intentionally small, but it demonstrates a real agent pattern rather than a one-shot prompt.

It shows:

- mutable application state
- LLM-driven action selection
- deterministic tool execution
- structured intermediate artifacts
- deterministic final rendering
- testability through fake / mocked model behavior

## Architecture

The system is split into five layers:

### 1. `src/agent.py`
Application entry point.

Responsibilities:

- parse CLI args
- load input files
- initialize the LLM
- initialize the agent state
- run the agent loop
- write the final markdown report

### 2. `src/agent_state.py`
Shared mutable state.

This is the agent's real memory between model calls. It stores:

- `jd_text`
- `resume_text`
- `notes`
- `artifacts`
- `tool_history`

Important design point:
the LLM does not directly mutate state. It proposes actions. Python executes them and updates state.

### 3. `src/llm.py`
Thin model adapter.

Responsibilities:

- validate API configuration
- submit prompts to the provider
- return raw model output
- surface quota / timeout / connection failures with clearer errors

This keeps provider-specific code isolated from the rest of the agent.

### 4. `src/tools.py`
Deterministic capability layer.

Current tools:

- `extract_jd_requirements`
- `find_resume_evidence`
- `score_resume_fit`
- `render_report`
- `dump_state_summary`

These tools do the concrete work. The LLM decides which tool to call next.

### 5. `src/agent_loop.py`
Orchestration layer.

Responsibilities:

- build the current prompt from state
- ask the LLM for the next action
- validate that action
- execute tools
- update state
- stop when the report is finished

This is the core agent loop:
think → act → observe → repeat

## Execution flow

1. Load `jd.txt` and `resume.txt`
2. Build an `AgentState`
3. Ask the model for the next action
4. Execute a tool if requested
5. Store results in `artifacts` and `tool_history`
6. Repeat until the report is rendered
7. Write `report.md`

## Example usage

```bash
python src/agent.py --jd jd.txt --resume resume.txt --out report.md