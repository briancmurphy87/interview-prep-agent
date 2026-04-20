# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the agent
python src/agent.py \
  --jd job_specs/netflix_swe5.txt \
  --resume resume_corpus/raw_resume.txt \
  --corpus resume_corpus \
  --out-resume output/targeted_resume.txt \
  --out-report output/report.md

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_tools.py

# Run a specific test
pytest tests/test_tools.py::test_score_resume_fit_creates_structured_fit_artifact
```

**Required env var:** `OPENAI_API_KEY` in `.env` (loaded via `python-dotenv`).

**Dependencies:** `openai`, `python-dotenv`, `pytest` — no `requirements.txt` exists; install manually.

## Architecture

This is a RAG + agentic pipeline that generates targeted resumes and evaluation reports. The system has two phases:

### Phase 1: Agent Loop (`agent_loop.py`)
An LLM agent (max 8 iterations) is given tools and orchestrates the workflow by emitting JSON tool calls. The agent is expected to call tools in this order:
1. `load_resume_corpus` — scans `resume_corpus/` for `(jd.md, resume_variant.txt)` pairs
2. `extract_jd_requirements` — deterministic keyword/phrase extraction from JD
3. `retrieve_similar_resume_examples` — keyword-overlap scoring to rank corpus examples
4. `score_resume_fit` — deterministic requirement↔resume matching; **runs before generation** to produce an evidence grounding block
5. `generate_target_resume` — LLM call using top examples as style guides + verified evidence snippets from step 4

Tool dispatch uses a `TOOLS` dict in `tools.py`. Tools that need the LLM instance (`generate_target_resume`, `evaluate_target_resume`) receive it injected by `agent_loop.py` at dispatch time.

### Phase 2: Deterministic Post-Loop Pipeline (`agent.py` main)
After the agent loop completes, `agent.py` always runs these unconditionally (if not already in artifacts):
1. `tool_score_resume_fit` — deterministic requirement↔resume matching with phrase/token/semantic scoring
2. `tool_evaluate_target_resume` — LLM-as-judge returning structured JSON scores across 5 dimensions
3. `tool_revise_target_resume` — if `overall_score < REVISION_THRESHOLD` (default 70), asks the LLM to revise the draft using evaluator feedback and re-evaluates the revision; writes `revised_resume_txt`, `revision_evaluation_json`, `revision_metadata_json`
4. `tool_render_report` — assembles all artifacts into a markdown report (includes revision section)

The output file (`--out-resume`) contains the best available draft: the revised version if revision was triggered, the initial draft otherwise.

### State (`agent_state.py`)
`AgentState` is the single shared object passed everywhere. It accumulates:
- **Inputs:** `jd_text`, `resume_text`
- **Corpus:** `corpus_examples: list[CorpusExample]`
- **Artifacts dict:** grows through the pipeline with typed keys (see `ArtifactKey` in `agent_state.py`):
  - `corpus_summary_json`, `requirements_json`, `retrieved_examples_json`
  - `fit_analysis_json`, `target_resume_txt`, `generation_metadata_json`
  - `resume_evaluation_json` (initial), `revised_resume_txt`, `revision_evaluation_json`, `revision_metadata_json`
  - `report_md`
- **Tool history:** full log of every tool call + result for auditability

### Key Design Decisions
- **Deterministic retrieval:** No embeddings/FAISS — uses regex tokenization + keyword overlap scoring (`_keyword_overlap_score` in `tools.py`)
- **Evidence-grounded generation:** `score_resume_fit` runs before `generate_target_resume` inside the agent loop. The generation prompt receives verified evidence snippets per requirement and explicit gap warnings via `_build_evidence_grounding_block()`.
- **Corpus skip warnings:** `tool_load_resume_corpus` reports skipped directories with reasons (`missing resume_variant.txt`, etc.) in the `skipped` list of its return payload.
- **Revision pass:** `tool_revise_target_resume` implements generate → evaluate → revise. The threshold is `REVISION_THRESHOLD = 70` in `tools.py`. The original `resume_evaluation_json` is preserved; the revision evaluation goes to `revision_evaluation_json`.
- **Graceful degradation:** If agent exits without generating a resume, `agent_loop.py` forces a fallback `generate_target_resume` call before returning.
- **LLM wrapper:** `llm.py` wraps OpenAI Responses API (`gpt-4.1-mini` default). Provides `complete()` (raw text) and `complete_json()` (parsed dict). Retries `APITimeoutError` and `APIConnectionError` up to `MAX_RETRIES` times with exponential back-off. Swap model via `--model` CLI arg.

## Test Structure

- `tests/test_tools.py` — unit tests for deterministic tools (no LLM calls)
- `tests/test_agent_loop.py` — integration tests using `FakeLLM` with scripted responses
- `tests/test_real_case.py` — end-to-end test using real Netflix SWE5 fixtures in `tests/fixtures/netflix_swe5/`