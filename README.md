
# Interview Prep Agent

A lightweight AI agent that generates **targeted resumes for specific job descriptions** using prior resume examples and automatically produces a detailed **fit and quality evaluation report**.

The system demonstrates a practical architecture used in real AI systems:

- Retrieval-augmented generation
- Tool-based agent loops
- Deterministic evaluation pipelines
- LLM-as-judge artifact evaluation
- Reproducible batch evaluation harness

---

# Overview

Given:

- a **target job description**
- a **raw base resume**
- a **corpus of previously tailored resumes**

the agent:

1. Extracts the most important job requirements
2. Retrieves similar examples from the resume corpus
3. Scores resume fit and identifies coverage gaps
4. Generates a **targeted resume** conditioned on verified evidence
5. Uses an **LLM-based evaluator** to critique the resume
6. Optionally **revises the draft** using evaluator feedback (if score < threshold)
7. Produces a detailed **review report** with both initial and revised scores

---

# Example Output

## input
```shell    
--jd
job_specs/netflix_swe5.txt
--resume
resume_corpus/raw_resume.txt
--corpus
resume_corpus
--out-resume
job_specs/netflix_swe5.targeted_resume.txt
--out-report
job_specs/netflix_swe5.report.md
```

## output 1 - targeted resume

will output the file here: 

![netflix_swe5.targeted_resume.txt](job_specs/netflix_swe5.targeted_resume.txt)

screenshot: 

![img.png](resources/example_output.targeted_resume.png)

## output 2 - role fit and resume evaluation report
will output the file here: 

![netflix_swe5.report.md](job_specs/netflix_swe5.report.md)

screenshot: 

![img.png](resources/example_output.evaluation_report.png)

# Example Workflow

```
job description
       +
raw resume
       +
resume corpus
       ↓
agent retrieves similar resume examples
       ↓
score fit + identify evidence gaps       <- grounds generation in real data
       ↓
generate targeted resume (evidence-conditioned)
       ↓
LLM resume evaluation
       ↓
if score < threshold: revise using evaluator feedback → re-evaluate
       ↓
final report (initial score + revised score when revision triggered)
```

Outputs:

```
targeted_resume.txt    (revised draft if revision triggered, initial draft otherwise)
resume_review_report.md
```

---

# Project Structure

```
src/
  agent.py          CLI entry point (single job run); runs post-loop pipeline
  agent_loop.py     agent orchestration loop (max 8 iters, tool dispatch)
  agent_state.py    shared state object; ArtifactKey covers all artifact names
  llm.py            LLM wrapper — complete() / complete_json(), retry backoff, call_log
  tools.py          all tools: retrieval, scoring, generation, evaluation, revision, report
  observability.py  run artifact assembly (timing, tokens, cost)
  run_evals.py      batch evaluation harness

resume_corpus/
  raw_resume.txt
  google_cloud_ai_ml/
  netflix_swe_n_tech/
  ...

evals/
  cases/
    netflix_swe_n_tech/
      jd.txt
      raw_resume.txt
      expected_requirements.json
    google_cloud_ai_ml/
      ...
    databricks_ml_ai_finserv/
      ...
  outputs/            per-case artifacts written here at runtime
  summaries/          aggregate summary JSONs written here at runtime
```

---

# Key Concepts Demonstrated

## Agent + Tool Architecture

The agent does not directly generate everything.

Instead it decides which tools to call, in this order:

- `load_resume_corpus`
- `extract_jd_requirements`
- `retrieve_similar_resume_examples`
- `score_resume_fit`          ← runs before generation to surface evidence
- `generate_target_resume`    ← conditioned on verified evidence and gaps

This keeps reasoning separated from implementation.

---

## Retrieval-Augmented Resume Generation

Instead of generating a resume from scratch, the agent retrieves relevant examples from a **resume corpus**.

This dramatically improves the quality of the generated resume.

---

## Evidence-Grounded Generation

Fit scoring runs **before** generation, not after.

The generation prompt receives:
- verified resume snippets per requirement
- a list of uncovered gaps

This anchors the LLM to real candidate evidence and reduces hallucination risk.
If fit scoring has not yet run (e.g. agent fallback), generation still proceeds with raw resume alone.

---

## Revision Pass

After the initial draft is generated and evaluated, the pipeline checks whether the overall score meets the revision threshold (default: 70/100).

If the score is below the threshold:

1. The LLM is given the original draft, the evaluator's red flags, and suggested improvements
2. It produces a revised draft
3. The revised draft is independently re-evaluated using the same rubric
4. Both scores are preserved — `resume_evaluation_json` (initial) and `revision_evaluation_json` (revised)
5. The report includes an initial score → revised score comparison

The output file always contains the **best available draft** (revised if triggered, initial otherwise).

This pattern — generate → judge → revise — is one of the most credible AI workflow patterns you can demonstrate.

---

## Deterministic Evaluation Pipeline

After the resume is generated, Python code performs:

- requirement matching
- evidence extraction
- fit scoring (if not already done in the agent loop)

This ensures evaluation remains reproducible and explainable.

---

## LLM-as-Judge Evaluation

A second LLM pass evaluates the generated resume across multiple dimensions:

- JD alignment
- keyword coverage
- clarity
- exaggeration risk
- ATS compatibility

This produces structured critique and suggested improvements.

---

## Batch Evaluation Harness

`src/run_evals.py` runs the full pipeline (including the revision pass) across multiple eval cases and aggregates results into a single summary.

Each eval case lives under `evals/cases/<slug>/` and contains:

- `jd.txt` — the target job description
- `raw_resume.txt` — the base resume input
- `expected_requirements.json` — optional keyword expectations used for pass/fail validation

The summary reports:

| Metric | Description |
|---|---|
| `average_overall_score` | Mean final score (revised score if revision triggered, initial otherwise) |
| `average_jd_alignment` | Mean JD alignment score |
| `average_keyword_coverage` | Mean keyword coverage score |
| `average_exaggeration_safety` | Mean exaggeration safety score (higher = safer) |
| `average_ats_compatibility` | Mean ATS compatibility score |
| `average_fit_score` | Mean deterministic fit score |
| `retrieval_hit_rate` | Fraction of cases where corpus retrieval returned ≥1 match |
| `num_failed` | Cases that raised an uncaught exception |

Per-case outputs in `evals/outputs/<slug>/` include `revision_metadata.json` and `revision_evaluation.json` when revision was triggered.

This turns the project from "I built an agent" into "I built and evaluated an AI workflow with reproducible cases."

---

## Corpus Loader Skip Warnings

`tool_load_resume_corpus` reports exactly which corpus directories were skipped and why:

```json
{
  "num_examples_loaded": 3,
  "num_examples_skipped": 2,
  "slugs": ["google_cloud_ai_ml", "netflix_swe_n_tech", "databricks_ml_ai_finserv"],
  "skipped": [
    {"slug": "confluent", "reason": "missing resume_variant.txt"},
    {"slug": "empty_dir", "reason": "missing jd.txt/jd.md and resume_variant.txt"}
  ]
}
```

This makes corpus gaps immediately visible instead of silently dropping entries.

---

## LLM Wrapper

`llm.py` provides two methods:

- `complete(system, user) → str` — raw text response; retries `APITimeoutError` and `APIConnectionError` up to `MAX_RETRIES` times with exponential back-off
- `complete_json(system, user) → dict` — parses and returns JSON; raises `ValueError` on parse failure

Every API attempt is logged to `llm.call_log` with timing, token counts, estimated cost, and (on retries) the attempt number. This feeds the observability run artifact.

---

# Installation

Create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Set your OpenAI API key:

```
export OPENAI_API_KEY="your_api_key_here"
```

---

# Usage

## Single job run

```
python -m src.agent \
  --jd job_specs/netflix_swe5.txt \
  --resume resume_corpus/raw_resume.txt \
  --corpus resume_corpus \
  --out-resume job_specs/netflix_swe5.targeted_resume.txt \
  --out-report job_specs/netflix_swe5.report.md
```

Outputs:

```
job_specs/netflix_swe5.targeted_resume.txt
job_specs/netflix_swe5.report.md
```

## Batch evaluation

Run all cases under `evals/cases/` and write outputs + a summary:

```
python -m src.run_evals \
  --eval-dir evals/cases \
  --out-dir  evals/outputs \
  --corpus   resume_corpus \
  --model    gpt-4.1-mini
```

Per-case outputs are written to `evals/outputs/<case_slug>/`:

```
target_resume.txt         initial tailored resume
revised_resume.txt        revised draft (present only when revision was triggered)
report.md                 full fit + evaluation + revision report
requirements.json         extracted JD requirements
retrieved_examples.json   corpus retrieval results
evaluation.json           LLM evaluator scores for the initial draft
revision_evaluation.json  LLM evaluator scores for the revised draft (when triggered)
revision_metadata.json    revision decision: triggered, initial_score, revised_score, delta
run_artifact.json         observability trace (tool spans, LLM spans, cost)
run_metadata.json         timing, tool counts, notes, validation result
```

A summary JSON is written to `evals/summaries/summary_<timestamp>.json` with aggregate metrics across all cases.

---

# Example Output Artifacts

### Targeted Resume

A rewritten resume tailored for the specific job description.

### Review Report

Includes:

- extracted job requirements
- resume evidence snippets
- fit analysis
- missing requirements
- LLM quality evaluation
- improvement suggestions

---

# Why This Project Exists

Many AI demos stop at generation.

Real systems require:

- retrieval
- structured reasoning
- evaluation pipelines
- automated critique

This project demonstrates how those pieces fit together in a practical workflow.

---

# Possible Extensions

Future improvements could include:

- vector search for corpus retrieval (replacing keyword overlap)
- multi-turn revision loops (currently one revision pass per run)
- baseline comparison mode: score raw prompt vs. prompt + corpus vs. prompt + corpus + evidence vs. full pipeline with revision
- ATS keyword optimization
- resume diff visualization (initial vs. revised draft)
- recruiter-style scoring rubrics
- automatic cover letter generation
- trend analysis across eval summary runs to track prompt improvements over time

---

# Disclaimer

Job descriptions included in this repository are publicly available postings used solely for demonstration purposes.

Company names may be anonymized in some cases.
