
# Interview Prep Agent

A lightweight AI agent that generates **targeted resumes for specific job descriptions** using prior resume examples and automatically produces a detailed **fit and quality evaluation report**.

The system demonstrates a practical architecture used in real AI systems:

- Retrieval-augmented generation
- Tool-based agent loops
- Deterministic evaluation pipelines
- LLM-as-judge artifact evaluation

---

# Overview

Given:

- a **target job description**
- a **raw base resume**
- a **corpus of previously tailored resumes**

the agent:

1. Extracts the most important job requirements
2. Retrieves similar examples from the resume corpus
3. Generates a **targeted resume**
4. Scores how well the resume fits the job
5. Uses an **LLM-based evaluator** to critique the resume
6. Produces a detailed **review report**

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
generate targeted resume
       ↓
fit analysis
       ↓
LLM resume evaluation
       ↓
final report
```

Outputs:

```
targeted_resume.txt
resume_review_report.md
```

---

# Project Structure

```
src/
  agent.py          CLI entry point
  agent_loop.py     agent orchestration loop
  agent_state.py    shared state object
  llm.py            LLM wrapper
  tools.py          agent + evaluation tools

job_specs/
  netflix_swe5.txt
  ...

resume_corpus/
  raw_resume.txt
  google_cloud_ai_ml/
  databricks_data_eng/
  ...
```

---

# Key Concepts Demonstrated

## Agent + Tool Architecture

The agent does not directly generate everything.

Instead it decides which tools to call:

- `load_resume_corpus`
- `extract_jd_requirements`
- `retrieve_similar_resume_examples`
- `generate_target_resume`

This keeps reasoning separated from implementation.

---

## Retrieval-Augmented Resume Generation

Instead of generating a resume from scratch, the agent retrieves relevant examples from a **resume corpus**.

This dramatically improves the quality of the generated resume.

---

## Deterministic Evaluation Pipeline

After the resume is generated, Python code performs:

- requirement matching
- evidence extraction
- fit scoring

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

Example:

```
python src/agent.py \
  --jd job_specs/netflix_swe5.txt \
  --resume resume_corpus/raw_resume.txt \
  --corpus resume_corpus \
  --out job_specs/netflix_swe5.targeted_resume.txt \
  --report-out job_specs/netflix_swe5.report.md
```

Outputs:

```
job_specs/netflix_swe5.targeted_resume.txt
job_specs/netflix_swe5.report.md
```

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

- automatic resume revision loops
- vector search for corpus retrieval
- ATS keyword optimization
- resume diff visualization
- recruiter-style scoring rubrics
- automatic cover letter generation

---

# Disclaimer

Job descriptions included in this repository are publicly available postings used solely for demonstration purposes.

Company names may be anonymized in some cases.
