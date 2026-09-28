# LLM-EvalBench

**LLM-EvalBench** is a Python-based LLM evaluation and benchmarking platform for assessing language-model responses across factual knowledge, reasoning, mathematics, coding, instruction following, and summarization. It uses a configurable JSONL benchmark and an **LLM-as-a-Judge** pipeline to generate criterion-level scores, rationales, and aggregate results displayed in a Next.js dashboard.

> **Status:** Active development. The frontend builds as a static site; public backend hosting and interactive deployment are not yet complete.

## Overview

The platform helps explore questions such as: Is a response correct and relevant? Is its reasoning sound? Does it follow instructions and faithfully summarize source material? How do scores vary by task and evaluation criterion?

```text
Benchmark dataset (JSONL)
          |
          v
  Candidate LLM (Ollama)
          |
          v
    Candidate response
          |
          v
   Judge LLM (Ollama) <--- Reference answer + requested criteria
          |
          v
 Structured criterion scores + rationales
          |
          v
 Python validation and score aggregation
          |
          v
      FastAPI backend
          |
          v
     Next.js dashboard
```

## Features

- **Multi-category benchmark:** Factual QA, reasoning, mathematics, coding, instruction following, and summarization.
- **Configurable dataset:** JSONL records specify prompts, reference answers, and evaluation criteria.
- **Separate candidate and judge models:** Select Ollama models to generate and evaluate answers.
- **Criterion-level evaluation:** Judge responses are parsed into scores (0–5) and written rationales.
- **Validated aggregation:** Python validates the judge's structured output and computes overall scores from the requested criteria.
- **Web interface:** Select candidate and judge models, run a benchmark, and inspect summary and per-question results.
- **Local inference:** Ollama enables development without paid model API dependencies.

Supported evaluation criteria include correctness, relevance, completeness, reasoning, instruction following, faithfulness, groundedness, hallucination, and logical consistency. **Note:** LLM judges can be inconsistent or biased; their scores are estimates rather than objective ground truth.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Backend | Python, FastAPI, Pydantic |
| Model runtime | Ollama |
| Evaluation | LLM-as-a-Judge, JSONL, structured JSON validation |
| Frontend | Next.js, React, TypeScript |
| Deployment | Static frontend export for GitHub Pages; backend hosting pending |

## Project Structure

```text
LLM-EvalBench/
├── analysis/                  # Analysis and judge-analysis scripts
├── data/
│   └── benchmark.jsonl       # Benchmark definitions
├── evaluation/
│   ├── evaluator.py
│   ├── llm_judge.py
│   └── judge_reliability.py
├── models/
│   ├── model_runner.py
│   └── test_ollama.py
├── dashboard/
│   └── llm-eval-dashboard/    # Next.js frontend
├── main.py                   # FastAPI application
├── requirements.txt
├── pyproject.toml
└── README.md
```

Some supporting scripts and files may vary as development continues.

## Benchmark Dataset

The current custom benchmark contains **20 questions**:

| Category | Questions | Evaluates |
| --- | ---: | --- |
| Factual QA | 4 | Knowledge and accurate explanations |
| Reasoning | 4 | Logical and multi-step reasoning |
| Mathematics | 3 | Calculations and mathematical reasoning |
| Coding | 3 | Programming knowledge and behavior |
| Instruction following | 3 | Explicit format and content constraints |
| Summarization | 3 | Faithful and concise summaries |

Example JSONL record:

```json
{
  "id": "fact_001",
  "category": "factual_qa",
  "difficulty": "easy",
  "source": "custom",
  "tags": ["geography", "basic_knowledge"],
  "prompt": "What is the capital of Australia?",
  "reference_answer": "Canberra.",
  "evaluation_criteria": ["correctness", "relevance"]
}
```

Each question can request different evaluation criteria, allowing task-specific assessment.

## How Evaluation Works

1. **Load** a question and its reference answer from `data/benchmark.jsonl`.
2. **Generate** a candidate response with the selected Ollama model.
3. **Evaluate** the response with a separate judge model, using the prompt, reference answer, and requested criteria.
4. **Validate** the judge's JSON response and ensure the requested criteria are present.
5. **Aggregate** validated criterion scores in Python and display the result in the dashboard.

Example judge output (illustrative):

```json
{
  "criteria_scores": {
    "correctness": {
      "score": 5,
      "rationale": "Provides the correct answer: Canberra."
    },
    "relevance": {
      "score": 5,
      "rationale": "Directly answers the question."
    }
  },
  "overall_rationale": "The response is accurate and relevant."
}
```

The backend calculates a question's overall score as the arithmetic mean of its validated criterion scores, rather than accepting an overall number from the judge.

## Getting Started

### Prerequisites

- Python and pip (or another compatible Python environment manager)
- Node.js and npm
- [Ollama](https://ollama.com/) running locally
- Enough local compute and memory for your selected models

### 1. Clone and install backend dependencies

```bash
git clone https://github.com/YOUR_USERNAME/LLM-EvalBench.git
cd LLM-EvalBench
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

Replace `YOUR_USERNAME` with the repository owner's GitHub username.

### 2. Prepare Ollama

Start Ollama, then install example models:

```bash
ollama pull qwen3:4b
ollama pull gemma:7b
ollama list
```

The available model variants and resource requirements depend on your Ollama installation. The application can discover local models and may pull selected catalog models when configured to do so.

### 3. Start FastAPI

From the repository root:

```bash
uvicorn main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs

### 4. Start Next.js

Open a second terminal:

```bash
cd dashboard/llm-eval-dashboard
npm install
npm run dev
```

Open http://localhost:3000. For local development, configure the frontend to call the running FastAPI server at `http://127.0.0.1:8000`.

### 5. Run a benchmark

1. Choose a **Prompt Model** (candidate model).
2. Choose a **Judge Model**.
3. Run the benchmark.
4. Inspect summary metrics, per-question responses, criterion scores, and rationales.

## API Overview

| Endpoint | Purpose |
| --- | --- |
| `GET /` | API information |
| `GET /api/health` | Health check |
| `GET /api/models` | Ollama model catalog and installation status |
| `POST /api/session/create` | Start a frontend evaluation session |
| `POST /api/session/cleanup` | Clean up eligible session-pulled models |
| `POST /api/benchmark` | Access benchmark functionality |
| `POST /api/evaluate` | Evaluate a candidate response |
| `POST /api/run-benchmark` | Generate and evaluate benchmark responses |

Refer to `/docs` for the current request and response schemas.

## Results and Limitations

During development, a Qwen3/Gemma benchmark run displayed **15 successful evaluations and 5 failed questions**, with a **4.77/5 mean score among the scored questions**. This is a *partial run*, **not** a validated 20-question performance result. Failures must be investigated before drawing comparative conclusions.

Other limitations:

- The current benchmark is small and custom-built; it is not a representative measure of general model ability.
- LLM-as-a-Judge scores can be sensitive to judge model, prompt design, and calibration.
- The system should report failed and unscored questions separately from evaluated questions.
- Local Ollama inference depends on the host machine and installed models.
- GitHub Pages can host the static dashboard but **cannot run FastAPI or Ollama**. A public interactive deployment requires a separately hosted backend and inference service.

## Deployment

The Next.js frontend supports static export for GitHub Pages. A static site can display the interface, but live benchmarking requires an accessible backend and model runtime. Do not point a publicly deployed frontend at `127.0.0.1`: that address refers to each visitor's own device.

For local frontend production builds:

```bash
cd dashboard/llm-eval-dashboard
npm run build
```

With `output: "export"` configured, the generated `out/` directory is the static deployment artifact. Configure the GitHub Pages repository `basePath` and a reachable production API URL before publishing an interactive experience.

## Portfolio Focus

This project demonstrates practical work in benchmark design, Python data processing, local LLM integration, structured evaluation, API development, and communicating model performance **with appropriate caveats**.

## License

## License

This project is licensed under the Apache License 2.0.

See the [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)(LICENSE) file for the full license text.
