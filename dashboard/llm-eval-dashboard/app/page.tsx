"use client";

import { useEffect, useState } from "react";


// ============================================================
// TYPES
// ============================================================

type ModelInfo = {
  id: string;
  name: string;
  provider: string;
  source: string;
  installed: boolean;
  model_type: string;
  benchmark_compatible: boolean;
};

type CriterionScore = {
  score: number;
  rationale: string;
};

type Evaluation = {
  overall_score: number;
  criteria_scores: Record<string, CriterionScore>;
  overall_rationale: string;
  judge_model?: string;
};

type BenchmarkResult = {
  summary: {
    model: string;
    judge_model: string;
    total_questions: number;
    successful_questions: number;
    failed_questions: number;
    scored_questions: number;
    overall_score: number | null;
  };

  results: Array<{
    question_id: string;
    question: string;
    category: string;
    difficulty: string;
    reference_answer: string;
    criteria: string[];
    model_response: string | null;
    evaluation: Evaluation | null;
    error: string | null;
  }>;
};


// ============================================================
// CONFIG
// ============================================================

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";


// ============================================================
// PAGE
// ============================================================

export default function Home() {

  // ----------------------------------------------------------
  // MODEL STATE
  // ----------------------------------------------------------

  const [models, setModels] = useState<ModelInfo[]>([]);

  const [selectedModel, setSelectedModel] =
    useState("");

  const [selectedJudge, setSelectedJudge] =
    useState("");


  // ----------------------------------------------------------
  // SESSION STATE
  // ----------------------------------------------------------

  const [sessionId, setSessionId] =
    useState<string | null>(null);


  // ----------------------------------------------------------
  // BENCHMARK STATE
  // ----------------------------------------------------------

  const [benchmarkResult, setBenchmarkResult] =
    useState<BenchmarkResult | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [loadingModels, setLoadingModels] =
    useState(true);

  const [error, setError] =
    useState("");

  const [status, setStatus] =
    useState("");

  const [expandedQuestion, setExpandedQuestion] =
    useState<string | null>(null);


  // ==========================================================
  // LOAD MODELS
  // ==========================================================

  useEffect(() => {

    let cancelled = false;

    async function loadModels() {

      try {

        setLoadingModels(true);
        setError("");

        const response = await fetch(
          `${API_BASE_URL}/api/models`,
          {
            method: "GET",
            cache: "no-store",
          }
        );

        if (!response.ok) {
          throw new Error(
            `Failed to load models (${response.status})`
          );
        }

        const data = await response.json();

        if (cancelled) {
          return;
        }

        const compatibleModels: ModelInfo[] =
          (data.models || []).filter(
            (model: ModelInfo) =>
              model.benchmark_compatible === true
          );

        setModels(compatibleModels);

      } catch (err) {

        if (cancelled) {
          return;
        }

        setError(
          err instanceof Error
            ? err.message
            : "Failed to load Ollama models."
        );

      } finally {

        if (!cancelled) {
          setLoadingModels(false);
        }
      }
    }

    loadModels();

    return () => {
      cancelled = true;
    };

  }, []);


  // ==========================================================
  // CREATE SESSION
  // ==========================================================

  useEffect(() => {

    let cancelled = false;

    async function createSession() {

      try {

        const response = await fetch(
          `${API_BASE_URL}/api/session/create`,
          {
            method: "POST",
          }
        );

        if (!response.ok) {
          throw new Error(
            `Failed to create session (${response.status})`
          );
        }

        const data = await response.json();

        if (!cancelled) {
          setSessionId(data.session_id);
        }

      } catch (err) {

        if (!cancelled) {

          setError(
            err instanceof Error
              ? err.message
              : "Failed to create session."
          );

        }
      }
    }

    createSession();

    return () => {
      cancelled = true;
    };

  }, []);


  // ==========================================================
  // CLEANUP TEMPORARY MODELS
  // ==========================================================

  useEffect(() => {

    if (!sessionId) {
      return;
    }

    const cleanup = () => {

      const payload = JSON.stringify({
        session_id: sessionId,
      });

      // keepalive allows the request to continue while the
      // browser is refreshing/closing the page.
      fetch(
        `${API_BASE_URL}/api/session/cleanup`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: payload,
          keepalive: true,
        }
      ).catch(() => {
        // Browser may terminate the request during shutdown.
      });
    };

    window.addEventListener(
      "pagehide",
      cleanup
    );

    return () => {

      window.removeEventListener(
        "pagehide",
        cleanup
      );

    };

  }, [sessionId]);


  // ==========================================================
  // MODEL CHANGE
  // ==========================================================

  function handleModelChange(
    value: string
  ) {

    setSelectedModel(value);

    // Changing either model invalidates the previous report.
    setBenchmarkResult(null);

    setError("");
    setStatus("");
    setExpandedQuestion(null);
  }


  // ==========================================================
  // JUDGE CHANGE
  // ==========================================================

  function handleJudgeChange(
    value: string
  ) {

    setSelectedJudge(value);

    // Changing either model invalidates the previous report.
    setBenchmarkResult(null);

    setError("");
    setStatus("");
    setExpandedQuestion(null);
  }


  // ==========================================================
  // RUN BENCHMARK
  // ==========================================================

  async function runBenchmark() {

    // --------------------------------------------------------
    // Validate selections
    // --------------------------------------------------------

    if (!selectedModel) {

      setError(
        "Please select a Prompt Model."
      );

      return;
    }

    if (!selectedJudge) {

      setError(
        "Please select a Judge Model."
      );

      return;
    }

    if (!sessionId) {

      setError(
        "Session is not ready yet. Please try again."
      );

      return;
    }

    // --------------------------------------------------------
    // Reset UI
    // --------------------------------------------------------

    setLoading(true);
    setError("");
    setBenchmarkResult(null);
    setExpandedQuestion(null);

    setStatus(
      `Preparing ${selectedModel}...`
    );

    try {

      setStatus(
        `Running benchmark with ${selectedModel} and judging with ${selectedJudge}...`
      );

      const response = await fetch(
        `${API_BASE_URL}/api/run-benchmark`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            model: selectedModel,
            judge_model: selectedJudge,
            session_id: sessionId,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {

        throw new Error(
          data.detail ||
          `Benchmark failed (${response.status})`
        );
      }

      setBenchmarkResult(data);

      setStatus(
        "Benchmark completed successfully."
      );

    } catch (err) {

      setError(
        err instanceof Error
          ? err.message
          : "Benchmark failed."
      );

      setStatus("");

    } finally {

      setLoading(false);
    }
  }


  // ==========================================================
  // FORMAT SCORE
  // ==========================================================

  function formatScore(
    score: number | null | undefined
  ) {

    if (
      score === null ||
      score === undefined
    ) {
      return "N/A";
    }

    return score.toFixed(2);
  }


  // ==========================================================
  // SCORE PERCENTAGE
  // ==========================================================

  function scorePercentage(
    score: number | null | undefined
  ) {

    if (
      score === null ||
      score === undefined
    ) {
      return 0;
    }

    return Math.min(
      100,
      Math.max(
        0,
        (score / 5) * 100
      )
    );
  }


  // ==========================================================
  // MODEL LISTS
  // ==========================================================

  const installedModels =
    models.filter(
      (model) => model.installed
    );

  const availableModels =
    models.filter(
      (model) => !model.installed
    );


  // ==========================================================
  // REPORT VISIBILITY
  // ==========================================================

  const showReport =
    Boolean(selectedModel) &&
    Boolean(selectedJudge) &&
    Boolean(benchmarkResult) &&
    !benchmarkResult?.summary?.model
      ? false
      : Boolean(
          selectedModel &&
          selectedJudge &&
          benchmarkResult
        );


  // ==========================================================
  // RENDER
  // ==========================================================

  return (

    <main className="min-h-screen bg-slate-950 text-white">

      {/* =====================================================
          HEADER
      ====================================================== */}

      <header className="border-b border-slate-800">

        <div className="mx-auto max-w-7xl px-6 py-6">

          <div className="flex items-center justify-between">

            <div>

              <h1 className="text-2xl font-bold">
                LLM-EvalBench
              </h1>

              <p className="mt-1 text-sm text-slate-400">
                LLM Evaluation & Benchmarking Platform
              </p>

            </div>

            <div className="rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-xs text-slate-300">
              Ollama
            </div>

          </div>

        </div>

      </header>


      {/* =====================================================
          MAIN CONTENT
      ====================================================== */}

      <div className="mx-auto max-w-7xl px-6 py-8">

        {/* ===================================================
            ERROR
        ==================================================== */}

        {error && (

          <div className="mb-6 rounded-lg border border-red-800 bg-red-950/40 p-4">

            <p className="text-sm font-medium text-red-300">
              {error}
            </p>

          </div>

        )}


        {/* ===================================================
            STATUS
        ==================================================== */}

        {status && !error && (

          <div className="mb-6 rounded-lg border border-blue-800 bg-blue-950/40 p-4">

            <p className="text-sm text-blue-300">
              {status}
            </p>

          </div>

        )}


        {/* ===================================================
            MODEL SELECTION
        ==================================================== */}

        <section className="rounded-xl border border-slate-800 bg-slate-900 p-6">

          <div className="mb-6">

            <h2 className="text-lg font-semibold">
              Benchmark Configuration
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Select the model being evaluated and the
              Ollama model that will judge its responses.
            </p>

          </div>


          <div className="grid gap-6 md:grid-cols-2">

            {/* =================================================
                PROMPT MODEL
            ================================================== */}

            <div>

              <label
                htmlFor="prompt-model"
                className="mb-2 block text-sm font-medium text-slate-300"
              >
                Prompt Model
              </label>

              <select
                id="prompt-model"
                value={selectedModel}
                onChange={(event) =>
                  handleModelChange(
                    event.target.value
                  )
                }
                disabled={
                  loadingModels ||
                  loading
                }
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none transition focus:border-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
              >

                <option value="">
                  {loadingModels
                    ? "Loading Ollama models..."
                    : "Select a prompt model"}
                </option>


                {/* Installed */}

                {installedModels.length > 0 && (

                  <optgroup label="Installed">

                    {installedModels.map(
                      (model) => (

                        <option
                          key={`prompt-${model.id}`}
                          value={model.id}
                        >
                          {model.name} — Installed
                        </option>

                      )
                    )}

                  </optgroup>

                )}


                {/* Available */}

                {availableModels.length > 0 && (

                  <optgroup label="Available from Ollama">

                    {availableModels.map(
                      (model) => (

                        <option
                          key={`prompt-${model.id}`}
                          value={model.id}
                        >
                          {model.name} — Download when selected
                        </option>

                      )
                    )}

                  </optgroup>

                )}

              </select>

              <p className="mt-2 text-xs text-slate-500">
                This model generates the answers that will
                be evaluated.
              </p>

            </div>


            {/* =================================================
                JUDGE MODEL
            ================================================== */}

            <div>

              <label
                htmlFor="judge-model"
                className="mb-2 block text-sm font-medium text-slate-300"
              >
                Judge Model
              </label>

              <select
                id="judge-model"
                value={selectedJudge}
                onChange={(event) =>
                  handleJudgeChange(
                    event.target.value
                  )
                }
                disabled={
                  loadingModels ||
                  loading
                }
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none transition focus:border-purple-500 disabled:cursor-not-allowed disabled:opacity-50"
              >

                <option value="">
                  {loadingModels
                    ? "Loading Ollama models..."
                    : "Select a judge model"}
                </option>


                {/* Installed */}

                {installedModels.length > 0 && (

                  <optgroup label="Installed">

                    {installedModels.map(
                      (model) => (

                        <option
                          key={`judge-${model.id}`}
                          value={model.id}
                        >
                          {model.name} — Installed
                        </option>

                      )
                    )}

                  </optgroup>

                )}


                {/* Available */}

                {availableModels.length > 0 && (

                  <optgroup label="Available from Ollama">

                    {availableModels.map(
                      (model) => (

                        <option
                          key={`judge-${model.id}`}
                          value={model.id}
                        >
                          {model.name} — Download when selected
                        </option>

                      )
                    )}

                  </optgroup>

                )}

              </select>

              <p className="mt-2 text-xs text-slate-500">
                This model evaluates the generated responses
                against the benchmark criteria.
              </p>

            </div>

          </div>


          {/* =================================================
              RUN BUTTON
          ================================================== */}

          <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center">

            <button
              type="button"
              onClick={runBenchmark}
              disabled={
                loading ||
                loadingModels ||
                !selectedModel ||
                !selectedJudge ||
                !sessionId
              }
              className="rounded-lg bg-blue-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
            >

              {loading
                ? "Running Benchmark..."
                : "Run Full Benchmark"}

            </button>


            {!selectedModel && (

              <span className="text-xs text-slate-500">
                Select a Prompt Model
              </span>

            )}

            {selectedModel &&
              !selectedJudge && (

                <span className="text-xs text-slate-500">
                  Select a Judge Model
                </span>

              )}

          </div>

        </section>


        {/* ===================================================
            REPORT
        ==================================================== */}

        {showReport &&
          benchmarkResult && (

            <section className="mt-8">

              {/* ===============================================
                  REPORT HEADER
              ================================================ */}

              <div className="mb-6">

                <h2 className="text-xl font-bold">
                  Benchmark Report
                </h2>

                <p className="mt-1 text-sm text-slate-400">

                  {benchmarkResult.summary.model}

                  {" "}
                  evaluated by

                  {" "}

                  {benchmarkResult.summary.judge_model}

                </p>

              </div>


              {/* ===============================================
                  SUMMARY CARDS
              ================================================ */}

              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

                {/* Overall */}

                <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">

                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    Overall Score
                  </p>

                  <p className="mt-2 text-3xl font-bold text-white">
                    {formatScore(
                      benchmarkResult.summary.overall_score
                    )}
                    <span className="ml-1 text-sm font-normal text-slate-500">
                      / 5
                    </span>
                  </p>

                  {benchmarkResult.summary.overall_score !== null && (

                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">

                      <div
                        className="h-full rounded-full bg-blue-500"
                        style={{
                          width: `${scorePercentage(
                            benchmarkResult.summary.overall_score
                          )}%`,
                        }}
                      />

                    </div>

                  )}

                </div>


                {/* Questions */}

                <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">

                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    Questions
                  </p>

                  <p className="mt-2 text-3xl font-bold">
                    {benchmarkResult.summary.total_questions}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Total benchmark questions
                  </p>

                </div>


                {/* Successful */}

                <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">

                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    Successful
                  </p>

                  <p className="mt-2 text-3xl font-bold text-green-400">
                    {benchmarkResult.summary.successful_questions}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Questions evaluated successfully
                  </p>

                </div>


                {/* Failed */}

                <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">

                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    Failed
                  </p>

                  <p className="mt-2 text-3xl font-bold text-red-400">
                    {benchmarkResult.summary.failed_questions}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Questions with errors
                  </p>

                </div>

              </div>


              {/* ===============================================
                  QUESTION RESULTS
              ================================================ */}

              <div className="mt-8">

                <div className="mb-4">

                  <h3 className="text-lg font-semibold">
                    Question Results
                  </h3>

                  <p className="mt-1 text-sm text-slate-500">
                    Click a question to inspect the response,
                    reference answer, and judge rationale.
                  </p>

                </div>


                <div className="space-y-3">

                  {benchmarkResult.results.map(
                    (result, index) => {

                      const isExpanded =
                        expandedQuestion ===
                        result.question_id;

                      return (

                        <div
                          key={result.question_id}
                          className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900"
                        >

                          {/* =================================
                              QUESTION HEADER
                          ================================== */}

                          <button
                            type="button"
                            onClick={() =>
                              setExpandedQuestion(
                                isExpanded
                                  ? null
                                  : result.question_id
                              )
                            }
                            className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left transition hover:bg-slate-800/60"
                          >

                            <div className="flex min-w-0 items-center gap-4">

                              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-800 text-xs font-semibold text-slate-300">
                                {index + 1}
                              </span>

                              <div className="min-w-0">

                                <div className="flex flex-wrap items-center gap-2">

                                  <span className="text-sm font-semibold text-white">
                                    {result.question_id}
                                  </span>

                                  <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] uppercase tracking-wide text-slate-400">
                                    {result.category}
                                  </span>

                                  <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] uppercase tracking-wide text-slate-400">
                                    {result.difficulty}
                                  </span>

                                </div>

                                <p className="mt-1 truncate text-sm text-slate-400">
                                  {result.question}
                                </p>

                              </div>

                            </div>


                            <div className="flex shrink-0 items-center gap-3">

                              {result.error ? (

                                <span className="rounded-full bg-red-950 px-3 py-1 text-xs font-medium text-red-400">
                                  Error
                                </span>

                              ) : (

                                <span className="rounded-full bg-green-950 px-3 py-1 text-xs font-medium text-green-400">

                                  {formatScore(
                                    result.evaluation?.overall_score
                                  )}
                                  /5

                                </span>

                              )}

                              <span className="text-slate-500">

                                {isExpanded
                                  ? "−"
                                  : "+"}

                              </span>

                            </div>

                          </button>


                          {/* =================================
                              EXPANDED CONTENT
                          ================================== */}

                          {isExpanded && (

                            <div className="border-t border-slate-800 px-5 py-6">

                              {/* Error */}

                              {result.error && (

                                <div className="mb-6 rounded-lg border border-red-800 bg-red-950/30 p-4">

                                  <p className="text-xs uppercase tracking-wide text-red-500">
                                    Error
                                  </p>

                                  <p className="mt-2 text-sm text-red-300">
                                    {result.error}
                                  </p>

                                </div>

                              )}


                              {/* Question */}

                              <div className="mb-6">

                                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                  Question
                                </p>

                                <p className="text-sm leading-6 text-slate-200">
                                  {result.question}
                                </p>

                              </div>


                              {/* Reference Answer */}

                              <div className="mb-6">

                                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                  Reference Answer
                                </p>

                                <div className="rounded-lg bg-slate-950 p-4">

                                  <p className="whitespace-pre-wrap text-sm leading-6 text-slate-300">
                                    {result.reference_answer}
                                  </p>

                                </div>

                              </div>


                              {/* Model Response */}

                              <div className="mb-6">

                                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                  Model Response
                                </p>

                                <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">

                                  <p className="whitespace-pre-wrap text-sm leading-6 text-slate-300">
                                    {result.model_response ||
                                      "No response generated."}
                                  </p>

                                </div>

                              </div>


                              {/* Evaluation */}

                              {result.evaluation && (

                                <div>

                                  <div className="mb-4 flex items-center justify-between">

                                    <div>

                                      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                                        Judge Evaluation
                                      </p>

                                      <p className="mt-1 text-xs text-slate-600">
                                        Judge:{" "}
                                        {result.evaluation.judge_model ||
                                          benchmarkResult.summary.judge_model}
                                      </p>

                                    </div>

                                    <div className="text-right">

                                      <p className="text-2xl font-bold text-white">
                                        {formatScore(
                                          result.evaluation.overall_score
                                        )}
                                        <span className="ml-1 text-xs font-normal text-slate-500">
                                          / 5
                                        </span>
                                      </p>

                                    </div>

                                  </div>


                                  {/* Criterion Scores */}

                                  <div className="grid gap-3 md:grid-cols-2">

                                    {Object.entries(
                                      result.evaluation.criteria_scores
                                    ).map(
                                      ([
                                        criterion,
                                        criterionResult,
                                      ]) => (

                                        <div
                                          key={criterion}
                                          className="rounded-lg border border-slate-800 bg-slate-950 p-4"
                                        >

                                          <div className="flex items-center justify-between">

                                            <span className="text-sm font-medium text-slate-300">
                                              {criterion
                                                .replace(
                                                  /_/g,
                                                  " "
                                                )}
                                            </span>

                                            <span className="text-sm font-bold text-white">
                                              {criterionResult.score}
                                              /5
                                            </span>

                                          </div>


                                          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-800">

                                            <div
                                              className="h-full rounded-full bg-purple-500"
                                              style={{
                                                width: `${
                                                  (criterionResult.score /
                                                    5) *
                                                  100
                                                }%`,
                                              }}
                                            />

                                          </div>


                                          <p className="mt-3 text-xs leading-5 text-slate-500">
                                            {
                                              criterionResult.rationale
                                            }
                                          </p>

                                        </div>

                                      )
                                    )}

                                  </div>


                                  {/* Overall Rationale */}

                                  {result.evaluation.overall_rationale && (

                                    <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950 p-4">

                                      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                                        Overall Rationale
                                      </p>

                                      <p className="mt-2 text-sm leading-6 text-slate-400">
                                        {
                                          result.evaluation
                                            .overall_rationale
                                        }
                                      </p>

                                    </div>

                                  )}

                                </div>

                              )}

                            </div>

                          )}

                        </div>

                      );
                    }
                  )}

                </div>

              </div>

            </section>

          )}

      </div>

    </main>
  );
}