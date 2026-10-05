"use client";

import {
  useEffect,
  useState,
} from "react";

const API_URL = "/.netlify/functions";

type Model = {
  id: string;
  name: string;
  provider: string;
  capabilities?: string[];
  benchmark_compatible?: boolean;
  input_modalities?: string[];
  output_modalities?: string[];
  context_length?: number;
  max_completion_tokens?: number;
  pricing?: {
    prompt?: string | null;
    completion?: string | null;
  };
};

type BenchmarkResult = {
  provider?: string;
  source?: string;
  model?: string;
  judge_model?: string;

  status?: "completed" | "partial";
  overall_score?: number | null;
  total_questions?: number;
  successful?: number;
  failed?: number;

  error_summary?: Record<string, number>;

  by_category?: Record<
    string,
    number
  >;

  by_difficulty?: Record<
    string,
    number
  >;

  by_criterion?: Record<
    string,
    number
  >;

  lowest_scoring?: {
    id: string;
    category: string;
    difficulty: string;
    score: number;
  }[];

  error?: string;
};


export default function Home() {

  const [
    apiOnline,
    setApiOnline,
  ] = useState(false);

  const [
    apiKey,
    setApiKey,
  ] = useState("");

  const [
    apiVerified,
    setApiVerified,
  ] = useState(false);

  const [
    verifyingKey,
    setVerifyingKey,
  ] = useState(false);

  const [
    verificationError,
    setVerificationError,
  ] = useState("");

  const [
    selectedModel,
    setSelectedModel,
  ] = useState("");

  const [
    selectedJudge,
    setSelectedJudge,
  ] = useState("");

  const [
    models,
    setModels,
  ] = useState<Model[]>([]);

  const [
    modelsLoading,
    setModelsLoading,
  ] = useState(false);

  const [
    benchmarkResult,
    setBenchmarkResult,
  ] = useState<
    BenchmarkResult | null
  >(null);

  const [
    benchmarking,
    setBenchmarking,
  ] = useState(false);


  // ==========================================================
  // HEALTH CHECK
  // ==========================================================

  useEffect(() => {

    async function checkApi() {

      try {

        const response =
          await fetch(
            `${API_URL}/health`
          );

        setApiOnline(
          response.ok
        );

      } catch (error) {

        console.error(
          "API health error:",
          error
        );

        setApiOnline(false);
      }
    }

    checkApi();

  }, []);


  // ==========================================================
  // MODEL NAME
  // ==========================================================

  function formatName(
    name: string
  ) {

    return name
      .replaceAll(
        "_",
        " "
      )
      .replace(
        /\b\w/g,
        (char) =>
          char.toUpperCase()
      );
  }


  function displayModelName(
    modelId?: string
  ) {

    if (!modelId) {
      return "--";
    }

    const found =
      models.find(
        (model) =>
          model.id === modelId
      );

    return (
      found?.name ||
      formatName(modelId)
    );
  }


  // ==========================================================
  // VERIFY API KEY
  // ==========================================================

  async function verifyApiKey() {

    if (!apiKey.trim()) {

      setVerificationError(
        "Enter your OpenRouter API key."
      );

      return;
    }

    setVerifyingKey(true);

    setVerificationError("");

    setApiVerified(false);

    setModels([]);

    setSelectedModel("");

    setSelectedJudge("");

    setBenchmarkResult(null);


    try {

      const response =
        await fetch(
          `${API_URL}/verify`,
          {
            method: "POST",

            headers: {
              "X-API-Key":
                apiKey.trim(),
            },
          }
        );


      if (!response.ok) {

        let message =
          "OpenRouter API key verification failed.";

        try {

          const error =
            await response.json();

          message =
            typeof error.detail ===
            "object"
              ? error.detail?.error ||
                message
              : error.detail ||
                message;

        } catch {
          // Keep default.
        }

        throw new Error(
          message
        );
      }


      const result =
        await response.json();


      if (!result.valid) {
        throw new Error(
          result.message ||
          "OpenRouter API key could not be verified."
        );
      }

      setApiVerified(true);

      await loadModels(
        apiKey.trim()
      );

    } catch (error) {

      console.error(
        "API key verification error:",
        error
      );

      setApiVerified(false);

      setVerificationError(
        error instanceof Error
          ? error.message
          : "API key verification failed."
      );

    } finally {

      setVerifyingKey(false);
    }
  }


  // ==========================================================
  // LOAD MODELS
  // ==========================================================

  async function loadModels(
    key: string
  ) {

    setModelsLoading(true);

    try {

      const response =
        await fetch(
          `${API_URL}/models`,
          {
            method: "GET",

            headers: {
              "X-API-Key": key,
            },
          }
        );


      if (!response.ok) {

        let message =
          "Failed to load OpenRouter models.";

        try {

          const error =
            await response.json();

          message =
            typeof error.detail ===
            "object"
              ? error.detail?.error ||
                message
              : error.detail ||
                message;

        } catch {
          // Keep default.
        }

        throw new Error(
          message
        );
      }


      const data =
        await response.json();


      setModels(
        data.models || []
      );

    } catch (error) {

      console.error(
        "Model loading error:",
        error
      );

      setApiVerified(false);

      setVerificationError(
        error instanceof Error
          ? error.message
          : "Failed to load models."
      );

    } finally {

      setModelsLoading(false);
    }
  }


  // ==========================================================
  // API KEY CHANGE
  // ==========================================================

  function handleApiKeyChange(
    value: string
  ) {

    setApiKey(value);

    // Changing the key invalidates
    // the previous verification.

    setApiVerified(false);

    setModels([]);

    setSelectedModel("");

    setSelectedJudge("");

    setBenchmarkResult(null);

    setVerificationError("");
  }


  // ==========================================================
  // MODEL CHANGE
  // ==========================================================

  function handleModelChange(
    value: string
  ) {

    setSelectedModel(value);

    setBenchmarkResult(null);
  }


  function handleJudgeChange(
    value: string
  ) {

    setSelectedJudge(value);

    setBenchmarkResult(null);
  }


  // ==========================================================
  // RUN BENCHMARK
  // ==========================================================

  async function runBenchmark() {

    if (!apiVerified) {

      return;
    }

    if (
      !selectedModel ||
      !selectedJudge
    ) {

      return;
    }

    setBenchmarking(true);

    setBenchmarkResult(null);


    try {

      const response =
        await fetch(
          `${API_URL}/run-benchmark`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",

              "X-API-Key":
                apiKey.trim(),
            },

            body: JSON.stringify({
              model:
                selectedModel,

              judge_model:
                selectedJudge,
            }),
          }
        );


      if (!response.ok) {

        let message =
          "Benchmark failed.";

        try {

          const error =
            await response.json();

          message =
            typeof error.detail ===
            "object"
              ? error.detail?.error ||
                message
              : error.detail ||
                message;

        } catch {

          message =
            `Benchmark failed with status ${response.status}.`;
        }

        throw new Error(
          message
        );
      }


      const result:
        BenchmarkResult =
        await response.json();


      setBenchmarkResult(
        result
      );

    } catch (error) {

      console.error(
        "Benchmark error:",
        error
      );

      setBenchmarkResult({
        error:
          error instanceof Error
            ? error.message
            : "Benchmark failed.",
      });

    } finally {

      setBenchmarking(false);
    }
  }


  // ==========================================================
  // REPORT VISIBILITY
  // ==========================================================

  const showReport =
    apiVerified &&
    Boolean(selectedModel) &&
    Boolean(selectedJudge) &&
    Boolean(benchmarkResult) &&
    !benchmarkResult?.error;


  // ==========================================================
  // UI
  // ==========================================================

  return (

    <main className="min-h-screen bg-slate-950 text-white">

      {/* ================================================== */}
      {/* HEADER                                             */}
      {/* ================================================== */}

      <header className="border-b border-slate-800 bg-slate-950">

        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">

          <div>

            <h1 className="text-2xl font-bold tracking-tight">
              LLM-EvalBench
            </h1>

            <p className="mt-1 text-sm text-slate-400">
              LLM Evaluation & Benchmarking Platform
            </p>

            <div className="mt-2">

              <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs text-slate-400">
                Provider: OpenRouter
              </span>

            </div>

          </div>


          <div className="flex items-center gap-2 rounded-full border border-slate-700 px-4 py-2 text-sm">

            <span
              className={`h-2.5 w-2.5 rounded-full ${
                apiOnline
                  ? "bg-green-400"
                  : "bg-red-400"
              }`}
            />

            {apiOnline
              ? "API Online"
              : "API Offline"}

          </div>

        </div>

      </header>


      <div className="mx-auto max-w-7xl px-6 py-8">


        {/* ================================================= */}
        {/* BACKEND ERROR                                    */}
        {/* ================================================= */}

        {!apiOnline && (

          <div className="mb-8 rounded-xl border border-red-900 bg-red-950/30 p-6">

            <p className="text-red-300">
              Unable to connect to the Netlify backend.
            </p>

            <p className="mt-2 text-sm text-red-400">
              Make sure the Netlify Functions are running.
            </p>

          </div>

        )}


        {/* ================================================= */}
        {/* MAIN PANEL                                       */}
        {/* ================================================= */}

        <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">


          <div>

            <h2 className="text-xl font-semibold text-white">
              Run Full Benchmark
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Evaluate a selected OpenRouter model across all
              20 benchmark questions.
            </p>

          </div>


          {/* ================================================= */}
          {/* API KEY                                          */}
          {/* ================================================= */}

          <div className="mt-6 rounded-xl border border-slate-800 bg-slate-950 p-5">

            <label
              htmlFor="openrouter-key"
              className="block text-sm font-medium text-slate-300"
            >
              OpenRouter API Key
            </label>


            <p className="mt-1 text-xs text-slate-500">
              Your API key is kept in this browser session and
              is sent to the backend only when making OpenRouter
              requests.
            </p>


            {/* WARNING */}

            <div className="mt-4 rounded-lg border border-amber-900 bg-amber-950/30 p-4">

              <p className="text-sm font-semibold text-amber-300">
                ⚠ Model catalog includes free and premium models.
              </p>

              <p className="mt-1 text-xs leading-5 text-amber-400">
                Make sure your OpenRouter account has sufficient
                credits before running a benchmark. Benchmarking
                can make multiple model requests.
              </p>

            </div>


            <div className="mt-4 flex flex-col gap-3 sm:flex-row">

              <input
                id="openrouter-key"
                type="password"
                value={apiKey}
                onChange={(event) =>
                  handleApiKeyChange(
                    event.target.value
                  )
                }
                placeholder="sk-or-v1-..."
                className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-white outline-none placeholder:text-slate-600 focus:border-blue-500"
              />


              <button
                onClick={verifyApiKey}
                disabled={
                  verifyingKey ||
                  !apiKey.trim() ||
                  !apiOnline
                }
                className="rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
              >

                {verifyingKey
                  ? "Verifying..."
                  : apiVerified
                    ? "✓ API Verified"
                    : "Verify API Key"}

              </button>

            </div>


            {/* VERIFICATION SUCCESS */}

            {apiVerified && (

              <div className="mt-4 rounded-lg border border-green-900 bg-green-950/30 p-4">

                <p className="text-sm font-semibold text-green-300">
                  ✓ OpenRouter API key verified
                </p>

                <p className="mt-1 text-xs text-green-400">
                  The OpenRouter model catalog is now available.
                </p>

              </div>

            )}


            {/* VERIFICATION ERROR */}

            {verificationError && (

              <div className="mt-4 rounded-lg border border-red-900 bg-red-950/30 p-4">

                <p className="text-sm font-semibold text-red-300">
                  API Key Verification Failed
                </p>

                <p className="mt-1 text-xs text-red-400">
                  {verificationError}
                </p>

              </div>

            )}

          </div>


          {/* ================================================= */}
          {/* MODEL SELECTORS                                  */}
          {/* IMPORTANT: ONLY RENDER AFTER VERIFICATION       */}
          {/* ================================================= */}

          {apiVerified && (

            <>

              <div className="mt-6 grid gap-5 md:grid-cols-2">


                {/* PROMPT MODEL */}

                <div>

                  <label className="mb-2 block text-sm text-slate-300">
                    Prompt Model
                  </label>

                  <select
                    value={selectedModel}
                    onChange={(event) =>
                      handleModelChange(
                        event.target.value
                      )
                    }
                    disabled={
                      modelsLoading
                    }
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                  >

                    <option value="">
                      {modelsLoading
                        ? "Loading models..."
                        : "Select model"}
                    </option>


                    {models
                      .filter(
                        (
                          model
                        ) =>
                          model.benchmark_compatible
                      )
                      .map(
                        (
                          model
                        ) => (

                          <option
                            key={
                              model.id
                            }
                            value={
                              model.id
                            }
                          >
                            {model.name}
                          </option>

                        )
                      )}

                  </select>

                  <p className="mt-2 text-xs text-slate-500">
                    {models.length} models available in the
                    OpenRouter catalog.
                  </p>

                </div>


                {/* JUDGE MODEL */}

                <div>

                  <label className="mb-2 block text-sm text-slate-300">
                    Judge Model
                  </label>

                  <select
                    value={selectedJudge}
                    onChange={(event) =>
                      handleJudgeChange(
                        event.target.value
                      )
                    }
                    disabled={
                      modelsLoading
                    }
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                  >

                    <option value="">
                      {modelsLoading
                        ? "Loading models..."
                        : "Select judge model"}
                    </option>


                    {models
                      .filter(
                        (
                          model
                        ) =>
                          model.benchmark_compatible
                      )
                      .map(
                        (
                          model
                        ) => (

                          <option
                            key={
                              model.id
                            }
                            value={
                              model.id
                            }
                          >
                            {model.name}
                          </option>

                        )
                      )}

                  </select>

                  <p className="mt-2 text-xs text-slate-500">
                    The judge evaluates the candidate responses
                    against the benchmark criteria.
                  </p>

                </div>

              </div>


              {/* SELECTED MODELS */}

              {(selectedModel ||
                selectedJudge) && (

                <div className="mt-5 grid gap-3 rounded-lg border border-slate-800 bg-slate-950 p-4 md:grid-cols-2">

                  <div>

                    <p className="text-xs text-slate-500">
                      Prompt Model
                    </p>

                    <p className="mt-1 text-sm text-slate-200">
                      {selectedModel
                        ? displayModelName(
                            selectedModel
                          )
                        : "Not selected"}
                    </p>

                  </div>


                  <div>

                    <p className="text-xs text-slate-500">
                      Judge Model
                    </p>

                    <p className="mt-1 text-sm text-slate-200">
                      {selectedJudge
                        ? displayModelName(
                            selectedJudge
                          )
                        : "Not selected"}
                    </p>

                  </div>

                </div>

              )}


              {/* RUN BUTTON */}

              <button
                onClick={
                  runBenchmark
                }
                disabled={
                  benchmarking ||
                  !apiVerified ||
                  !selectedModel ||
                  !selectedJudge ||
                  !apiOnline
                }
                className="mt-6 rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
              >

                {benchmarking
                  ? "Running 20-Question Benchmark..."
                  : "Run Full Benchmark"}

              </button>


              {/* SELECTION MESSAGE */}

              {!selectedModel ||
              !selectedJudge ? (

                <div className="mt-6 rounded-lg border border-slate-800 bg-slate-950 p-5">

                  <h3 className="font-semibold text-white">
                    Select Models to Begin
                  </h3>

                  <p className="mt-2 text-sm text-slate-400">
                    Select both a Prompt Model and a Judge Model
                    to run the benchmark.
                  </p>

                </div>

              ) : null}


              {/* RUNNING */}

              {benchmarking && (

                <div className="mt-6 rounded-lg border border-blue-900 bg-blue-950/30 p-5">

                  <p className="font-medium text-blue-300">
                    Benchmark is running...
                  </p>

                  <p className="mt-2 text-sm text-blue-400">
                    The selected model is being evaluated across
                    all 20 benchmark questions. OpenRouter usage
                    may incur charges depending on the models selected.
                  </p>

                </div>

              )}


              {/* ERROR */}

              {benchmarkResult?.error && (

                <div className="mt-6 rounded-lg border border-red-900 bg-red-950/30 p-5">

                  <p className="font-semibold text-red-300">
                    Benchmark Error
                  </p>

                  <p className="mt-2 text-sm text-red-400">
                    {benchmarkResult.error}
                  </p>

                </div>

              )}


              {/* ================================================= */}
              {/* REPORT                                            */}
              {/* ================================================= */}

              {showReport && benchmarkResult && (
                <div className="mt-8 border-t border-slate-800 pt-8">

                  {/* REPORT HEADER */}
                  <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">

                    <div>
                      <h3 className="text-xl font-semibold text-white">
                        Benchmark Report
                      </h3>

                      <p className="mt-1 text-sm text-slate-400">
                        {displayModelName(benchmarkResult.model)}
                        {" → "}
                        {displayModelName(benchmarkResult.judge_model)}
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="rounded-full border border-slate-700 px-3 py-1 text-sm text-slate-300">
                        OpenRouter
                      </span>

                      <span
                        className={`rounded-full border px-3 py-1 text-sm ${
                          benchmarkResult.status === "completed"
                            ? "border-green-800 bg-green-950/30 text-green-300"
                            : "border-amber-800 bg-amber-950/30 text-amber-300"
                        }`}
                      >
                        {benchmarkResult.status === "completed"
                          ? "Completed"
                          : "Partial"}
                      </span>
                    </div>

                  </div>


                  {/* PARTIAL RUN WARNING */}
                  {benchmarkResult.status === "partial" && (
                    <div className="mt-6 rounded-xl border border-amber-900 bg-amber-950/30 p-5">

                      <div className="flex items-start gap-3">

                        <div className="text-xl">
                          ⚠
                        </div>

                        <div>
                          <h4 className="font-semibold text-amber-300">
                            Benchmark partially completed
                          </h4>

                          <p className="mt-2 text-sm leading-6 text-amber-400">
                            {benchmarkResult.successful || 0} of{" "}
                            {benchmarkResult.total_questions || 0} questions
                            were evaluated successfully.
                            {" "}
                            {benchmarkResult.failed || 0} questions failed.
                          </p>

                          <p className="mt-2 text-sm leading-6 text-amber-400">
                            The overall benchmark score is unavailable until
                            all questions have been evaluated successfully.
                          </p>

                          {benchmarkResult.error_summary &&
                            Object.keys(benchmarkResult.error_summary).length > 0 && (
                              <div className="mt-4 rounded-lg border border-amber-900 bg-slate-950 p-4">

                                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                                  Failure Summary
                                </p>

                                <div className="mt-3 space-y-2">

                                  {Object.entries(
                                    benchmarkResult.error_summary
                                  ).map(([errorType, count]) => (
                                    <div
                                      key={errorType}
                                      className="flex items-center justify-between text-sm"
                                    >
                                      <span className="text-slate-300">
                                        {formatName(errorType)}
                                      </span>

                                      <span className="font-semibold text-amber-300">
                                        {count}
                                      </span>
                                    </div>
                                  ))}

                                </div>

                              </div>
                            )}

                        </div>

                      </div>

                    </div>
                  )}


                  {/* METRICS */}
                  <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

                    <MetricCard
                      title="Overall Score"
                      value={
                        benchmarkResult.status === "completed" &&
                        benchmarkResult.overall_score != null
                          ? `${(
                              benchmarkResult.overall_score * 100
                            ).toFixed(1)}%`
                          : "Unavailable"
                      }
                    />

                    <MetricCard
                      title="Questions"
                      value={`${benchmarkResult.successful || 0}/${
                        benchmarkResult.total_questions || 0
                      }`}
                    />

                    <MetricCard
                      title="Failed"
                      value={String(benchmarkResult.failed || 0)}
                    />

                    <MetricCard
                      title="Judge Model"
                      value={displayModelName(
                        benchmarkResult.judge_model
                      )}
                    />

                  </div>


                  {/* ONLY SHOW ANALYTICS FOR A COMPLETE RUN */}
                  {benchmarkResult.status === "completed" && (
                    <>

                      {/* CATEGORY */}
                      <div className="mt-8">

                        <h4 className="text-lg font-semibold text-white">
                          Performance by Category
                        </h4>

                        <p className="mt-1 text-sm text-slate-400">
                          Average evaluation score across benchmark categories.
                        </p>

                        <div className="mt-6 space-y-5">

                          {Object.entries(
                            benchmarkResult.by_category || {}
                          ).map(([category, score]) => (
                            <ScoreBar
                              key={category}
                              label={formatName(category)}
                              score={Number(score)}
                            />
                          ))}

                        </div>

                      </div>


                      {/* DIFFICULTY */}
                      <div className="mt-8">

                        <h4 className="text-lg font-semibold text-white">
                          Performance by Difficulty
                        </h4>

                        <div className="mt-5 grid gap-4 md:grid-cols-3">

                          {Object.entries(
                            benchmarkResult.by_difficulty || {}
                          ).map(([difficulty, score]) => (
                            <div
                              key={difficulty}
                              className="rounded-lg border border-slate-800 bg-slate-950 p-5"
                            >

                              <div className="text-sm capitalize text-slate-400">
                                {difficulty}
                              </div>

                              <div className="mt-2 text-2xl font-semibold text-white">
                                {(Number(score) * 100).toFixed(1)}%
                              </div>

                            </div>
                          ))}

                        </div>

                      </div>


                      {/* CRITERIA */}
                      <div className="mt-8">

                        <h4 className="text-lg font-semibold text-white">
                          Performance by Criterion
                        </h4>

                        <div className="mt-5 grid gap-x-8 md:grid-cols-2">

                          {Object.entries(
                            benchmarkResult.by_criterion || {}
                          ).map(([criterion, score]) => (
                            <div
                              key={criterion}
                              className="flex items-center justify-between border-b border-slate-800 py-3"
                            >

                              <span className="text-slate-300">
                                {formatName(criterion)}
                              </span>

                              <span className="font-semibold text-white">
                                {(Number(score) * 100).toFixed(1)}%
                              </span>

                            </div>
                          ))}

                        </div>

                      </div>


                      {/* LOWEST SCORING */}
                      <div className="mt-8">

                        <h4 className="text-lg font-semibold text-white">
                          Lowest-Scoring Questions
                        </h4>

                        <div className="mt-5 overflow-hidden rounded-lg border border-slate-800">

                          {benchmarkResult.lowest_scoring &&
                          benchmarkResult.lowest_scoring.length > 0 ? (

                            benchmarkResult.lowest_scoring.map((item) => (
                              <div
                                key={item.id}
                                className="flex items-center justify-between border-b border-slate-800 px-4 py-4 last:border-b-0"
                              >

                                <div>

                                  <div className="font-medium text-white">
                                    {item.id}
                                  </div>

                                  <div className="mt-1 text-sm text-slate-400">
                                    {formatName(item.category)}
                                    {" · "}
                                    {formatName(item.difficulty)}
                                  </div>

                                </div>

                                <div className="font-semibold text-white">
                                  {(item.score * 100).toFixed(1)}%
                                </div>

                              </div>
                            ))

                          ) : (

                            <div className="p-5 text-sm text-slate-400">
                              No low-scoring questions available.
                            </div>

                          )}

                        </div>

                      </div>

                    </>
                  )}

                </div>
              )}

            </>

          )}


          {/* ================================================= */}
          {/* BEFORE VERIFICATION                              */}
          {/* ================================================= */}

          {!apiVerified && (

            <div className="mt-6 rounded-lg border border-slate-800 bg-slate-950 p-5">

              <h3 className="font-semibold text-white">
                Verify OpenRouter API Key to Continue
              </h3>

              <p className="mt-2 text-sm text-slate-400">
                Model selection will appear only after your
                OpenRouter API key has been successfully verified.
              </p>

            </div>

          )}

        </section>

      </div>

    </main>
  );
}


/* ============================================================= */
/* COMPONENTS                                                     */
/* ============================================================= */

function MetricCard({
  title,
  value,
}: {
  title: string;
  value: string;
}) {

  return (

    <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">

      <p className="text-sm text-slate-400">
        {title}
      </p>

      <p className="mt-3 text-2xl font-bold tracking-tight text-white">
        {value}
      </p>

    </div>
  );
}


function ScoreBar({
  label,
  score,
}: {
  label: string;
  score: number;
}) {

  const percentage =
    Math.max(
      0,
      Math.min(
        100,
        score * 100
      )
    );

  return (

    <div>

      <div className="mb-2 flex justify-between text-sm">

        <span className="text-slate-300">
          {label}
        </span>

        <span className="font-semibold text-white">
          {percentage.toFixed(1)}%
        </span>

      </div>


      <div className="h-3 overflow-hidden rounded-full bg-slate-800">

        <div
          className="h-full rounded-full bg-blue-500 transition-all"
          style={{
            width:
              `${percentage}%`,
          }}
        />

      </div>

    </div>
  );
}