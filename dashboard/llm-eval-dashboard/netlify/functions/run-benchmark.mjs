/**
 * LLM-EvalBench
 * Netlify Function: run-benchmark
 *
 * Runs the benchmark through:
 *   benchmark.mjs -> evaluate.mjs
 *
 * The function fails fast on provider-level errors such as:
 *   - 429 rate limits
 *   - 402 insufficient credits
 *   - 401 authentication errors
 *   - timeouts
 *
 * This prevents the local Netlify 30-second Lambda timeout.
 */

const JSON_HEADERS = {
  "Content-Type": "application/json",
  "Cache-Control": "no-store",
};

const REQUEST_TIMEOUT_MS = 20000;;

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: JSON_HEADERS,
  });
}

function classifyError(message = "") {
  const text = String(message).toLowerCase();

  if (
    text.includes("402") ||
    text.includes("insufficient credits") ||
    text.includes("credit") ||
    text.includes("in_flight_budget_exhausted") ||
    text.includes("payment required") ||
    text.includes("quota")
  ) {
    return "insufficient_credits";
  }

  if (
    text.includes("401") ||
    text.includes("unauthorized") ||
    text.includes("invalid api key") ||
    text.includes("invalid, expired") ||
    text.includes("authentication")
  ) {
    return "authentication_error";
  }

  if (
    text.includes("429") ||
    text.includes("rate limit") ||
    text.includes("too many requests") ||
    text.includes("rate_limit")
  ) {
    return "rate_limit";
  }

  if (
    text.includes("timeout") ||
    text.includes("timed out") ||
    text.includes("abort") ||
    text.includes("deadline")
  ) {
    return "timeout";
  }

  return "evaluation_error";
}

async function readError(response) {
  const text = await response.text();

  try {
    const data = JSON.parse(text);

    if (typeof data?.error === "string") {
      return data.error;
    }

    if (typeof data?.message === "string") {
      return data.message;
    }

    if (typeof data?.detail === "string") {
      return data.detail;
    }

    if (typeof data?.detail?.error === "string") {
      return data.detail.error;
    }

    return text || `Request failed with status ${response.status}.`;
  } catch {
    return text || `Request failed with status ${response.status}.`;
  }
}

async function getJson(response) {
  const text = await response.text();

  try {
    return JSON.parse(text);
  } catch {
    throw new Error(
      `Invalid JSON response from Netlify function (HTTP ${response.status}): ${text.slice(
        0,
        500
      )}`
    );
  }
}

async function fetchWithTimeout(
  url,
  options = {},
  timeoutMs = REQUEST_TIMEOUT_MS
) {
  const controller = new AbortController();

  const timer = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

async function getBenchmark(origin) {
  const url = `${origin}/.netlify/functions/benchmark`;

  const response = await fetchWithTimeout(url, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    const message = await readError(response);
    throw new Error(`Benchmark dataset failed: ${message}`);
  }

  const data = await getJson(response);

  if (Array.isArray(data)) {
    return data;
  }

  if (Array.isArray(data.questions)) {
    return data.questions;
  }

  if (Array.isArray(data.benchmark)) {
    return data.benchmark;
  }

  if (Array.isArray(data.data)) {
    return data.data;
  }

  throw new Error(
    "Benchmark function returned an unexpected response. Expected an array of questions."
  );
}

async function evaluateQuestion({
  origin,
  apiKey,
  model,
  judgeModel,
  questionId,
}) {
  const url = `${origin}/.netlify/functions/evaluate`;

  const response = await fetchWithTimeout(
    url,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },

      body: JSON.stringify({
        question_id: questionId,
        model,
        judge_model: judgeModel,
      }),
    },
    REQUEST_TIMEOUT_MS
  );

  if (!response.ok) {
    const message = await readError(response);

    const error = new Error(message);
    error.status = response.status;

    throw error;
  }

  return getJson(response);
}

function aggregate(results, totalQuestions) {
  const successfulResults = results.filter(
    (result) => result.status === "success"
  );

  const failedResults = results.filter(
    (result) => result.status === "failed"
  );

  const successful = successfulResults.length;
  const failed = failedResults.length;

  const complete = successful === totalQuestions;

  let overallScore = null;

  const byCategory = {};
  const byDifficulty = {};
  const byCriterion = {};

  let lowestScoring = [];

  /*
   * Only calculate the final benchmark score when ALL questions
   * have successfully completed.
   */
  if (complete && successfulResults.length > 0) {
    const scores = successfulResults
      .map((result) => Number(result?.evaluation?.overall_score))
      .filter((score) => Number.isFinite(score));

    if (scores.length === successfulResults.length) {
      overallScore = Number(
        (
          scores.reduce((sum, score) => sum + score, 0) /
          scores.length
        ).toFixed(2)
      );
    }

    const categoryScores = {};
    const difficultyScores = {};
    const criterionScores = {};

    for (const result of successfulResults) {
      const evaluation = result.evaluation || {};

      const score = Number(evaluation.overall_score);

      if (Number.isFinite(score)) {
        const category = result.category || "unknown";
        const difficulty = result.difficulty || "unknown";

        if (!categoryScores[category]) {
          categoryScores[category] = [];
        }

        if (!difficultyScores[difficulty]) {
          difficultyScores[difficulty] = [];
        }

        categoryScores[category].push(score);
        difficultyScores[difficulty].push(score);
      }

      const criteriaScores = evaluation.criteria_scores || {};

      for (const [criterion, criterionData] of Object.entries(
        criteriaScores
      )) {
        const criterionScore =
          typeof criterionData === "object"
            ? Number(criterionData?.score)
            : Number(criterionData);

        if (!Number.isFinite(criterionScore)) {
          continue;
        }

        if (!criterionScores[criterion]) {
          criterionScores[criterion] = [];
        }

        criterionScores[criterion].push(criterionScore);
      }
    }

    for (const [category, scores] of Object.entries(categoryScores)) {
      byCategory[category] = Number(
        (
          scores.reduce((a, b) => a + b, 0) /
          scores.length
        ).toFixed(2)
      );
    }

    for (const [difficulty, scores] of Object.entries(
      difficultyScores
    )) {
      byDifficulty[difficulty] = Number(
        (
          scores.reduce((a, b) => a + b, 0) /
          scores.length
        ).toFixed(2)
      );
    }

    for (const [criterion, scores] of Object.entries(
      criterionScores
    )) {
      byCriterion[criterion] = Number(
        (
          scores.reduce((a, b) => a + b, 0) /
          scores.length
        ).toFixed(2)
      );
    }

    lowestScoring = successfulResults
      .map((result) => ({
        id: result.id,
        category: result.category,
        difficulty: result.difficulty,
        score: Number(result?.evaluation?.overall_score),
      }))
      .filter((item) => Number.isFinite(item.score))
      .sort((a, b) => a.score - b.score)
      .slice(0, 5);
  }

  const errorSummary = {};

  for (const result of failedResults) {
    const errorType = result.error_type || "evaluation_error";

    errorSummary[errorType] =
      (errorSummary[errorType] || 0) + 1;
  }

  return {
    status: complete ? "completed" : "partial",

    total_questions: totalQuestions,

    successful,

    failed,

    overall_score: overallScore,

    by_category: byCategory,

    by_difficulty: byDifficulty,

    by_criterion: byCriterion,

    lowest_scoring: lowestScoring,

    error_summary: errorSummary,
  };
}

export default async function handler(request) {
  /*
   * CORS preflight
   */
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,

      headers: {
        ...JSON_HEADERS,

        "Access-Control-Allow-Origin": "*",

        "Access-Control-Allow-Headers":
          "Content-Type, X-API-Key",

        "Access-Control-Allow-Methods":
          "POST, OPTIONS",
      },
    });
  }

  /*
   * Only POST is allowed.
   */
  if (request.method !== "POST") {
    return json(
      {
        error: "Method not allowed",
      },
      405
    );
  }

  /*
   * Read OpenRouter API key.
   */
  const apiKey =
    request.headers.get("x-api-key") ||
    request.headers.get("X-API-Key") ||
    "";

  if (!apiKey.trim()) {
    return json(
      {
        error: "API key is required",
      },
      401
    );
  }

  /*
   * Parse request body.
   */
  let body;

  try {
    body = await request.json();
  } catch {
    return json(
      {
        error: "Request body must be valid JSON.",
      },
      400
    );
  }

  const model = String(
    body?.model || ""
  ).trim();

  const judgeModel = String(
    body?.judge_model || ""
  ).trim();

  if (!model) {
    return json(
      {
        error: "Model is required.",
      },
      400
    );
  }

  if (!judgeModel) {
    return json(
      {
        error: "Judge model is required.",
      },
      400
    );
  }

  /*
   * Use the current Netlify origin.
   *
   * Works with:
   *   localhost:8888
   *   deployed Netlify site
   */
  const origin = new URL(request.url).origin;

  try {
    console.log(
      `[run-benchmark] Starting benchmark: model=${model}, judge=${judgeModel}`
    );

    /*
     * Load benchmark dataset.
     */
    const benchmark = await getBenchmark(origin);

    if (!benchmark.length) {
      return json(
        {
          error: "Benchmark dataset is empty.",
        },
        400
      );
    }

    console.log(
      `[run-benchmark] Loaded ${benchmark.length} benchmark questions`
    );

    const results = [];

    /*
     * IMPORTANT:
     *
     * We intentionally execute sequentially.
     *
     * If OpenRouter is rate-limiting the request, we stop after the
     * first provider-level failure rather than attempting all 20
     * questions and hitting the Netlify 30-second timeout.
     */
    for (const question of benchmark) {
      const questionId = question?.id;

      if (!questionId) {
        results.push({
          id: null,

          category: question?.category,

          difficulty: question?.difficulty,

          model,

          judge_model: judgeModel,

          provider: "openrouter",

          status: "failed",

          error_type: "evaluation_error",

          error: "Benchmark question has no id.",
        });

        continue;
      }

      console.log(
        `[run-benchmark] Evaluating ${questionId}`
      );

      try {
        const evaluationResult =
          await evaluateQuestion({
            origin,
            apiKey,
            model,
            judgeModel,
            questionId,
          });

        results.push({
          id: questionId,

          category: question?.category,

          difficulty: question?.difficulty,

          model,

          judge_model: judgeModel,

          provider: "openrouter",

          model_response:
            evaluationResult?.model_response ?? null,

          evaluation:
            evaluationResult?.evaluation ?? null,

          status: "success",
        });

      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : String(error);

        const errorType =
          classifyError(message);

        console.error(
          `[run-benchmark] ${questionId} failed: ${errorType}: ${message}`
        );

        results.push({
          id: questionId,

          category: question?.category,

          difficulty: question?.difficulty,

          model,

          judge_model: judgeModel,

          provider: "openrouter",

          status: "failed",

          error_type: errorType,

          error: message,
        });

        /*
         * Stop immediately for provider-level failures.
         */
        if (
          errorType === "rate_limit" ||
          errorType === "insufficient_credits" ||
          errorType === "authentication_error" ||
          errorType === "timeout"
        ) {
          console.log(
            `[run-benchmark] Stopping early because of ${errorType}.`
          );

          break;
        }
      }
    }

    /*
     * Aggregate results.
     *
     * total_questions remains 20 even when we stop early,
     * so the dashboard knows this was only a partial run.
     */
    const summary = aggregate(
      results,
      benchmark.length
    );

    console.log(
      `[run-benchmark] Complete: ${summary.successful}/${summary.total_questions} successful`
    );

    return json({
      source: "runtime",

      provider: "openrouter",

      model,

      judge_model: judgeModel,

      ...summary,

      results,
    });

  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : String(error);

    console.error(
      `[run-benchmark] Fatal error: ${message}`
    );

    /*
     * IMPORTANT:
     * Return 200 instead of 500 so the dashboard can render
     * a structured partial/error result.
     */
    return json({
      source: "runtime",

      provider: "openrouter",

      model,

      judge_model: judgeModel,

      status: "partial",

      total_questions: 0,

      successful: 0,

      failed: 0,

      overall_score: null,

      by_category: {},

      by_difficulty: {},

      by_criterion: {},

      lowest_scoring: [],

      error_summary: {
        evaluation_error: 1,
      },

      results: [],

      error: message,
    });
  }
}