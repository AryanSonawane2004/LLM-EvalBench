const OPENROUTER_CHAT_URL =
    "https://openrouter.ai/api/v1/chat/completions";

const OPENROUTER_MODELS_URL =
    "https://openrouter.ai/api/v1/models";

const MAX_RETRIES = 2;
const INITIAL_BACKOFF_MS = 1500;
const MAX_BACKOFF_MS = 10000;

const ALLOWED_CRITERIA = new Set([
    "correctness",
    "relevance",
    "completeness",
    "reasoning",
    "instruction_following",
    "faithfulness",
    "groundedness",
    "hallucination",
    "logical_consistency"
]);


// ============================================================
// RESPONSE HELPER
// ============================================================

function jsonResponse(data, status = 200) {
    return new Response(
        JSON.stringify(data),
        {
            status,
            headers: {
                "Content-Type": "application/json",
                "Cache-Control": "no-store"
            }
        }
    );
}


// ============================================================
// API KEY
// ============================================================

function getApiKey(req) {
    const apiKey =
        req.headers.get("x-api-key") ||
        req.headers.get("X-API-Key");

    if (!apiKey || !apiKey.trim()) {
        throw new Error(
            "OpenRouter API key is required."
        );
    }

    return apiKey.trim();
}


// ============================================================
// OPENROUTER ERROR
// ============================================================

async function extractOpenRouterError(response) {
    try {
        const data = await response.json();

        if (data?.error?.message) {
            return data.error.message;
        }

        if (data?.error) {
            return String(data.error);
        }

        if (data?.message) {
            return data.message;
        }
    } catch {
        // Ignore JSON parsing errors.
    }

    return `HTTP ${response.status}`;
}


// ============================================================
// OPENROUTER CHAT
// ============================================================

async function callOpenRouter({
    model,
    messages,
    apiKey,
    temperature = 0,
    maxTokens = 1024,
    jsonMode = false
}) {
    const payload = {
        model,
        messages,
        temperature,
        max_tokens: maxTokens
    };

    /*
     * Ask OpenRouter for JSON when supported.
     *
     * The prompt also explicitly requires JSON, so models
     * that don't support response_format can still work.
     */
    if (jsonMode) {
        payload.response_format = {
            type: "json_object"
        };
    }

    let lastError = "Unknown OpenRouter error.";

    for (
        let attempt = 0;
        attempt <= MAX_RETRIES;
        attempt++
    ) {
        try {
            const response = await fetch(
                OPENROUTER_CHAT_URL,
                {
                    method: "POST",
                    headers: {
                        "Authorization":
                            `Bearer ${apiKey}`,
                        "Content-Type":
                            "application/json",
                        "Accept":
                            "application/json"
                    },
                    body: JSON.stringify(payload)
                }
            );

            if (response.ok) {
                const data =
                    await response.json();

                const content =
                    data?.choices?.[0]?.message?.content;

                if (
                    content === undefined ||
                    content === null
                ) {
                    throw new Error(
                        "OpenRouter returned an empty response content."
                    );
                }

                const text =
                    typeof content === "string"
                        ? content.trim()
                        : String(content).trim();

                if (!text) {
                    throw new Error(
                        "OpenRouter returned an empty response."
                    );
                }

                return {
                    content: text,
                    raw: data
                };
            }

            lastError =
                await extractOpenRouterError(
                    response
                );

            // Authentication / permission
            if (
                response.status === 401 ||
                response.status === 403
            ) {
                throw new Error(
                    `OpenRouter API key is invalid, expired, or unauthorized: ${lastError}`
                );
            }

            // Insufficient credits
            if (response.status === 402) {
                throw new Error(
                    `Insufficient OpenRouter credits: ${lastError}`
                );
            }

            // Retry rate limits
            if (response.status === 429) {
                if (attempt >= MAX_RETRIES) {
                    throw new Error(
                        `OpenRouter rate limit exceeded: ${lastError}`
                    );
                }

                const delay =
                    Math.min(
                        INITIAL_BACKOFF_MS *
                            (2 ** attempt),
                        MAX_BACKOFF_MS
                    );

                await sleep(delay);
                continue;
            }

            // Retry server errors
            if (
                response.status >= 500 &&
                response.status <= 599
            ) {
                if (attempt >= MAX_RETRIES) {
                    throw new Error(
                        `OpenRouter server error: ${lastError}`
                    );
                }

                const delay =
                    Math.min(
                        INITIAL_BACKOFF_MS *
                            (2 ** attempt),
                        MAX_BACKOFF_MS
                    );

                await sleep(delay);
                continue;
            }

            throw new Error(
                `OpenRouter API error (${response.status}): ${lastError}`
            );

        } catch (error) {
            lastError =
                error instanceof Error
                    ? error.message
                    : String(error);

            /*
             * Don't retry authentication, credit,
             * validation, or other explicit errors.
             */
            if (
                lastError.includes(
                    "invalid, expired, or unauthorized"
                ) ||
                lastError.includes(
                    "Insufficient OpenRouter credits"
                ) ||
                lastError.includes(
                    "rate limit exceeded"
                )
            ) {
                throw error;
            }

            if (attempt >= MAX_RETRIES) {
                throw new Error(
                    `OpenRouter request failed: ${lastError}`
                );
            }

            const delay =
                Math.min(
                    INITIAL_BACKOFF_MS *
                        (2 ** attempt),
                    MAX_BACKOFF_MS
                );

            await sleep(delay);
        }
    }

    throw new Error(
        `OpenRouter request failed: ${lastError}`
    );
}


function sleep(ms) {
    return new Promise(
        resolve => setTimeout(resolve, ms)
    );
}


// ============================================================
// MODEL VALIDATION
// ============================================================

async function validateModel(
    model,
    apiKey
) {
    if (
        !model ||
        typeof model !== "string"
    ) {
        throw new Error(
            "Model name is required."
        );
    }

    const response = await fetch(
        OPENROUTER_MODELS_URL,
        {
            method: "GET",
            headers: {
                "Authorization":
                    `Bearer ${apiKey}`,
                "Accept":
                    "application/json"
            }
        }
    );

    if (!response.ok) {
        const message =
            await extractOpenRouterError(
                response
            );

        if (
            response.status === 401 ||
            response.status === 403
        ) {
            throw new Error(
                `OpenRouter API key is invalid, expired, or unauthorized: ${message}`
            );
        }

        throw new Error(
            `Failed to validate model: ${message}`
        );
    }

    const data =
        await response.json();

    const models =
        Array.isArray(data?.data)
            ? data.data
            : [];

    const target =
        model.trim().toLowerCase();

    const found =
        models.find(
            item =>
                String(item?.id || "")
                    .toLowerCase() === target
        );

    if (!found) {
        throw new Error(
            `OpenRouter model '${model}' was not found in the current catalog.`
        );
    }

    const inputModalities =
        found?.architecture
            ?.input_modalities || [];

    const outputModalities =
        found?.architecture
            ?.output_modalities || [];

    const benchmarkCompatible =
        inputModalities.includes("text") &&
        outputModalities.includes("text");

    if (!benchmarkCompatible) {
        throw new Error(
            `Model '${model}' is not compatible with text-to-text benchmarking.`
        );
    }

    return found;
}


// ============================================================
// DETERMINISTIC INSTRUCTION CHECKS
// ============================================================

function runDeterministicChecks(
    question,
    response
) {
    const prompt =
        String(question?.prompt || "");

    const checks = {};

    let passed = true;


    // --------------------------------------------------------
    // EXACTLY N SENTENCES
    // --------------------------------------------------------

    const sentenceMatch =
        prompt.match(
            /exactly\s+(\d+)\s+sentences?/i
        );

    if (sentenceMatch) {
        const required =
            Number(sentenceMatch[1]);

        /*
         * Count sentence-ending punctuation.
         * This intentionally mirrors the lightweight
         * machine-check approach used by the project.
         */
        const matches =
            response.match(
                /[.!?](?=\s|$)/g
            ) || [];

        const actual =
            matches.length;

        const check = {
            required,
            actual,
            passed: actual === required
        };

        checks.sentence_count = check;

        if (!check.passed) {
            passed = false;
        }
    }


    // --------------------------------------------------------
    // EXACTLY N BULLET POINTS
    // --------------------------------------------------------

    const bulletMatch =
        prompt.match(
            /exactly\s+(\d+)\s+bullet\s+points?/i
        );

    if (bulletMatch) {
        const required =
            Number(bulletMatch[1]);

        const lines =
            response
                .split(/\r?\n/)
                .map(line => line.trim())
                .filter(Boolean);

        const bulletLines =
            lines.filter(
                line =>
                    /^[-*•]\s+/.test(line) ||
                    /^\d+[.)]\s+/.test(line)
            );

        const actual =
            bulletLines.length;

        const check = {
            required,
            actual,
            passed: actual === required
        };

        checks.bullet_points = check;

        if (!check.passed) {
            passed = false;
        }
    }


    // --------------------------------------------------------
    // FORBIDDEN WORD
    // --------------------------------------------------------

    const forbiddenMatch =
        prompt.match(
            /do\s+not\s+use\s+the\s+word\s+['"]([^'"]+)['"]/i
        );

    if (forbiddenMatch) {
        const forbiddenWord =
            forbiddenMatch[1];

        const regex =
            new RegExp(
                `\\b${escapeRegExp(forbiddenWord)}\\b`,
                "i"
            );

        const contains =
            regex.test(response);

        const check = {
            word: forbiddenWord,
            passed: !contains
        };

        checks.forbidden_word = check;

        if (!check.passed) {
            passed = false;
        }
    }


    return {
        passed,
        checks
    };
}


function escapeRegExp(value) {
    return value.replace(
        /[.*+?^${}()|[\]\\]/g,
        "\\$&"
    );
}


// ============================================================
// JUDGE PROMPT
// ============================================================

function buildJudgePrompt(
    question,
    modelResponse,
    deterministicChecks
) {
    const prompt =
        question?.prompt || "";

    const referenceAnswer =
        question?.reference_answer || "";

    const criteria =
        Array.isArray(
            question?.evaluation_criteria
        )
            ? question.evaluation_criteria
            : [];

    const category =
        question?.category || "";

    const difficulty =
        question?.difficulty || "";

    const criteriaText =
        criteria
            .map(
                criterion =>
                    `- ${criterion}`
            )
            .join("\n");

    const deterministicEvidence =
        JSON.stringify(
            deterministicChecks,
            null,
            2
        );

    return `
You are an expert LLM evaluator.

Evaluate the candidate model response against the benchmark
prompt, reference answer, requested evaluation criteria,
and deterministic validation evidence.

Be strict and evidence-based.

Do not reward an answer simply because it sounds confident.

Do not invent facts.

IMPORTANT:
When deterministic validation reports that an explicit
instruction was violated, treat that result as authoritative
evidence for the instruction_following criterion.

For every requested criterion, provide:
1. A score from 0 to 5.
2. A concise rationale explaining the score.

Scoring scale:

0 = completely fails
1 = major problems
2 = substantial problems
3 = acceptable / partially successful
4 = strong
5 = excellent

Criterion definitions:

correctness:
Whether the answer reaches the correct conclusion.

relevance:
Whether the answer directly addresses the prompt.

completeness:
Whether the important required information is present.

reasoning:
Whether the reasoning is logically sound.

instruction_following:
Whether explicit instructions were followed.
Use the deterministic validation evidence when available.

faithfulness:
Whether the answer accurately represents the supplied information.

groundedness:
Whether claims are supported by the supplied information.

hallucination:
Whether unsupported information was introduced.

logical_consistency:
Whether the answer is internally consistent.

Benchmark category:
${category}

Benchmark difficulty:
${difficulty}

PROMPT:
${prompt}

REFERENCE ANSWER:
${referenceAnswer}

CANDIDATE MODEL RESPONSE:
${modelResponse}

REQUESTED EVALUATION CRITERIA:
${criteriaText}

DETERMINISTIC VALIDATION:
${deterministicEvidence}

Return ONLY valid JSON.

Required structure:

{
  "criteria_scores": {
    "criterion_name": {
      "score": 0,
      "rationale": "..."
    }
  },
  "overall_rationale": "..."
}

Rules:

- Include every requested criterion.
- Do not include criteria that were not requested.
- Every score must be an integer from 0 to 5.
- Keep rationales concise.
- Do not use Markdown.
- Do not wrap the JSON in code fences.
`.trim();
}


// ============================================================
// JSON EXTRACTION
// ============================================================

function extractJson(text) {
    if (!text) {
        throw new Error(
            "Judge returned an empty response."
        );
    }

    let cleaned =
        String(text).trim();

    // Remove Markdown fences if a model ignores
    // the instruction to return raw JSON.
    cleaned =
        cleaned.replace(
            /^```json\s*/i,
            ""
        );

    cleaned =
        cleaned.replace(
            /^```\s*/i,
            ""
        );

    cleaned =
        cleaned.replace(
            /\s*```$/i,
            ""
        );

    try {
        return JSON.parse(cleaned);
    } catch {
        /*
         * Try extracting the first JSON object
         * from surrounding text.
         */
        const start =
            cleaned.indexOf("{");

        const end =
            cleaned.lastIndexOf("}");

        if (
            start === -1 ||
            end === -1 ||
            end <= start
        ) {
            throw new Error(
                "Judge returned invalid JSON."
            );
        }

        try {
            return JSON.parse(
                cleaned.slice(
                    start,
                    end + 1
                )
            );
        } catch {
            throw new Error(
                "Judge returned invalid JSON."
            );
        }
    }
}


// ============================================================
// SCORE VALIDATION
// ============================================================

function normalizeScore(value) {
    const number =
        Number(value);

    if (!Number.isFinite(number)) {
        throw new Error(
            `Invalid judge score: ${value}`
        );
    }

    return Math.max(
        0,
        Math.min(
            5,
            Math.round(number)
        )
    );
}


function validateEvaluation(
    evaluation,
    requestedCriteria
) {
    if (
        !evaluation ||
        typeof evaluation !== "object"
    ) {
        throw new Error(
            "Judge response must be a JSON object."
        );
    }

    const criteriaScores =
        evaluation.criteria_scores;

    if (
        !criteriaScores ||
        typeof criteriaScores !== "object" ||
        Array.isArray(criteriaScores)
    ) {
        throw new Error(
            "Judge response is missing 'criteria_scores'."
        );
    }

    const normalizedScores = {};

    for (
        const criterion
        of requestedCriteria
    ) {
        if (
            !ALLOWED_CRITERIA.has(
                criterion
            )
        ) {
            throw new Error(
                `Unsupported criterion: ${criterion}`
            );
        }

        if (
            !Object.prototype.hasOwnProperty.call(
                criteriaScores,
                criterion
            )
        ) {
            throw new Error(
                `Judge did not score criterion: ${criterion}`
            );
        }

        const result =
            criteriaScores[criterion];

        if (
            !result ||
            typeof result !== "object"
        ) {
            throw new Error(
                `Invalid result for criterion: ${criterion}`
            );
        }

        if (
            !Object.prototype.hasOwnProperty.call(
                result,
                "score"
            )
        ) {
            throw new Error(
                `Missing score for criterion: ${criterion}`
            );
        }

        normalizedScores[criterion] = {
            score:
                normalizeScore(
                    result.score
                ),
            rationale:
                String(
                    result.rationale || ""
                ).trim()
        };
    }

    const scores =
        Object.values(
            normalizedScores
        ).map(
            item => item.score
        );

    const overallScore =
        scores.length
            ? Math.round(
                (
                    scores.reduce(
                        (sum, score) =>
                            sum + score,
                        0
                    ) /
                    scores.length
                ) * 100
            ) / 100
            : 0;

    return {
        overall_score:
            overallScore,

        criteria_scores:
            normalizedScores,

        overall_rationale:
            String(
                evaluation.overall_rationale ||
                ""
            ).trim()
    };
}


// ============================================================
// SINGLE QUESTION EVALUATION
// ============================================================

export default async (req, context) => {
    if (req.method !== "POST") {
        return jsonResponse(
            {
                error:
                    "Method not allowed"
            },
            405
        );
    }

    try {
        const apiKey =
            getApiKey(req);

        let body;

        try {
            body =
                await req.json();
        } catch {
            return jsonResponse(
                {
                    error:
                        "Request body must be valid JSON."
                },
                400
            );
        }

        const questionId =
            body?.question_id;

        const model =
            body?.model;

        const judgeModel =
            body?.judge_model;

        if (!questionId) {
            return jsonResponse(
                {
                    error:
                        "question_id is required."
                },
                400
            );
        }

        if (!model) {
            return jsonResponse(
                {
                    error:
                        "model is required."
                },
                400
            );
        }

        if (!judgeModel) {
            return jsonResponse(
                {
                    error:
                        "judge_model is required."
                },
                400
            );
        }


        // ----------------------------------------------------
        // Load benchmark through our Netlify benchmark function
        // ----------------------------------------------------

        const benchmarkUrl =
            new URL(
                "/.netlify/functions/benchmark",
                req.url
            );

        const benchmarkResponse =
            await fetch(
                benchmarkUrl
            );

        if (!benchmarkResponse.ok) {
            const message =
                await benchmarkResponse.text();

            throw new Error(
                `Failed to load benchmark: ${message}`
            );
        }

        const benchmarkData =
            await benchmarkResponse.json();

        const benchmark =
            Array.isArray(
                benchmarkData?.questions
            )
                ? benchmarkData.questions
                : [];

        const question =
            benchmark.find(
                item =>
                    item?.id ===
                    questionId
            );

        if (!question) {
            return jsonResponse(
                {
                    error:
                        `Question '${questionId}' not found.`
                },
                404
            );
        }


        // ----------------------------------------------------
        // Validate models
        // ----------------------------------------------------

        try {
            await validateModel(
                model,
                apiKey
            );

            await validateModel(
                judgeModel,
                apiKey
            );
        } catch (error) {
            return jsonResponse(
                {
                    error:
                        error instanceof Error
                            ? error.message
                            : String(error)
                },
                400
            );
        }


        // ----------------------------------------------------
        // Candidate model
        // ----------------------------------------------------

        let modelResponse;

        try {
            const result =
                await callOpenRouter({
                    model,
                    apiKey,
                    messages: [
                        {
                            role: "user",
                            content:
                                question.prompt
                        }
                    ],
                    temperature: 0,
                    maxTokens: 1024
                });

            modelResponse =
                result.content;

        } catch (error) {
            const message =
                error instanceof Error
                    ? error.message
                    : String(error);

            const status =
                message.includes(
                    "Insufficient OpenRouter credits"
                )
                    ? 402
                    : message.includes(
                        "invalid, expired, or unauthorized"
                    )
                        ? 401
                        : message.includes(
                            "rate limit"
                        )
                            ? 429
                            : 502;

            return jsonResponse(
                {
                    error:
                        `Candidate model failed: ${message}`
                },
                status
            );
        }


        // ----------------------------------------------------
        // Deterministic checks
        // ----------------------------------------------------

        const deterministicChecks =
            runDeterministicChecks(
                question,
                modelResponse
            );


        // ----------------------------------------------------
        // Judge
        // ----------------------------------------------------

        const judgePrompt =
            buildJudgePrompt(
                question,
                modelResponse,
                deterministicChecks
            );

        let rawJudgeResponse;

        try {
            const result =
                await callOpenRouter({
                    model: judgeModel,
                    apiKey,
                    messages: [
                        {
                            role: "system",
                            content:
                                "You are a strict LLM evaluation judge. Return only valid JSON."
                        },
                        {
                            role: "user",
                            content:
                                judgePrompt
                        }
                    ],
                    temperature: 0,
                    maxTokens: 1024,
                    jsonMode: true
                });

            rawJudgeResponse =
                result.content;

        } catch (error) {
            const message =
                error instanceof Error
                    ? error.message
                    : String(error);

            const status =
                message.includes(
                    "Insufficient OpenRouter credits"
                )
                    ? 402
                    : message.includes(
                        "invalid, expired, or unauthorized"
                    )
                        ? 401
                        : message.includes(
                            "rate limit"
                        )
                            ? 429
                            : 502;

            return jsonResponse(
                {
                    error:
                        `Judge model failed: ${message}`
                },
                status
            );
        }


        // ----------------------------------------------------
        // Parse + validate judge response
        // ----------------------------------------------------

        let parsedJudge;

        try {
            parsedJudge =
                extractJson(
                    rawJudgeResponse
                );
        } catch (error) {
            return jsonResponse(
                {
                    error:
                        error instanceof Error
                            ? error.message
                            : String(error)
                },
                502
            );
        }

        let evaluation;

        try {
            evaluation =
                validateEvaluation(
                    parsedJudge,
                    question.evaluation_criteria
                );
        } catch (error) {
            return jsonResponse(
                {
                    error:
                        error instanceof Error
                            ? error.message
                            : String(error)
                },
                502
            );
        }


        // ----------------------------------------------------
        // Deterministic enforcement
        // ----------------------------------------------------

        if (
            question.evaluation_criteria.includes(
                "instruction_following"
            ) &&
            !deterministicChecks.passed
        ) {
            evaluation
                .criteria_scores
                .instruction_following = {
                    score: 0,
                    rationale:
                        "The response failed one or more explicit machine-checkable instruction constraints."
                };

            const scores =
                Object.values(
                    evaluation.criteria_scores
                ).map(
                    item => item.score
                );

            evaluation.overall_score =
                Math.round(
                    (
                        scores.reduce(
                            (sum, score) =>
                                sum + score,
                            0
                        ) /
                        scores.length
                    ) * 100
                ) / 100;
        }


        // ----------------------------------------------------
        // Final metadata
        // ----------------------------------------------------

        evaluation.judge_model =
            judgeModel;

        evaluation.provider =
            "openrouter";

        evaluation.deterministic_checks =
            deterministicChecks;


        // ----------------------------------------------------
        // Final response
        // ----------------------------------------------------

        return jsonResponse(
            {
                question_id:
                    questionId,

                provider:
                    "openrouter",

                model,

                judge_model:
                    judgeModel,

                model_response:
                    modelResponse,

                evaluation
            },
            200
        );

    } catch (error) {
        console.error(
            "Evaluate function error:",
            error
        );

        return jsonResponse(
            {
                error:
                    error instanceof Error
                        ? error.message
                        : String(error)
            },
            500
        );
    }
};