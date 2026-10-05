// netlify/functions/models.mjs

const OPENROUTER_MODELS_URL =
  "https://openrouter.ai/api/v1/models";

// ------------------------------------------------------------
// JSON RESPONSE HELPER
// ------------------------------------------------------------

function jsonResponse(body, status = 200) {
  return new Response(
    JSON.stringify(body),
    {
      status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
      },
    }
  );
}

// ------------------------------------------------------------
// OPTIONS / CORS
// ------------------------------------------------------------

export default async function handler(request) {

  // ----------------------------------------------------------
  // CORS preflight
  // ----------------------------------------------------------

  if (request.method === "OPTIONS") {
    return jsonResponse(
      { ok: true },
      200
    );
  }

  // ----------------------------------------------------------
  // ONLY GET
  // ----------------------------------------------------------

  if (request.method !== "GET") {
    return jsonResponse(
      {
        error: "Method not allowed",
      },
      405
    );
  }

  // ----------------------------------------------------------
  // API KEY
  // ----------------------------------------------------------

  const apiKey =
    request.headers.get("X-API-Key");

  if (!apiKey || !apiKey.trim()) {
    return jsonResponse(
      {
        error: "API key is required",
      },
      401
    );
  }

  // ----------------------------------------------------------
  // FETCH OPENROUTER MODELS
  // ----------------------------------------------------------

  try {

    const response =
      await fetch(
        OPENROUTER_MODELS_URL,
        {
          method: "GET",

          headers: {
            "Authorization":
              `Bearer ${apiKey.trim()}`,

            "Accept":
              "application/json",
          },
        }
      );

    // --------------------------------------------------------
    // HANDLE OPENROUTER ERROR
    // --------------------------------------------------------

    if (!response.ok) {

      let errorMessage =
        `OpenRouter returned HTTP ${response.status}.`;

      try {

        const errorData =
          await response.json();

        if (
          errorData?.error?.message
        ) {
          errorMessage =
            errorData.error.message;
        }

        else if (
          typeof errorData?.error ===
          "string"
        ) {
          errorMessage =
            errorData.error;
        }

      } catch {
        // Keep default error message.
      }

      return jsonResponse(
        {
          error: errorMessage,
          status: response.status,
        },
        response.status
      );
    }

    // --------------------------------------------------------
    // PARSE RESPONSE
    // --------------------------------------------------------

    const data =
      await response.json();

    if (
      !data ||
      !Array.isArray(data.data)
    ) {

      return jsonResponse(
        {
          error:
            "OpenRouter returned an unexpected models response.",
          models: [],
        },
        502
      );
    }

    // --------------------------------------------------------
    // NORMALIZE MODELS
    // --------------------------------------------------------

    const models =
      data.data
        .map((model) => {

          const architecture =
            model?.architecture || {};

          const inputModalities =
            Array.isArray(
              architecture.input_modalities
            )
              ? architecture.input_modalities
              : [];

          const outputModalities =
            Array.isArray(
              architecture.output_modalities
            )
              ? architecture.output_modalities
              : [];

          const supportedParameters =
            Array.isArray(
              model?.supported_parameters
            )
              ? model.supported_parameters
              : [];

          // --------------------------------------------------
          // TEXT SUPPORT
          // --------------------------------------------------

          const acceptsText =
            inputModalities.length === 0 ||
            inputModalities.includes("text");

          const producesText =
            outputModalities.length === 0 ||
            outputModalities.includes("text");

          // --------------------------------------------------
          // BENCHMARK COMPATIBILITY
          //
          // Our benchmark sends text prompts and expects
          // text responses.
          //
          // If OpenRouter doesn't provide modality metadata,
          // we assume compatibility rather than incorrectly
          // eliminating the model.
          // --------------------------------------------------

          const benchmarkCompatible =
            acceptsText &&
            producesText;

          // --------------------------------------------------
          // PROVIDER
          // --------------------------------------------------

          let provider =
            "OpenRouter";

          if (
            typeof model?.id === "string" &&
            model.id.includes("/")
          ) {
            provider =
              model.id.split("/")[0];
          }

          // --------------------------------------------------
          // PRICING
          // --------------------------------------------------

          const pricing =
            model?.pricing || {};

          // --------------------------------------------------
          // RETURN NORMALIZED MODEL
          // --------------------------------------------------

          return {
            id:
              model?.id || "",

            name:
              model?.name ||
              model?.id ||
              "Unknown model",

            provider,

            description:
              model?.description ||
              "",

            capabilities:
              supportedParameters,

            benchmark_compatible:
              benchmarkCompatible,

            input_modalities:
              inputModalities,

            output_modalities:
              outputModalities,

            context_length:
              Number.isFinite(
                model?.context_length
              )
                ? model.context_length
                : null,

            max_completion_tokens:
              Number.isFinite(
                model?.top_provider
                  ?.max_completion_tokens
              )
                ? model.top_provider
                    .max_completion_tokens
                : null,

            pricing: {
              prompt:
                pricing.prompt ??
                null,

              completion:
                pricing.completion ??
                null,
            },

            architecture: {
              modality:
                architecture.modality ??
                null,

              tokenizer:
                architecture.tokenizer ??
                null,

              instruct_type:
                architecture.instruct_type ??
                null,
            },
          };
        })

        // ----------------------------------------------------
        // REMOVE INVALID ENTRIES
        // ----------------------------------------------------

        .filter(
          (model) =>
            model.id &&
            model.name
        )

        // ----------------------------------------------------
        // SORT ALPHABETICALLY
        // ----------------------------------------------------

        .sort(
          (a, b) =>
            a.name.localeCompare(
              b.name
            )
        );

    // --------------------------------------------------------
    // COMPATIBLE MODEL COUNT
    // --------------------------------------------------------

    const compatibleModels =
      models.filter(
        (model) =>
          model.benchmark_compatible
      );

    // --------------------------------------------------------
    // RESPONSE
    // --------------------------------------------------------

    return jsonResponse(
      {
        provider:
          "openrouter",

        total:
          models.length,

        benchmark_compatible:
          compatibleModels.length,

        models,
      },
      200
    );

  } catch (error) {

    console.error(
      "OpenRouter models error:",
      error
    );

    return jsonResponse(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch OpenRouter models.",

        models: [],
      },
      500
    );
  }
}