import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export default async (req, context) => {
    if (req.method !== "GET") {
        return new Response(
            JSON.stringify({
                error: "Method not allowed"
            }),
            {
                status: 405,
                headers: {
                    "Content-Type": "application/json",
                    "Allow": "GET"
                }
            }
        );
    }

    try {
        const functionDir = path.dirname(fileURLToPath(import.meta.url));

        const possiblePaths = [
            path.join(functionDir, "data", "benchmark.jsonl"),
            path.join(functionDir, "..", "data", "benchmark.jsonl"),
            path.resolve(process.cwd(), "data", "benchmark.jsonl"),
        ];

        let fileContent = null;
        let benchmarkPath = null;

        for (const candidate of possiblePaths) {
            try {
                fileContent = await readFile(candidate, "utf-8");
                benchmarkPath = candidate;
                break;
            } catch {
                // Try the next possible location
            }
        }

        if (!fileContent) {
            throw new Error(
                `benchmark.jsonl not found. Tried:\n${possiblePaths.join("\n")}`
            );
        }

        console.log(
            `[benchmark] Loading dataset from: ${benchmarkPath}`
        );

        const questions = [];

        const lines = fileContent
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter((line) => line.length > 0);

        for (const line of lines) {
            try {
                const question = JSON.parse(line);

                if (!question.id) continue;
                if (!question.category) continue;
                if (!question.difficulty) continue;
                if (!question.prompt) continue;
                if (!question.reference_answer) continue;
                if (!Array.isArray(question.evaluation_criteria)) continue;

                questions.push(question);
            } catch (parseError) {
                console.error(
                    "[benchmark] Skipping invalid JSONL line:",
                    parseError
                );
            }
        }

        if (questions.length === 0) {
            return new Response(
                JSON.stringify({
                    error: "Benchmark contains no valid questions"
                }),
                {
                    status: 500,
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );
        }

        console.log(
            `[benchmark] Loaded ${questions.length} questions`
        );

        return new Response(
            JSON.stringify({
                questions,
                total: questions.length
            }),
            {
                status: 200,
                headers: {
                    "Content-Type": "application/json",
                    "Cache-Control": "public, max-age=300"
                }
            }
        );

    } catch (error) {
        console.error(
            "[benchmark] Function error:",
            error
        );

        return new Response(
            JSON.stringify({
                error: "Failed to load benchmark",
                details: error instanceof Error
                    ? error.message
                    : String(error)
            }),
            {
                status: 500,
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );
    }
};