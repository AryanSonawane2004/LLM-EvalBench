export default async (req, context) => {
    // Only allow POST requests
    if (req.method !== "POST") {
        return new Response(
            JSON.stringify({
                error: "Method not allowed"
            }),
            {
                status: 405,
                headers: {
                    "Content-Type": "application/json",
                    "Allow": "POST"
                }
            }
        );
    }

    try {
        // Read the API key from the request header
        const apiKey =
            req.headers.get("x-api-key") ||
            req.headers.get("X-API-Key");

        if (!apiKey || !apiKey.trim()) {
            return new Response(
                JSON.stringify({
                    valid: false,
                    error: "API key is required"
                }),
                {
                    status: 400,
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );
        }

        // Verify the key directly against OpenRouter
        const response = await fetch(
            "https://openrouter.ai/api/v1/auth/key",
            {
                method: "GET",
                headers: {
                    "Authorization": `Bearer ${apiKey.trim()}`
                }
            }
        );

        if (response.ok) {
            let data = {};

            try {
                data = await response.json();
            } catch {
                data = {};
            }

            return new Response(
                JSON.stringify({
                    valid: true,
                    message: "API key is valid",
                    data: data.data || data
                }),
                {
                    status: 200,
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );
        }

        // Invalid API key
        let errorData = {};

        try {
            errorData = await response.json();
        } catch {
            errorData = {};
        }

        return new Response(
            JSON.stringify({
                valid: false,
                error:
                    errorData?.error?.message ||
                    errorData?.message ||
                    "Invalid API key"
            }),
            {
                status: 401,
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );

    } catch (error) {
        console.error("Verify API error:", error);

        return new Response(
            JSON.stringify({
                valid: false,
                error: "Failed to verify API key"
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