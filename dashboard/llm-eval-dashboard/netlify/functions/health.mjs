export default async (req, context) => {
    return new Response(
        JSON.stringify({
            status: "ok",
            service: "LLM-EvalBench",
            environment: "netlify",
            timestamp: new Date().toISOString()
        }),
        {
            status: 200,
            headers: {
                "Content-Type": "application/json"
            }
        }
    );
};