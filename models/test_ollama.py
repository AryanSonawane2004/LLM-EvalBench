import ollama


def test_ollama():
    model_name = "qwen3:4b"

    print(f"Testing Ollama with model: {model_name}")
    print("Sending test prompt...\n")

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "What is the capital of Australia? Answer in one sentence."
            }
        ]
    )

    content = response["message"]["content"]

    print("Ollama response:")
    print(content)


if __name__ == "__main__":
    test_ollama()