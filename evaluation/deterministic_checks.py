import re
from typing import Any, Dict, List


def count_sentences(text: str) -> int:
    """
    Estimate sentence count using terminal punctuation.

    This is intentionally simple and deterministic. It is suitable
    for the benchmark's explicit sentence-count constraints.
    """

    if not text or not text.strip():
        return 0

    sentences = re.findall(
        r"[^.!?]+[.!?]+",
        text.strip(),
    )

    return len(sentences)


def find_forbidden_words(
    text: str,
    forbidden_words: List[str],
) -> List[str]:
    """
    Return forbidden words that occur as complete words.
    Matching is case-insensitive.
    """

    found = []

    for word in forbidden_words:
        pattern = rf"\b{re.escape(word)}\b"

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            found.append(word)

    return found


def count_bullets(text: str) -> int:
    """
    Count common Markdown-style bullet lines.
    """

    if not text:
        return 0

    lines = text.splitlines()

    return sum(
        1
        for line in lines
        if re.match(
            r"^\s*[-*+]\s+",
            line,
        )
    )


def check_instruction_constraints(
    question: Dict[str, Any],
    response: str,
) -> Dict[str, Any]:
    """
    Run deterministic checks for explicit instruction constraints.

    The benchmark currently expresses many constraints directly
    inside the prompt, so this function detects the constraints
    currently used by the custom benchmark.
    """

    prompt = question.get(
        "prompt",
        "",
    )

    result = {
        "passed": True,
        "checks": {},
    }

    # ---------------------------------------------------------
    # EXACT SENTENCE COUNT
    # ---------------------------------------------------------

    sentence_match = re.search(
        r"exactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?",
        prompt,
        flags=re.IGNORECASE,
    )

    if sentence_match:

        value = sentence_match.group(1).lower()

        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }

        required = (
            int(value)
            if value.isdigit()
            else number_words[value]
        )

        actual = count_sentences(
            response
        )

        passed = actual == required

        result["checks"]["sentence_count"] = {
            "required": required,
            "actual": actual,
            "passed": passed,
        }

        if not passed:
            result["passed"] = False

    # ---------------------------------------------------------
    # FORBIDDEN WORDS
    # ---------------------------------------------------------

    forbidden_words = re.findall(
        r"(?:do not use|don't use|must not use)"
        r"\s+(?:the\s+)?word\s+['\"]([^'\"]+)['\"]",
        prompt,
        flags=re.IGNORECASE,
    )

    if forbidden_words:

        found = find_forbidden_words(
            response,
            forbidden_words,
        )

        passed = len(found) == 0

        result["checks"]["forbidden_words"] = {
            "required": forbidden_words,
            "found": found,
            "passed": passed,
        }

        if not passed:
            result["passed"] = False

    # ---------------------------------------------------------
    # EXACT BULLET COUNT
    # ---------------------------------------------------------

    bullet_match = re.search(
        r"exactly\s+(\d+)\s+bullets?",
        prompt,
        flags=re.IGNORECASE,
    )

    if bullet_match:

        required = int(
            bullet_match.group(1)
        )

        actual = count_bullets(
            response
        )

        passed = actual == required

        result["checks"]["bullet_count"] = {
            "required": required,
            "actual": actual,
            "passed": passed,
        }

        if not passed:
            result["passed"] = False

    return result