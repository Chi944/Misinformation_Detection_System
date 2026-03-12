"""Smoke test for the misinformation detector.

This script will later exercise the full pipeline on a handful of examples.
For now it performs lightweight environment checks, including Ollama health.
"""

import sys

import requests


def check_ollama(host: str = "http://localhost:11434") -> None:
    """Check that Ollama is reachable.

    This is a soft check: on failure we print a warning but do not exit with
    an error to keep the smoke test lightweight.
    """

    url = f"{host.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        print(f"Ollama reachable at {url}")
    except Exception:
        print(
            f"WARNING: Ollama not reachable at {url}. "
            "Run `ollama serve` and `ollama pull llama3` if you want LLM judging."
        )


def main() -> None:
    """Run a minimal smoke test: environment + Ollama health."""

    print("Running smoke test...")
    check_ollama()
    print("Smoke test completed.")


if __name__ == "__main__":
    sys.exit(main())

