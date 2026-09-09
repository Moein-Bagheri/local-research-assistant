from __future__ import annotations

import argparse
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from research_assistant import ResearchAssistant, Settings


COMMANDS = {
    "/results": "Extract experimental results and common metrics",
    "/compare": "Compare all papers on the common benchmark",
    "/survey": "Generate the mini survey",
    "/gaps": "Find research gaps and future work",
    "/references": "Compare bibliography years",
    "/limitations": "Compare method limitations",
    "/quit": "Exit",
}

COMMAND_QUERIES = {
    "/results": "Extract the experimental results and metrics from the selected papers.",
    "/compare": "Compare results of all papers and create the comparison chart.",
    "/survey": "Generate a mini survey of the four selected papers.",
    "/gaps": "Find the research gaps and future work in the papers.",
    "/references": "Compare references by publication year.",
    "/limitations": "Compare limitations of all four papers.",
}


def print_response(response) -> None:
    print("\n" + response.answer)
    print(f"\n[mode={response.mode}; route={response.route_source}; time={response.elapsed_seconds:.2f}s]")
    if response.hits:
        print("Retrieved sources:")
        for hit in response.hits[:3]:
            meta = hit.metadata
            print(f"  - {meta['paper_short_name']} p.{meta['page']} | cosine={hit.score:.3f} | {meta['section']}")
    if response.artifacts:
        print("Artifacts:")
        for path in response.artifacts:
            print(f"  - {path}")


def interactive(assistant: ResearchAssistant) -> None:
    print("\nLocal Research Assistant - REALM / DPR / RAG / FiD")
    print(f"LLM: {assistant.settings.llm_model} | Embedding: {assistant.settings.embedding_model} | DB: FAISS")
    print("Commands:")
    for command, description in COMMANDS.items():
        print(f"  {command:<13} {description}")
    while True:
        try:
            query = input("\nAsk your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not query:
            continue
        if query == "/quit":
            print("Goodbye.")
            return
        query = COMMAND_QUERIES.get(query, query)
        try:
            print_response(assistant.ask(query))
        except Exception as exc:  # CLI boundary: keep the demo alive and show actionable errors.
            print(f"ERROR: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local paper research assistant")
    parser.add_argument("--question", "-q", help="Run one question and exit")
    parser.add_argument("--no-tools", action="store_true", help="Disable function-call routing")
    args = parser.parse_args()
    assistant = ResearchAssistant(Settings())
    if args.question:
        print_response(assistant.ask(args.question, allow_tools=not args.no_tools))
    else:
        interactive(assistant)


if __name__ == "__main__":
    main()
