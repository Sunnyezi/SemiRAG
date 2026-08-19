"""Command-line entry point for SemiRAG."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SemiRAG workflow.")
    parser.add_argument(
        "workflow",
        choices=("agentic", "adaptive"),
        help="agentic: tool-calling RAG; adaptive: routing and self-correcting RAG.",
    )
    args = parser.parse_args()

    if args.workflow == "agentic":
        from semirag.workflows.agentic.graph1 import main as run_workflow
    else:
        from semirag.workflows.adaptive.graph_2 import main as run_workflow

    run_workflow()


if __name__ == "__main__":
    main()
