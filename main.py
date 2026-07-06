from llm.groq_client import LLMError
from workflow.rag_workflow import run_workflow


def main():
    print("Agentic RAG — ask a question ('exit' to quit)")
    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        try:
            trace = run_workflow(query, return_trace=True)
        except LLMError as error:
            print(f"LLM call failed: {error}")
            continue

        print(trace["answer"])

        sources = {
            (chunk.get("source", "unknown"), chunk.get("page", "?"))
            for chunk in trace["retrieved_chunks"]
            if isinstance(chunk, dict)
        }
        if sources:
            print("\nSources:")
            for source, page in sorted(sources, key=str):
                print(f"  - {source}, page {page}")


if __name__ == "__main__":
    main()
