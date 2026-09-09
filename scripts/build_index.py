from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import SELECTED_PAPERS, Settings
from research_assistant.ollama_client import OllamaClient
from research_assistant.pdf_pipeline import build_verified_result_chunks, extract_all_papers
from research_assistant.vector_store import FaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract, chunk, embed, and index the four selected papers.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    client = OllamaClient(
        settings.ollama_url,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        timeout=settings.request_timeout,
        seed=settings.random_seed,
    )
    client.ensure_models()
    chunks = extract_all_papers(
        SELECTED_PAPERS,
        chunk_words=settings.chunk_words,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks.extend(build_verified_result_chunks(SELECTED_PAPERS, settings.facts_path))
    store = FaissVectorStore(settings.index_dir, client)
    manifest = store.build(chunks, batch_size=args.batch_size)
    stats = {
        **manifest,
        "papers": [paper.short_name for paper in SELECTED_PAPERS],
        "iqr_outlier_chunks": sum(chunk.length_iqr_outlier for chunk in chunks),
        "mean_words": round(sum(chunk.word_count for chunk in chunks) / len(chunks), 2),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
