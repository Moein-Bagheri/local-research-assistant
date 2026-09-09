from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .ollama_client import OllamaClient
from .pdf_pipeline import Chunk


@dataclass(frozen=True)
class SearchHit:
    score: float
    metadata: dict[str, Any]

    @property
    def citation(self) -> str:
        return f"[{self.metadata['paper_short_name']}, p. {self.metadata['page']}]"


class FaissVectorStore:
    INDEX_FILE = "papers.index"
    METADATA_FILE = "metadata.json"
    EMBEDDINGS_FILE = "embeddings.npy"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, index_dir: Path, client: OllamaClient) -> None:
        self.index_dir = index_dir
        self.client = client
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    @property
    def index_path(self) -> Path:
        return self.index_dir / self.INDEX_FILE

    def exists(self) -> bool:
        return self.index_path.exists() and (self.index_dir / self.METADATA_FILE).exists()

    def build(self, chunks: list[Chunk], *, batch_size: int = 32) -> dict[str, Any]:
        if not chunks:
            raise ValueError("Cannot build an index without chunks.")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        texts = [chunk.text for chunk in chunks]
        embeddings = self.client.embed(texts, batch_size=batch_size)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        metadata = [chunk.to_dict() for chunk in chunks]

        faiss.write_index(index, str(self.index_path))
        np.save(self.index_dir / self.EMBEDDINGS_FILE, embeddings)
        (self.index_dir / self.METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "chunk_count": len(chunks),
            "dimension": int(embeddings.shape[1]),
            "embedding_model": self.client.embedding_model,
            "similarity": "cosine via L2-normalized inner product",
            "paper_hashes": {
                item.paper_id: item.source_sha256 for item in chunks
            },
        }
        (self.index_dir / self.MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.index = index
        self.metadata = metadata
        return manifest

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(
                "FAISS index is missing. Run `python scripts/build_index.py` first."
            )
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(
            (self.index_dir / self.METADATA_FILE).read_text(encoding="utf-8")
        )
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata have different row counts.")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[SearchHit]:
        if self.index is None:
            self.load()
        assert self.index is not None
        query_vector = self.client.embed([query])
        candidate_k = min(max(top_k * 4, top_k), self.index.ntotal)
        scores, indices = self.index.search(query_vector, candidate_k)
        hits: list[SearchHit] = []
        for score, index_id in zip(scores[0], indices[0], strict=True):
            if index_id < 0:
                continue
            item = self.metadata[int(index_id)]
            if paper_id and item["paper_id"] != paper_id:
                continue
            hits.append(SearchHit(score=float(score), metadata=item))
            if len(hits) >= top_k:
                break
        return hits

