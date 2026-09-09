from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PaperSpec:
    paper_id: str
    short_name: str
    title: str
    filename: str

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / "Researchs" / self.filename


SELECTED_PAPERS: tuple[PaperSpec, ...] = (
    PaperSpec(
        paper_id="realm",
        short_name="REALM",
        title="REALM: Retrieval-Augmented Language Model Pre-Training",
        filename="REALM- Retrieval-Augmented Language Model Pre-Training.pdf",
    ),
    PaperSpec(
        paper_id="dpr",
        short_name="DPR",
        title="Dense Passage Retrieval for Open-Domain Question Answering",
        filename="Dense_Passage_Retrieval_for_Open_Domain_Question_Answering.pdf",
    ),
    PaperSpec(
        paper_id="rag",
        short_name="RAG",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        filename="Retrieval_Augmented_Generation_for_Knowledge_Intensive_NLP_Tasks.pdf",
    ),
    PaperSpec(
        paper_id="fid",
        short_name="FiD",
        title="Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
        filename="Leveraging_Passage_Retrieval_with_Generative_Models_for_Open_Domain.pdf",
    ),
)


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-minilm")
    chunk_words: int = int(os.getenv("CHUNK_WORDS", "80"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "15"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.47"))
    request_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "180"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "faiss"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def facts_path(self) -> Path:
        return self.project_root / "data" / "paper_facts.json"

    @property
    def evaluation_path(self) -> Path:
        return self.project_root / "data" / "evaluation_questions.json"

    def ensure_directories(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
