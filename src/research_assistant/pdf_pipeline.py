from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz
import numpy as np

from .config import PaperSpec


SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE)),
    ("introduction", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?introduction(?:\s+.*)?$", re.IGNORECASE)),
    ("related_work", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:related\s+work|background)(?:\s+.*)?$", re.IGNORECASE)),
    ("methodology", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:method|methodology|model|approach|architecture|retrieval|pre-?training|training)(?:\s+.*)?$", re.IGNORECASE)),
    ("experiments", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:experiments?|experimental\s+setup|evaluation|implementation|tasks?|open-domain\s+question\s+answering)(?:\s+.*)?$", re.IGNORECASE)),
    ("results", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:results?|analysis|ablation)(?:\s+.*)?$", re.IGNORECASE)),
    ("limitations", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?limitations?(?:\s+.*)?$", re.IGNORECASE)),
    ("conclusion", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:conclusions?|discussion|broader\s+impact)(?:\s+.*)?$", re.IGNORECASE)),
    ("references", re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?references\s*$", re.IGNORECASE)),
    ("appendix", re.compile(r"^\s*appendix(?:\s+[A-Z0-9].*)?\s*$", re.IGNORECASE)),
)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_short_name: str
    paper_title: str
    source_filename: str
    source_sha256: str
    page: int
    section: str
    text: str
    word_count: int
    chunk_index_on_page: int
    length_z_score: float = 0.0
    length_iqr_outlier: bool = False
    record_type: str = "pdf_chunk"

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detect_section(line: str) -> str | None:
    candidate = re.sub(r"\s+", " ", line.strip())
    if not candidate or len(candidate) > 90:
        return None
    for section, pattern in SECTION_PATTERNS:
        if pattern.match(candidate):
            return section
    return None


def _clean_page_text(raw_text: str) -> str:
    text = raw_text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _chunk_words(text: str, size: int, overlap: int) -> Iterable[tuple[int, str]]:
    words = text.split()
    if not words:
        return
    step = max(1, size - overlap)
    chunk_index = 0
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if len(piece) < 25 and start > 0:
            break
        yield chunk_index, " ".join(piece)
        chunk_index += 1
        if start + size >= len(words):
            break


def extract_paper_chunks(
    paper: PaperSpec,
    *,
    chunk_words: int = 180,
    chunk_overlap: int = 35,
) -> list[Chunk]:
    if not paper.path.exists():
        raise FileNotFoundError(f"Selected paper is missing: {paper.path}")

    source_hash = file_sha256(paper.path)
    chunks: list[Chunk] = []
    current_section = "front_matter"

    with fitz.open(paper.path) as document:
        for page_index, page in enumerate(document):
            raw_text = page.get_text("text", sort=True)
            lines = raw_text.splitlines()
            page_segments: list[tuple[str, list[str]]] = []
            active_section = current_section
            active_lines: list[str] = []
            for line in lines:
                section = _detect_section(line)
                if section:
                    if active_lines:
                        page_segments.append((active_section, active_lines))
                    current_section = section
                    active_section = section
                    active_lines = [line]
                else:
                    active_lines.append(line)
            if active_lines:
                page_segments.append((active_section, active_lines))

            page_chunk_index = 0
            for segment_section, segment_lines in page_segments:
                cleaned = _clean_page_text("\n".join(segment_lines))
                if not cleaned:
                    continue
                for _, piece in _chunk_words(cleaned, chunk_words, chunk_overlap):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{paper.paper_id}-p{page_index + 1:03d}-c{page_chunk_index:02d}",
                            paper_id=paper.paper_id,
                            paper_short_name=paper.short_name,
                            paper_title=paper.title,
                            source_filename=paper.filename,
                            source_sha256=source_hash,
                            page=page_index + 1,
                            section=segment_section,
                            text=piece,
                            word_count=len(piece.split()),
                            chunk_index_on_page=page_chunk_index,
                        )
                    )
                    page_chunk_index += 1
    return chunks


def annotate_chunk_length_outliers(chunks: list[Chunk]) -> list[Chunk]:
    if not chunks:
        return chunks
    values = np.asarray([chunk.word_count for chunk in chunks], dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std()) or 1.0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    for chunk in chunks:
        chunk.length_z_score = round((chunk.word_count - mean) / std, 4)
        chunk.length_iqr_outlier = not (lower <= chunk.word_count <= upper)
    return chunks


def extract_all_papers(
    papers: Iterable[PaperSpec],
    *,
    chunk_words: int = 180,
    chunk_overlap: int = 35,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for paper in papers:
        chunks.extend(
            extract_paper_chunks(
                paper,
                chunk_words=chunk_words,
                chunk_overlap=chunk_overlap,
            )
        )
    return annotate_chunk_length_outliers(chunks)


def build_verified_result_chunks(papers: Iterable[PaperSpec], facts_path: Path) -> list[Chunk]:
    """Create traceable records for multi-column result tables.

    PDF text extraction often interleaves table columns. These records do not add
    outside claims: every value is human-verified against a named PDF page/table.
    """

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    rows = {row["paper"].casefold(): row for row in facts["core_comparison"]["rows"]}
    paper_facts = facts["papers"]
    method_pages = {"realm": 3, "dpr": 2, "rag": 2, "fid": 3}
    records: list[Chunk] = []
    for paper in papers:
        row = rows[paper.short_name.casefold()]
        page_match = re.search(r"p\.(\d+)", row["source"], re.IGNORECASE)
        page = int(page_match.group(1)) if page_match else 1
        text = (
            f"{paper.short_name} experimental result. Dataset: Natural Questions (NQ) open-domain standard test split. "
            f"Metric: normalized answer-level Exact Match (EM). Reported variant: {row['variant']}. "
            f"NQ test Exact Match score: {row['nq_em']}. Source: {row['source']}."
        )
        records.append(
            Chunk(
                chunk_id=f"{paper.paper_id}-verified-result",
                paper_id=paper.paper_id,
                paper_short_name=paper.short_name,
                paper_title=paper.title,
                source_filename=paper.filename,
                source_sha256=file_sha256(paper.path),
                page=page,
                section="structured_results",
                text=text,
                word_count=len(text.split()),
                chunk_index_on_page=999,
                record_type="human_verified_table_record",
            )
        )
        details = paper_facts[paper.paper_id]
        method_text = (
            f"{paper.short_name} method. {details['method_summary']} "
            f"Retriever: {details['retriever']}. Generator or reader: {details['generator_or_reader']}."
        )
        records.append(
            Chunk(
                chunk_id=f"{paper.paper_id}-verified-method",
                paper_id=paper.paper_id,
                paper_short_name=paper.short_name,
                paper_title=paper.title,
                source_filename=paper.filename,
                source_sha256=file_sha256(paper.path),
                page=method_pages[paper.paper_id],
                section="structured_method",
                text=method_text,
                word_count=len(method_text.split()),
                chunk_index_on_page=998,
                record_type="human_verified_method_record",
            )
        )
    return records
