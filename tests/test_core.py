from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.assistant import ResearchAssistant
from research_assistant.config import SELECTED_PAPERS, Settings
from research_assistant.pdf_pipeline import build_verified_result_chunks, extract_all_papers
from research_assistant.reference_analysis import extract_reference_year_counts
from research_assistant.research_tools import ResearchTools


def test_selected_core_has_four_existing_papers_and_excludes_self_rag() -> None:
    assert len(SELECTED_PAPERS) == 4
    assert {paper.paper_id for paper in SELECTED_PAPERS} == {"realm", "dpr", "rag", "fid"}
    assert all(paper.path.exists() for paper in SELECTED_PAPERS)
    assert all("self" not in paper.filename.casefold() for paper in SELECTED_PAPERS)


def test_verified_nq_comparison_is_consistent() -> None:
    facts = json.loads(Settings().facts_path.read_text(encoding="utf-8"))
    comparison = facts["core_comparison"]
    assert "Natural Questions" in comparison["dataset"]
    assert "Exact Match" in comparison["metric"]
    values = {row["paper"]: row["nq_em"] for row in comparison["rows"]}
    assert values == {"REALM": 40.4, "DPR": 41.5, "RAG": 44.5, "FiD": 51.4}


def test_pdf_chunking_respects_word_budget_and_unique_ids() -> None:
    settings = Settings()
    chunks = extract_all_papers(
        SELECTED_PAPERS,
        chunk_words=settings.chunk_words,
        chunk_overlap=settings.chunk_overlap,
    )
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunks) > 400
    assert len(ids) == len(set(ids))
    assert max(chunk.word_count for chunk in chunks) <= settings.chunk_words
    assert {chunk.paper_id for chunk in chunks} == {"realm", "dpr", "rag", "fid"}
    assert {"abstract", "introduction", "references"}.issubset({chunk.section for chunk in chunks})


def test_human_verified_table_records_are_traceable() -> None:
    settings = Settings()
    records = build_verified_result_chunks(SELECTED_PAPERS, settings.facts_path)
    table_records = [record for record in records if record.record_type == "human_verified_table_record"]
    method_records = [record for record in records if record.record_type == "human_verified_method_record"]
    assert len(table_records) == 4
    assert len(method_records) == 4
    assert all("Exact Match" in record.text and "PDF p." in record.text for record in table_records)


def test_reference_year_parser_finds_each_bibliography() -> None:
    totals = {
        paper.paper_id: sum(extract_reference_year_counts(paper.path).values())
        for paper in SELECTED_PAPERS
    }
    assert all(total > 20 for total in totals.values())


def test_tools_and_fallback_router_cover_required_operations() -> None:
    settings = Settings()
    tools = ResearchTools(settings)
    assert set(tools.function_names) == {
        "extract_experimental_results",
        "compare_results",
        "generate_survey",
        "find_research_gap",
        "compare_references",
        "compare_limitations",
    }
    assistant = ResearchAssistant(settings)
    assert assistant._fallback_route("مقایسه نتایج و نمودار") == "compare_results"
    assert assistant._fallback_route("شکاف پژوهشی و کار آینده") == "find_research_gap"
    assert assistant._fallback_route("How does FiD work?") is None

