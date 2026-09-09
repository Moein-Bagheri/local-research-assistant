from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import SELECTED_PAPERS, Settings
from .reference_analysis import extract_reference_year_counts


@dataclass(frozen=True)
class ToolOutput:
    name: str
    text: str
    artifacts: tuple[Path, ...] = ()


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]
    header_line = "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


class ResearchTools:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.facts = json.loads(settings.facts_path.read_text(encoding="utf-8"))
        self._functions: dict[str, Callable[..., ToolOutput]] = {
            "extract_experimental_results": self.extract_experimental_results,
            "compare_results": self.compare_results,
            "generate_survey": self.generate_survey,
            "find_research_gap": self.find_research_gap,
            "compare_references": self.compare_references,
            "compare_limitations": self.compare_limitations,
        }

    @property
    def function_names(self) -> tuple[str, ...]:
        return tuple(self._functions)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolOutput:
        if name not in self._functions:
            raise KeyError(f"Unknown research tool: {name}")
        return self._functions[name](**(arguments or {}))

    def ollama_schemas(self) -> list[dict[str, Any]]:
        descriptions = {
            "extract_experimental_results": "Extract experimental metrics and verified values from the selected papers.",
            "compare_results": "Compare paper performance and create a table and chart for the common benchmark.",
            "generate_survey": "Generate a mini survey of the four selected papers.",
            "find_research_gap": "Find limitations, unsolved problems, and future research directions.",
            "compare_references": "Compare bibliography publication years across papers.",
            "compare_limitations": "Compare weaknesses and limitations of the selected methods.",
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name, description in descriptions.items()
        ]

    def _core_frame(self) -> pd.DataFrame:
        rows = self.facts["core_comparison"]["rows"]
        return pd.DataFrame(rows)[["paper", "variant", "nq_em", "source"]].rename(
            columns={
                "paper": "Paper",
                "variant": "Reported variant",
                "nq_em": "NQ test EM",
                "source": "PDF source",
            }
        )

    def extract_experimental_results(self) -> ToolOutput:
        core = self.facts["core_comparison"]
        frame = self._core_frame()
        auxiliary = pd.DataFrame(self.facts["auxiliary_metrics"])
        text = (
            f"Common benchmark: {core['dataset']} ({core['split']})\n"
            f"Common metric: {core['metric']}\n\n"
            f"{markdown_table(frame)}\n\n"
            "Additional metrics (kept separate because their definitions/splits differ):\n\n"
            f"{markdown_table(auxiliary)}\n\n"
            "Important: retrieval Hit@K/Accuracy@K is not the same quantity as answer Exact Match."
        )
        return ToolOutput("extract_experimental_results", text)

    def compare_results(self) -> ToolOutput:
        frame = self._core_frame().sort_values("NQ test EM", ascending=False)
        csv_path = self.settings.output_dir / "paper_comparison_nq_em.csv"
        chart_path = self.settings.output_dir / "paper_comparison_nq_em.png"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

        fig, axis = plt.subplots(figsize=(9, 5.3))
        colors = ["#2667ff", "#3a86ff", "#65a1ff", "#9bc1ff"]
        bars = axis.bar(frame["Paper"], frame["NQ test EM"], color=colors)
        axis.set_title("Open-domain Natural Questions - Test Exact Match")
        axis.set_ylabel("Exact Match (higher is better)")
        axis.set_ylim(0, max(frame["NQ test EM"]) + 10)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.1f", padding=4, fontweight="bold")
        fig.tight_layout()
        fig.savefig(chart_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        best = frame.iloc[0]
        baseline = frame.iloc[-1]
        gain = float(best["NQ test EM"] - baseline["NQ test EM"])
        text = (
            f"{markdown_table(frame)}\n\n"
            f"Best reported result: {best['Paper']} ({best['Reported variant']}) = "
            f"{best['NQ test EM']:.1f} EM. Its absolute gain over "
            f"{baseline['Paper']} is {gain:.1f} points.\n\n"
            "This is an across-paper benchmark comparison, not a reproduction on this laptop. "
            "All rows use the NQ open-domain test split and Exact Match, while corpus chunking and retrieval depth still differ."
        )
        return ToolOutput("compare_results", text, (csv_path, chart_path))

    def generate_survey(self) -> ToolOutput:
        papers = self.facts["papers"]
        ordered = [papers[key] for key in ("realm", "dpr", "rag", "fid")]
        method_lines = "\n".join(
            f"- {paper['short_name']}: {paper['method_summary']}"
            for paper in ordered
        )
        survey = (
            "Mini Survey: Retrieval-Augmented Open-Domain Question Answering\n\n"
            "Problem. Parametric language models can answer from their weights, but updating, tracing, "
            "and covering long-tail knowledge is difficult. The four papers attach an external text memory "
            "to question answering, but improve different parts of the pipeline.\n\n"
            f"Methods.\n{method_lines}\n\n"
            "Progression. REALM established retrieval during pre-training; DPR strengthened dense retrieval; "
            "RAG coupled retrieval with a generative seq2seq model; FiD let the decoder fuse evidence from many passages.\n\n"
            "Comparison. On the shared Natural Questions test benchmark, the reported EM values rise from "
            "40.4 (REALM) to 41.5 (DPR), 44.5 (RAG-Sequence), and 51.4 (FiD-large). "
            "This trend is descriptive, because model capacity and retrieval budgets also change.\n\n"
            "Conclusion. Retrieval quality and multi-passage evidence fusion both matter. A small student system "
            "can reproduce the architectural idea with local embeddings, cosine similarity, FAISS, and a compact generator, "
            "but it must not claim to reproduce the papers' large-scale training results."
        )
        path = self.settings.output_dir / "mini_survey.md"
        path.write_text(survey, encoding="utf-8")
        return ToolOutput("generate_survey", survey, (path,))

    def find_research_gap(self) -> ToolOutput:
        rows: list[str] = []
        for paper_id in ("realm", "dpr", "rag", "fid"):
            paper = self.facts["papers"][paper_id]
            rows.append(f"{paper['short_name']}: " + "; ".join(paper["future_work"]))
        shared = self.facts["shared_research_gaps"]
        text = (
            "Paper-specific future work:\n- "
            + "\n- ".join(rows)
            + "\n\nShared research gaps:\n- "
            + "\n- ".join(shared)
        )
        return ToolOutput("find_research_gap", text)

    def compare_limitations(self) -> ToolOutput:
        records = []
        for paper_id in ("realm", "dpr", "rag", "fid"):
            paper = self.facts["papers"][paper_id]
            records.append(
                {
                    "Paper": paper["short_name"],
                    "Primary limitation": paper["limitations"][0],
                    "Compute / scope caveat": paper["limitations"][1],
                }
            )
        return ToolOutput("compare_limitations", markdown_table(pd.DataFrame(records)))

    def compare_references(self) -> ToolOutput:
        all_counts: dict[str, dict[int, int]] = {}
        for paper in SELECTED_PAPERS:
            all_counts[paper.short_name] = dict(extract_reference_year_counts(paper.path))
        years = sorted({year for counts in all_counts.values() for year in counts})
        frame = pd.DataFrame(
            {
                "Year": years,
                **{
                    paper: [counts.get(year, 0) for year in years]
                    for paper, counts in all_counts.items()
                },
            }
        )
        csv_path = self.settings.output_dir / "reference_year_counts.csv"
        chart_path = self.settings.output_dir / "reference_year_distribution.png"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

        recent = frame[frame["Year"] >= 2010]
        fig, axis = plt.subplots(figsize=(10, 5.5))
        for paper in all_counts:
            axis.plot(recent["Year"], recent[paper], marker="o", linewidth=1.8, label=paper)
        axis.set_title("Heuristic bibliography year distribution (2010 onward)")
        axis.set_xlabel("Publication year")
        axis.set_ylabel("Year mentions in References")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(chart_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        totals = pd.DataFrame(
            [{"Paper": paper, "Detected references/year mentions": sum(counts.values())} for paper, counts in all_counts.items()]
        )
        text = (
            f"{markdown_table(totals)}\n\n"
            "Method note: this optional analysis starts at each References heading and counts four-digit publication years. "
            "It is a transparent regex heuristic, so editions with two years or malformed PDF lines can slightly over/under-count."
        )
        return ToolOutput("compare_references", text, (csv_path, chart_path))

