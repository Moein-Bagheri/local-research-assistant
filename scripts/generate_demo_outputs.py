from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from research_assistant import Settings
from research_assistant.research_tools import ResearchTools


def architecture_diagram(output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 7.875))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 8)
    axis.axis("off")
    fig.patch.set_facecolor("#f7f9fc")
    axis.set_facecolor("#f7f9fc")

    boxes = [
        (0.4, 5.0, 1.65, 1.25, "4 papers\nREALM / DPR\nRAG / FiD", "#dbeafe"),
        (2.35, 5.0, 1.8, 1.25, "PyMuPDF\nsection-aware\nextraction", "#e0f2fe"),
        (4.45, 5.0, 1.75, 1.25, "Chunking\n80 words\n15 overlap", "#dcfce7"),
        (6.5, 5.0, 1.8, 1.25, "all-minilm\n384-D local\nembedding", "#fef3c7"),
        (8.6, 5.0, 1.75, 1.25, "FAISS\nnormalized IP\n= cosine", "#ffedd5"),
        (10.65, 5.0, 1.35, 1.25, "Top-3\nchunks", "#fce7f3"),
        (12.3, 5.0, 1.35, 1.25, "Qwen2.5\n1.5B local", "#ede9fe"),
        (10.1, 2.35, 2.25, 1.25, "Automatic tool router\n6 validated functions", "#fee2e2"),
        (12.65, 2.35, 1.05, 1.25, "Answer\n+ pages", "#d1fae5"),
        (7.1, 2.35, 2.25, 1.25, "Cosine threshold\n0.47 calibrated\nOOD fallback", "#cffafe"),
    ]
    for x, y, width, height, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.4, edgecolor="#334155", facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10.5, color="#0f172a", fontweight="bold")

    for start_x, end_x in [(2.05, 2.35), (4.15, 4.45), (6.2, 6.5), (8.3, 8.6), (10.35, 10.65), (12.0, 12.3)]:
        axis.add_patch(FancyArrowPatch((start_x, 5.62), (end_x, 5.62), arrowstyle="-|>", mutation_scale=13, color="#475569", linewidth=1.5))
    axis.add_patch(FancyArrowPatch((12.95, 5.0), (11.25, 3.62), connectionstyle="arc3,rad=0.05", arrowstyle="-|>", mutation_scale=13, color="#b91c1c", linewidth=1.5))
    axis.add_patch(FancyArrowPatch((12.35, 2.98), (12.65, 2.98), arrowstyle="-|>", mutation_scale=13, color="#475569", linewidth=1.5))
    axis.add_patch(FancyArrowPatch((12.95, 5.0), (13.17, 3.62), arrowstyle="-|>", mutation_scale=13, color="#475569", linewidth=1.5))
    axis.add_patch(FancyArrowPatch((8.22, 3.62), (8.22, 5.0), arrowstyle="-|>", mutation_scale=13, color="#0891b2", linewidth=1.5))
    axis.text(7.0, 7.25, "Local research-assistant architecture", ha="center", fontsize=22, fontweight="bold", color="#0f172a")
    axis.text(7.0, 6.8, "Course concepts: preprocessing, embeddings, cosine distance, FAISS, RAG, function calling, Accuracy / Precision / F1", ha="center", fontsize=11.5, color="#475569")
    axis.text(7.0, 0.85, "No paid API. Table values are human-verified records with PDF page/table provenance; raw PDF chunks remain the primary knowledge base.", ha="center", fontsize=11, color="#334155")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def dpr_retrieval_chart(output_path: Path) -> None:
    labels = ["Hit@20", "Hit@100"]
    bm25 = [59.1, 73.7]
    dpr = [78.4, 85.4]
    x = [0, 1]
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    left = axis.bar([value - width / 2 for value in x], bm25, width, label="BM25", color="#94a3b8")
    right = axis.bar([value + width / 2 for value in x], dpr, width, label="Single DPR", color="#2563eb")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("NQ test retrieval accuracy (%)")
    axis.set_title("Retrieval metric kept separate from answer Exact Match")
    axis.bar_label(left, fmt="%.1f", padding=3)
    axis.bar_label(right, fmt="%.1f", padding=3)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    settings = Settings()
    settings.ensure_directories()
    tools = ResearchTools(settings)
    outputs = [
        tools.extract_experimental_results(),
        tools.compare_results(),
        tools.generate_survey(),
        tools.find_research_gap(),
        tools.compare_references(),
        tools.compare_limitations(),
    ]
    combined = []
    for output in outputs:
        combined.append(f"# {output.name}\n\n{output.text}\n")
    (settings.output_dir / "demo_outputs.md").write_text("\n".join(combined), encoding="utf-8")
    architecture_diagram(settings.output_dir / "architecture.png")
    dpr_retrieval_chart(settings.output_dir / "dpr_retrieval_metrics.png")
    print("Generated:")
    for path in sorted(settings.output_dir.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()

