from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, silhouette_score

from research_assistant import ResearchAssistant, Settings


def normalize_answer(text: str) -> str:
    text = text.casefold().replace("–", "-").replace("—", "-")
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    shared = sum(common.values())
    if not shared:
        return 0.0
    precision = shared / len(predicted)
    recall = shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def alias_contained(answer: str, aliases: list[str]) -> bool:
    normalized = normalize_answer(answer)
    return any(normalize_answer(alias) in normalized for alias in aliases)


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def retrieve_scores(assistant: ResearchAssistant, questions: list[dict], label: int) -> list[dict]:
    records = []
    for item in questions:
        started = time.perf_counter()
        hits = assistant.store.search(item["question"], top_k=5)
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "label_in_domain": label,
                "top_score": hits[0].score if hits else -1.0,
                "top1_paper": hits[0].metadata["paper_id"] if hits else "",
                "top3_papers": ",".join(hit.metadata["paper_id"] for hit in hits[:3]),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "relevant_papers": ",".join(item.get("relevant_papers", [])),
            }
        )
    return records


def choose_threshold(frame: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    candidates = np.linspace(float(frame.top_score.min()), float(frame.top_score.max()), 200)
    rows = []
    y_true = frame.label_in_domain.astype(int).tolist()
    for threshold in candidates:
        predicted = (frame.top_score >= threshold).astype(int).tolist()
        metrics = binary_metrics(y_true, predicted)
        rows.append({"threshold": float(threshold), **metrics})
    curve = pd.DataFrame(rows)
    best = curve.sort_values(["f1", "accuracy", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(best.threshold), curve


def evaluate_retrieval(assistant: ResearchAssistant, data: dict, output_dir: Path) -> tuple[dict, float]:
    in_items = data["retrieval_questions"]
    out_items = data["out_of_domain_questions"]
    records = retrieve_scores(assistant, in_items, 1) + retrieve_scores(assistant, out_items, 0)
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "retrieval_query_results.csv", index=False, encoding="utf-8-sig")

    # Deterministic alternating split: threshold is tuned on calibration only and reported on held-out rows.
    calibration = pd.concat([frame[frame.label_in_domain == 1].iloc[::2], frame[frame.label_in_domain == 0].iloc[::2]])
    test = frame.drop(calibration.index)
    threshold, curve = choose_threshold(calibration)
    test_pred = (test.top_score >= threshold).astype(int).tolist()
    ood_metrics = binary_metrics(test.label_in_domain.astype(int).tolist(), test_pred)
    configured_pred = (test.top_score >= assistant.settings.relevance_threshold).astype(int).tolist()
    configured_metrics = binary_metrics(test.label_in_domain.astype(int).tolist(), configured_pred)

    positive = frame[frame.label_in_domain == 1].copy()
    top1_correct = []
    hit3 = []
    precision3 = []
    recall3 = []
    f1_values = []
    for _, row in positive.iterrows():
        relevant = set(filter(None, row.relevant_papers.split(",")))
        predicted = row.top3_papers.split(",")
        top1_correct.append(int(row.top1_paper in relevant))
        hit3.append(int(bool(relevant.intersection(predicted))))
        true_chunks = sum(paper in relevant for paper in predicted)
        precision = true_chunks / max(len(predicted), 1)
        recall = len(relevant.intersection(predicted)) / max(len(relevant), 1)
        precision3.append(precision)
        recall3.append(recall)
        f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))

    fig, axis = plt.subplots(figsize=(8.7, 5.2))
    axis.plot(curve.threshold, curve.f1, label="Calibration F1", linewidth=2)
    axis.plot(curve.threshold, curve.accuracy, label="Calibration accuracy", linewidth=2)
    axis.axvline(threshold, color="#e63946", linestyle="--", label=f"Chosen threshold={threshold:.3f}")
    axis.set_xlabel("Top cosine-similarity threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_title("In-domain vs out-of-domain threshold calibration")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ood_threshold_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "retrieval_top1_paper_accuracy": round(float(np.mean(top1_correct)), 4),
        "retrieval_hit_at_3": round(float(np.mean(hit3)), 4),
        "retrieval_precision_at_3": round(float(np.mean(precision3)), 4),
        "retrieval_recall_at_3": round(float(np.mean(recall3)), 4),
        "retrieval_f1_at_3": round(float(np.mean(f1_values)), 4),
        "mean_retrieval_latency_ms": round(float(frame.latency_ms.mean()), 2),
        "calibrated_threshold": round(threshold, 4),
        "held_out_ood_detection": ood_metrics,
        "held_out_ood_with_configured_threshold": configured_metrics,
        "calibration_rows": int(len(calibration)),
        "held_out_rows": int(len(test)),
    }
    return metrics, threshold


def evaluate_embeddings(settings: Settings, output_dir: Path) -> dict:
    embeddings = np.load(settings.index_dir / "embeddings.npy")
    metadata = json.loads((settings.index_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = np.asarray([item["paper_id"] for item in metadata])
    word_counts = np.asarray([item["word_count"] for item in metadata])

    pca = PCA(n_components=2, random_state=settings.random_seed)
    coordinates = pca.fit_transform(embeddings)
    paper_order = ["realm", "dpr", "rag", "fid"]
    colors = {"realm": "#2a9d8f", "dpr": "#e9c46a", "rag": "#f4a261", "fid": "#e76f51"}
    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    for paper in paper_order:
        mask = labels == paper
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=18, alpha=0.68, label=paper.upper(), c=colors[paper])
    axis.set_title("PCA projection of chunk embeddings")
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    axis.grid(alpha=0.18)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    k_values = list(range(2, 9))
    inertias: list[float] = []
    silhouettes: list[float] = []
    fitted: dict[int, np.ndarray] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=settings.random_seed)
        cluster_ids = model.fit_predict(embeddings)
        fitted[k] = cluster_ids
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(embeddings, cluster_ids, metric="euclidean")))
    best_k = k_values[int(np.argmax(silhouettes))]
    cluster_ids = fitted[best_k]
    purity_matches = 0
    for cluster in range(best_k):
        members = labels[cluster_ids == cluster]
        if len(members):
            purity_matches += Counter(members).most_common(1)[0][1]
    purity = purity_matches / len(labels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(k_values, inertias, marker="o")
    axes[0].set_title("K-Means elbow curve")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Within-cluster sum of squares")
    axes[0].grid(alpha=0.25)
    axes[1].plot(k_values, silhouettes, marker="o", color="#e76f51")
    axes[1].axvline(best_k, linestyle="--", color="#264653", label=f"best K={best_k}")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "clustering_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    q1, q3 = np.percentile(word_counts, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = int(((word_counts < lower) | (word_counts > upper)).sum())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].hist(word_counts, bins=20, color="#3a86ff", edgecolor="white")
    axes[0].axvline(q1, color="#e63946", linestyle="--", label="Q1/Q3")
    axes[0].axvline(q3, color="#e63946", linestyle="--")
    axes[0].set_title("Chunk-length histogram")
    axes[0].set_xlabel("Words per chunk")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[1].boxplot(word_counts, vert=True)
    axes[1].set_title("Chunk-length IQR boxplot")
    axes[1].set_ylabel("Words per chunk")
    fig.tight_layout()
    fig.savefig(output_dir / "chunk_length_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "chunk_count": int(len(metadata)),
        "embedding_dimension": int(embeddings.shape[1]),
        "pca_explained_variance_2d": round(float(pca.explained_variance_ratio_.sum()), 4),
        "kmeans_best_k": int(best_k),
        "kmeans_best_silhouette": round(float(max(silhouettes)), 4),
        "cluster_purity_by_paper": round(float(purity), 4),
        "chunk_length_mean": round(float(word_counts.mean()), 2),
        "chunk_length_q1": round(float(q1), 2),
        "chunk_length_q3": round(float(q3), 2),
        "chunk_length_iqr_outliers": outlier_count,
    }


def evaluate_routing(assistant: ResearchAssistant, data: dict, output_dir: Path) -> dict:
    rows = []
    for item in data["tool_routing_questions"]:
        started = time.perf_counter()
        predicted, _, source = assistant.route(item["question"])
        rows.append(
            {
                "question": item["question"],
                "expected_tool": item["tool"],
                "predicted_tool": predicted or "none",
                "route_source": source,
                "correct": int(predicted == item["tool"]),
                "latency_seconds": time.perf_counter() - started,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "tool_routing_results.csv", index=False, encoding="utf-8-sig")
    return {
        "accuracy": round(float(frame.correct.mean()), 4),
        "mean_latency_seconds": round(float(frame.latency_seconds.mean()), 3),
        "llm_tool_call_count": int((frame.route_source == "llm_tool_call").sum()),
        "fallback_count": int((frame.route_source == "deterministic_fallback").sum()),
    }


def short_generation_answer(assistant: ResearchAssistant, question: str, use_rag: bool) -> tuple[str, float]:
    context = ""
    if use_rag:
        hits = assistant.store.search(question, top_k=5)
        context = assistant._context_from_hits(hits)
    system = "Return only the shortest factual answer. Do not explain."
    if use_rag:
        user = f"Use only these paper excerpts.\n\n{context}\n\nQuestion: {question}\nShort answer:"
    else:
        user = f"Question: {question}\nShort answer:"
    started = time.perf_counter()
    result = assistant.client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=60,
        temperature=0.0,
    )
    return result.content, time.perf_counter() - started


def evaluate_generation(assistant: ResearchAssistant, data: dict, output_dir: Path) -> dict:
    rows = []
    for item in data["generation_questions"]:
        for mode, use_rag in (("no_rag", False), ("rag", True)):
            answer, elapsed = short_generation_answer(assistant, item["question"], use_rag)
            best_f1 = max(token_f1(answer, alias) for alias in item["aliases"])
            rows.append(
                {
                    "mode": mode,
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "answer": answer,
                    "answer_containment": int(alias_contained(answer, item["aliases"])),
                    "token_f1": best_f1,
                    "latency_seconds": elapsed,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "generation_results.csv", index=False, encoding="utf-8-sig")
    summary: dict[str, dict] = {}
    for mode, group in frame.groupby("mode"):
        summary[mode] = {
            "answer_containment_accuracy": round(float(group.answer_containment.mean()), 4),
            "mean_token_f1": round(float(group.token_f1.mean()), 4),
            "mean_latency_seconds": round(float(group.latency_seconds.mean()), 3),
        }

    plot_frame = pd.DataFrame(summary).T
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    plot_frame[["answer_containment_accuracy", "mean_token_f1"]].plot.bar(ax=axis, color=["#3a86ff", "#ff9f1c"])
    axis.set_title("Lightweight Qwen2.5-1.5B: no-RAG vs RAG")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_xticklabels(["No RAG", "RAG"], rotation=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(["Answer-containment accuracy", "Token F1"])
    fig.tight_layout()
    fig.savefig(output_dir / "rag_vs_no_rag.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval, OOD fallback, routing, clustering, and generation.")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-routing", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    assistant = ResearchAssistant(settings)
    assistant.client.ensure_models()
    assistant.store.load()
    data = json.loads(settings.evaluation_path.read_text(encoding="utf-8"))

    summary: dict[str, object] = {
        "configuration": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chunk_words": settings.chunk_words,
            "chunk_overlap": settings.chunk_overlap,
            "online_retrieval_top_k": settings.retrieval_top_k,
            "relevance_threshold": settings.relevance_threshold,
            "vector_database": "FAISS IndexFlatIP",
            "similarity": "cosine (L2 normalization + inner product)",
            "seed": settings.random_seed,
        }
    }
    summary["embedding_and_clustering"] = evaluate_embeddings(settings, settings.output_dir)
    retrieval, threshold = evaluate_retrieval(assistant, data, settings.output_dir)
    summary["retrieval_and_ood"] = retrieval
    if not args.skip_routing:
        summary["function_routing"] = evaluate_routing(assistant, data, settings.output_dir)
    if not args.skip_generation:
        summary["generation"] = evaluate_generation(assistant, data, settings.output_dir)

    summary_path = settings.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSuggested relevance threshold from calibration split: {threshold:.4f}")


if __name__ == "__main__":
    main()
