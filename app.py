from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from research_assistant import ResearchAssistant, Settings


st.set_page_config(
    page_title="Local Research Assistant",
    page_icon="RA",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading local FAISS index and Ollama clients...")
def load_assistant() -> ResearchAssistant:
    assistant = ResearchAssistant(Settings())
    assistant.store.load()
    return assistant


assistant = load_assistant()

st.title("Local Research Assistant")
st.caption("REALM / DPR / RAG / FiD - local Qwen2.5-1.5B + all-minilm + FAISS")

with st.sidebar:
    st.subheader("Reproducible configuration")
    st.code(
        f"""LLM: {assistant.settings.llm_model}
Embedding: {assistant.settings.embedding_model}
Chunk: {assistant.settings.chunk_words} words
Overlap: {assistant.settings.chunk_overlap} words
Top-k: {assistant.settings.retrieval_top_k}
Similarity: cosine
Vector DB: FAISS"""
    )
    st.info(
        "Self-RAG is supplementary only. DPR replaced it in the numeric core so all four papers share Natural Questions test Exact Match."
    )

chat_tab, compare_tab, evaluation_tab, methods_tab = st.tabs(
    ["Ask papers", "Compare papers", "Evaluation", "Methods"]
)

with chat_tab:
    st.write(
        "Ask a normal paper question, or request a comparison, mini survey, research gaps, references, or limitations. "
        "The lightweight LLM automatically routes explicit operations to functions."
    )
    example = st.selectbox(
        "Example",
        [
            "",
            "What Natural Questions test Exact Match is reported for Single DPR?",
            "How does Fusion-in-Decoder combine evidence from multiple passages?",
            "Compare the results of all papers and draw a chart.",
            "Find research gaps and future work.",
            "What is the weather in Tehran tomorrow?",
        ],
    )
    question = st.text_input("Question", value=example)
    if st.button("Run assistant", type="primary", disabled=not question.strip()):
        with st.spinner("Running local retrieval and generation..."):
            response = assistant.ask(question)
        st.markdown(response.answer)
        left, middle, right = st.columns(3)
        left.metric("Mode", response.mode)
        middle.metric("Router", response.route_source)
        right.metric("Latency", f"{response.elapsed_seconds:.2f} s")
        if response.hits:
            with st.expander("Retrieved chunks"):
                for hit in response.hits:
                    meta = hit.metadata
                    st.markdown(
                        f"**{meta['paper_short_name']} - PDF p.{meta['page']} - cosine {hit.score:.3f}**  \n"
                        f"Section: `{meta['section']}` / record: `{meta.get('record_type', 'pdf_chunk')}`"
                    )
                    st.write(meta["text"])
        for artifact in response.artifacts:
            path = Path(artifact)
            if path.suffix.lower() == ".png" and path.exists():
                st.image(str(path), caption=path.stem)

with compare_tab:
    output = assistant.tools.compare_results()
    st.markdown(output.text)
    chart = assistant.settings.output_dir / "paper_comparison_nq_em.png"
    st.image(str(chart), caption="Common benchmark: Natural Questions test Exact Match")
    st.warning(
        "EM rows are benchmark-comparable, but not a controlled ablation: retrieval depth, model size, and chunking differ. "
        "DPR retrieval Hit@20/100 is kept in a separate table because it is not answer EM."
    )

with evaluation_tab:
    summary_path = assistant.settings.output_dir / "evaluation_summary.json"
    if not summary_path.exists():
        st.info("Run `python scripts/run_evaluation.py` to populate this tab.")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        retrieval = summary["retrieval_and_ood"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top-1 paper accuracy", f"{retrieval['retrieval_top1_paper_accuracy']:.1%}")
        c2.metric("Retrieval F1@3", f"{retrieval['retrieval_f1_at_3']:.1%}")
        c3.metric("OOD F1", f"{retrieval['held_out_ood_detection']['f1']:.1%}")
        c4.metric("Retrieval latency", f"{retrieval['mean_retrieval_latency_ms']:.1f} ms")
        images = [
            ("rag_vs_no_rag.png", "RAG vs no-RAG on the local QA set"),
            ("ood_threshold_curve.png", "Cosine threshold calibration"),
            ("embedding_pca.png", "PCA of paper chunks"),
            ("clustering_diagnostics.png", "K-Means elbow and silhouette"),
            ("chunk_length_analysis.png", "IQR / histogram preprocessing analysis"),
        ]
        for row_start in range(0, len(images), 2):
            columns = st.columns(2)
            for column, (filename, caption) in zip(columns, images[row_start : row_start + 2]):
                path = assistant.settings.output_dir / filename
                if path.exists():
                    column.image(str(path), caption=caption)
        st.json(summary)

with methods_tab:
    facts = assistant.tools.facts
    records = []
    for paper_id in ("realm", "dpr", "rag", "fid"):
        paper = facts["papers"][paper_id]
        records.append(
            {
                "Paper": paper["short_name"],
                "Method": paper["method_summary"],
                "Retriever": paper["retriever"],
                "Reader/generator": paper["generator_or_reader"],
            }
        )
    st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
    st.subheader("Function-calling operations")
    st.code("\n".join(assistant.tools.function_names))
    st.caption(
        "Streamlit is only the optional presentation layer; the data-mining methods remain PDF extraction, chunking, embeddings, cosine similarity, FAISS, RAG, and course evaluation metrics."
    )

