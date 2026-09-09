# extract_experimental_results

Common benchmark: Natural Questions (open-domain QA) (standard test split (minor 3610/3611 preprocessing-version discrepancy is documented))
Common metric: Exact Match (normalized answer-level EM; higher is better)

| Paper | Reported variant | NQ test EM | PDF source             |
| ----- | ---------------- | ---------- | ---------------------- |
| REALM | REALM-CC-News    | 40.4       | REALM PDF p.7, Table 1 |
| DPR   | Single DPR       | 41.5       | DPR PDF p.8, Table 4   |
| RAG   | RAG-Sequence     | 44.5       | RAG PDF p.6, Table 1   |
| FiD   | FiD-large (770M) | 51.4       | FiD PDF p.3, Table 1   |

Additional metrics (kept separate because their definitions/splits differ):

| Paper | Dataset/split          | Metric            | Value | Source               |
| ----- | ---------------------- | ----------------- | ----- | -------------------- |
| DPR   | Natural Questions test | Retrieval Hit@20  | 78.4  | DPR PDF p.5, Table 2 |
| DPR   | Natural Questions test | Retrieval Hit@100 | 85.4  | DPR PDF p.5, Table 2 |
| RAG   | TriviaQA test          | Exact Match       | 56.8  | RAG PDF p.6, Table 1 |
| FiD   | TriviaQA open-domain   | Exact Match       | 67.6  | FiD PDF p.3, Table 1 |
| FiD   | SQuAD Open             | Token F1          | 63.2  | FiD PDF p.3, Table 1 |

Important: retrieval Hit@K/Accuracy@K is not the same quantity as answer Exact Match.

# compare_results

| Paper | Reported variant | NQ test EM | PDF source             |
| ----- | ---------------- | ---------- | ---------------------- |
| FiD   | FiD-large (770M) | 51.4       | FiD PDF p.3, Table 1   |
| RAG   | RAG-Sequence     | 44.5       | RAG PDF p.6, Table 1   |
| DPR   | Single DPR       | 41.5       | DPR PDF p.8, Table 4   |
| REALM | REALM-CC-News    | 40.4       | REALM PDF p.7, Table 1 |

Best reported result: FiD (FiD-large (770M)) = 51.4 EM. Its absolute gain over REALM is 11.0 points.

This is an across-paper benchmark comparison, not a reproduction on this laptop. All rows use the NQ open-domain test split and Exact Match, while corpus chunking and retrieval depth still differ.

# generate_survey

Mini Survey: Retrieval-Augmented Open-Domain Question Answering

Problem. Parametric language models can answer from their weights, but updating, tracing, and covering long-tail knowledge is difficult. The four papers attach an external text memory to question answering, but improve different parts of the pipeline.

Methods.
- REALM: Jointly pre-trains a neural retriever and a masked-language-model knowledge augmenter over an external corpus, then fine-tunes the model for open-domain QA.
- DPR: Trains separate BERT question and passage encoders with in-batch and BM25 hard negatives, indexes 21M passage vectors in FAISS, and feeds retrieved passages to an extractive reader.
- RAG: Combines a DPR retriever with a BART generator and marginalizes over retrieved documents either once per output sequence (RAG-Sequence) or at every output token (RAG-Token).
- FiD: Encodes each question-passage pair independently with T5, concatenates all encoder states, and lets one decoder fuse evidence from as many as 100 passages.

Progression. REALM established retrieval during pre-training; DPR strengthened dense retrieval; RAG coupled retrieval with a generative seq2seq model; FiD let the decoder fuse evidence from many passages.

Comparison. On the shared Natural Questions test benchmark, the reported EM values rise from 40.4 (REALM) to 41.5 (DPR), 44.5 (RAG-Sequence), and 51.4 (FiD-large). This trend is descriptive, because model capacity and retrieval budgets also change.

Conclusion. Retrieval quality and multi-passage evidence fusion both matter. A small student system can reproduce the architectural idea with local embeddings, cosine similarity, FAISS, and a compact generator, but it must not claim to reproduce the papers' large-scale training results.

# find_research_gap

Paper-specific future work:
- REALM: Use larger and more diverse external memories while updating them efficiently.; Improve latent retrieval supervision and joint optimization stability.
- DPR: Improve joint retriever-reader training and reduce the cost of index construction.; Combine dense semantic retrieval with sparse lexical signals for rare exact phrases.
- RAG: Jointly update the document encoder and index during task learning.; Apply retrieval-augmented generation to broader tasks and make provenance more explicit.
- FiD: Train retrieval jointly with the generative reader.; Improve computational efficiency as retrieval depth grows.

Shared research gaps:
- A lightweight way to update dense indexes when the corpus changes, without repeating full-corpus embedding.
- A calibrated decision rule for when retrieval is necessary and when retrieved evidence is insufficient.
- Faithful, page-level provenance: answer quality metrics such as EM do not prove that the generated explanation is supported.
- Controlled comparisons that keep generator size, corpus snapshot, chunking, and top-k retrieval budget fixed.
- Better handling of tables, formulas, multi-column PDF text, and cross-page evidence during real document ingestion.

# compare_references

| Paper | Detected references/year mentions |
| ----- | --------------------------------- |
| REALM | 46                                |
| DPR   | 59                                |
| RAG   | 108                               |
| FiD   | 42                                |

Method note: this optional analysis starts at each References heading and counts four-digit publication years. It is a transparent regex heuristic, so editions with two years or malformed PDF lines can slightly over/under-count.

# compare_limitations

| Paper | Primary limitation                                                                                                     | Compute / scope caveat                                                                                                                                    |
| ----- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| REALM | Index refresh and large-corpus pre-training are computationally expensive and hard to reproduce on student hardware.   | The retriever, corpus snapshot, block size, and top-5 budget differ from later systems, so score gains are not a controlled architecture-only comparison. |
| DPR   | Retrieval Hit@K only checks whether an answer string occurs in a passage; accidental mentions can count as relevant.   | Encoding and indexing about 21 million 768-dimensional passages needs substantial GPU time, RAM, and storage.                                             |
| RAG   | The passage encoder and Wikipedia index are fixed during downstream training, limiting full end-to-end adaptation.     | Generation and marginalization over documents add inference cost, while the paper does not provide user-facing evidence citations.                        |
| FiD   | Training with 100 passages is still heavy (the paper used 64 V100 GPUs), even though passage encoding scales linearly. | The retriever is not trained end-to-end with the generator and the passage encodings interact only in the decoder.                                        |
