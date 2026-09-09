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