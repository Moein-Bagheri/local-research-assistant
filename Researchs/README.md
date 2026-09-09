# Research papers

Download the PDFs from the paper pages below. Save them directly in this directory using the exact filenames expected by `src/research_assistant/config.py`.

| Paper | Source | Required local filename |
| --- | --- | --- |
| REALM | [Guu et al., ICML 2020](https://proceedings.mlr.press/v119/guu20a.html) | `REALM- Retrieval-Augmented Language Model Pre-Training.pdf` |
| DPR | [Karpukhin et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.550/) | `Dense_Passage_Retrieval_for_Open_Domain_Question_Answering.pdf` |
| RAG | [Lewis et al., 2020](https://arxiv.org/abs/2005.11401) | `Retrieval_Augmented_Generation_for_Knowledge_Intensive_NLP_Tasks.pdf` |
| FiD | [Izacard and Grave, EACL 2021](https://aclanthology.org/2021.eacl-main.74/) | `Leveraging_Passage_Retrieval_with_Generative_Models_for_Open_Domain.pdf` |

Then run `setup.ps1` from the project root, or rebuild the index using `scripts/build_index.py` if the environment already exists.

The saved curated facts contain page/table references from the PDFs used for the original project. Paper revisions can have different pagination. Keep the original PDF versions when reproducing those exact references. PDF files and generated indexes are stored locally and excluded from version control.
