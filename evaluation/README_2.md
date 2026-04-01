# Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2504.19413)
[![Website](https://img.shields.io/badge/Website-Project-blue)](https://mem0.ai/research)

This repository contains the code and dataset for our paper: **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory**.

## Overview

This project evaluates Mem0 and compares it with different memory and retrieval techniques for AI systems using the [LOCOMO](https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-?usp=drive_link) dataset. Techniques tested include Mem0 (with and without graph memory), RAG, LangMem, Zep, OpenAI memory, and several established LOCOMO benchmarks (ReadAgent, MemoryBank, MemGPT, A-Mem).

---

## Setup Guide: Running Mem0 Evaluation Experiments

This guide walks through running the **Mem0 with graph memory** (Mem0+) evaluation pipeline end-to-end. The pipeline has four stages: **ingest memories**, **search & answer questions**, **evaluate answers**, and **generate scores**.

### Prerequisites

- Python 3.10+
- A running **YugabyteDB** instance with [Apache AGE](https://age.apache.org/) support (used as the vector store and graph store)
- An **OpenAI API key** with access to `gpt-4o` and `text-embedding-3-small`

### 1. Install Dependencies

```bash
pip install openai python-dotenv jinja2 tqdm pandas nltk rouge-score bert-score sentence-transformers meko-mem0
```

### 2. Download the Dataset

Download the LOCOMO dataset from Google Drive:

[Download LOCOMO Dataset](https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-?usp=drive_link)

Place the files in the `datasets/` directory:

```
evaluation/
  datasets/
    locomo10.json
    locomo10_rag.json    # (only needed for RAG experiments)
```

### 3. Configure Environment Variables

Export the following environment variables (or add them to a `.env` file in the `evaluation/` directory):

```bash
export OPENAI_API_KEY='<your-openai-api-key>'
export MODEL='gpt-4o'
```

### 4. Configure Database Connection

Update the connection config in both `src/memzero/add.py` and `src/memzero/search.py`. Replace the placeholder values with your YugabyteDB credentials:

```python
"vector_store": {
    "provider": "pgvector",
    "config": {
        "host": "<YUGABYTE_HOST>",
        "port": <YUGABYTE_PORT>,
        "dbname": "<YUGABYTE_DB_NAME>",
        "user": "<YUGABYTE_USER>",
        "password": "<YUGABYTE_PASSWORD>",
        "embedding_model_dims": 1536,
    },
},
"graph_store": {
    "provider": "apache_age",
    "config": {
        "host": "<YUGABYTE_HOST>",
        "port": <YUGABYTE_PORT>,
        "database": "<YUGABYTE_DB_NAME>",
        "user": "<YUGABYTE_USER>",
        "password": "<YUGABYTE_PASSWORD>",
        "graph_name": "<YUGABYTE_GRAPH_NAME>",
    },
    "threshold": 0.7,
},
```

Also replace `<YOUR_OPENAI_API_KEY>` in the `llm` and `embedder` config sections of both files with your actual key, or refactor them to read from the environment.

### 5. Set the Dataset Path

In `run_experiments.py`, replace `<dataset_path>` with the absolute path to your downloaded `locomo10.json` file. There are two occurrences to update (lines 41 and 49):

```python
# Line 41 – used during memory ingestion (add)
memory_manager = MemoryADD(data_path="<dataset_path>", is_graph=args.is_graph)

# Line 49 – used during search & answer
memory_searcher.process_data_file("<dataset_path>")
```

For example, if you placed the dataset in `evaluation/datasets/`:

```python
memory_manager = MemoryADD(data_path="/absolute/path/to/evaluation/datasets/locomo10.json", is_graph=args.is_graph)
# ...
memory_searcher.process_data_file("/absolute/path/to/evaluation/datasets/locomo10.json")
```

### 6. Run the Experiment Pipeline

All commands below should be run from the `evaluation/` directory.

#### Step 1 -- Ingest Memories

Processes all conversations from the LOCOMO dataset, extracts memories for each speaker, and stores them in the vector + graph store.

```bash
python3 run_experiments.py --technique_type mem0 --method add --is_graph
```

This calls `MemoryADD.process_all_conversations()`, which iterates through each conversation, adds memories for both speakers (in parallel threads), and stores them with timestamps as metadata.

#### Step 2 -- Search & Answer Questions

For each question in the dataset, retrieves relevant memories (semantic + graph relations) and generates an answer using the configured LLM.

```bash
python3 run_experiments.py --technique_type mem0 --method search --is_graph
```

Results are written incrementally to:

```
results/mem0_results_top_30_filter_False_graph_True.json
```

The output file name is derived from the default `--top_k 30`, `--filter_memories False`, and `--is_graph True` flags.

#### Step 3 -- Evaluate Answers

Scores each generated answer against the ground truth using BLEU, token-level F1, and an LLM judge (`gpt-4o-mini`). Category 5 questions are skipped.

```bash
python3 evals.py \
  --input_file results/mem0_results_top_30_filter_False_graph_True.json \
  --output_file results/evals_result.json
```

#### Step 4 -- Generate Final Scores

Aggregates per-category and overall means for BLEU, F1, and LLM scores.

```bash
python3 generate_scores.py
```

> **Note:** `generate_scores.py` reads from a hardcoded path (`results/evals_result.json`). Make sure the `--output_file` in the previous step matches this path.

Expected output:

```
Mean Scores Per Category:
         bleu_score  f1_score  llm_score  count
category
1           0.xxxx    0.xxxx     0.xxxx     xx
2           0.xxxx    0.xxxx     0.xxxx     xx
3           0.xxxx    0.xxxx     0.xxxx     xx

Overall Mean Scores:
bleu_score    0.xxxx
f1_score      0.xxxx
llm_score     0.xxxx
```

### Running Without Graph Memory (Mem0 only)

To run the same pipeline without the graph store (vector-only retrieval), omit the `--is_graph` flag:

```bash
python3 run_experiments.py --technique_type mem0 --method add
python3 run_experiments.py --technique_type mem0 --method search
```

The results file will be named `results/mem0_results_top_30_filter_False_graph_False.json`. Update the `--input_file` argument to `evals.py` accordingly.

### Using Makefile Shortcuts

The project includes a Makefile with predefined targets:

```bash
make run-mem0-plus-add      # Ingest with graph
make run-mem0-plus-search   # Search with graph
make run-mem0-add           # Ingest without graph
make run-mem0-search        # Search without graph
```

---

## Command-line Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--technique_type` | Memory technique (`mem0`, `rag`, `langmem`, `zep`, `openai`) | `mem0` |
| `--method` | Method to use (`add`, `search`) | `add` |
| `--chunk_size` | Chunk size for RAG processing | `1000` |
| `--top_k` | Number of top memories to retrieve | `30` |
| `--filter_memories` | Enable memory filtering | `False` |
| `--is_graph` | Enable graph-based memory (Mem0+) | `False` |
| `--num_chunks` | Number of chunks for RAG | `1` |
| `--output_folder` | Directory for result files | `results/` |

## Project Structure

```
evaluation/
├── src/
│   ├── memzero/          # Mem0 add & search implementations
│   ├── openai/           # OpenAI memory implementation
│   ├── zep/              # Zep memory implementation
│   ├── rag.py            # RAG technique
│   ├── langmem.py        # LangMem technique
│   └── utils.py          # Constants (technique/method lists)
├── metrics/
│   ├── llm_judge.py      # LLM-based answer evaluation
│   └── utils.py          # BLEU, F1, ROUGE, BERTScore, METEOR, SBERT
├── datasets/             # LOCOMO dataset files
├── results/              # Experiment output and evaluation results
├── run_experiments.py    # Main experiment runner
├── evals.py              # Answer evaluation pipeline
├── generate_scores.py    # Score aggregation
├── prompts.py            # LLM prompts (with/without graph context)
└── Makefile              # Shortcut targets
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram overlap between generated and ground-truth answers |
| **F1** | Token-level precision/recall harmonic mean |
| **LLM Score** | Binary (0/1) correctness judgment by `gpt-4o-mini` |
| **Latency** | Time for memory search + answer generation |
| **Token Consumption** | Tokens used to generate the final answer |

## Citation

```bibtex
@article{mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}
```

## License

[MIT License](LICENSE)

## Contributors

- [Prateek Chhikara](https://github.com/prateekchhikara)
- [Dev Khant](https://github.com/Dev-Khant)
- [Saket Aryan](https://github.com/whysosaket)
- [Taranjeet Singh](https://github.com/taranjeet)
- [Deshraj Yadav](https://github.com/deshraj)

