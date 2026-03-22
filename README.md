# FIA Infringements — NLP Summarization Project

A **natural language processing (NLP) and text mining** project that builds and evaluates automatic summarization pipelines over Mercedes F1 infringement documents from the FIA (2020–2024). The focus is on experimenting with extractive and abstractive summarization models and comparing them to LLM-based ground truth summaries using standard evaluation metrics.

---

## Project Overview

This project applies NLP techniques to legal/regulatory documents: FIA infringement notices for Mercedes-AMG Petronas F1 Team. It investigates how well different summarization approaches capture and compress the content of these texts.

### NLP and Experimentation

The work is structured as an experimental pipeline:

1. **Data extraction & preprocessing** – PDF → text, then cleaning and standardization.
2. **Summarization models** – Extractive (TextRank, LexRank) and abstractive (DistilBART, Pegasus).
3. **Versioned experiments** – Original and V2 configurations with tuned parameters to compare performance.
4. **Evaluation** – ROUGE scores, intrinsic metrics (compression, redundancy, key-entity overlap), and comparisons against LLM-generated ground truth (GPT-4o-mini, Groq-Llama).

The dataset spans **82 documents** (2020–2024), covering topics such as track limits, collisions, pit lane infringements, and technical breaches.

---

## Findings

### Table 3: ROUGE Evaluation (vs. Ground Truth)

Summarization models were evaluated against two ground truth references (GPT-4o-mini and Groq-Llama) using ROUGE metrics. Target ranges: ROUGE-1 (0.3–0.5), ROUGE-2 (0.05–0.20), ROUGE-L (0.2–0.4).

**Ground Truth: gpt-4o-mini**

| Method        | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------------|---------|---------|---------|
| TextRank      | 0.209   | 0.034   | 0.124   |
| TextRank_v2   | 0.209   | 0.025   | 0.138   |
| LexRank       | 0.248   | 0.038   | 0.137   |
| LexRank_v2    | 0.257   | 0.004   | 0.128   |
| **DistilBart** | **0.330** | 0.035 | **0.165** |
| DistilBart_v2 | 0.302   | 0.029   | 0.140   |
| Pegasus       | 0.293   | 0.012   | 0.145   |
| **Pegasus_v2** | 0.268  | **0.047** | 0.134 |

**Ground Truth: groq-llama**

| Method        | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------------|---------|---------|---------|
| TextRank      | 0.181   | 0.018   | 0.133   |
| TextRank_v2   | 0.188   | 0.013   | 0.131   |
| LexRank       | 0.227   | 0.013   | 0.133   |
| LexRank_v2    | 0.256   | 0.009   | 0.130   |
| **DistilBart** | **0.302** | 0.021 | **0.143** |
| DistilBart_v2 | 0.260   | 0.017   | 0.119   |
| Pegasus       | 0.265   | 0.012   | 0.118   |
| **Pegasus_v2** | 0.262  | **0.038** | 0.137 |

**Key takeaways:**
- **DistilBart** is the top performer for ROUGE-1 and ROUGE-L across both ground truths.
- **Pegasus_v2** achieves the highest ROUGE-2 scores (0.047 and 0.038).
- Scores are generally higher against the gpt-4o-mini ground truth than groq-llama.
- Most models do not reach the target ranges, especially for ROUGE-2 and ROUGE-L; DistilBart reaches the lower end for ROUGE-1 (~0.3).

---

### Table 4: Content-Quality Metrics

| Model         | Key-Entity Overlap | Compression Ratio | Redundancy Check | FactCC  |
|---------------|--------------------|-------------------|------------------|---------|
| TextRank      | 100%               | 99.02%            | 60.62            | 0.8842  |
| TextRank_v2   | 69.57%             | 97.78%            | 39.78            | 0.99    |
| LexRank       | 100%               | 99.37%            | 7.72             | 0.7537  |
| LexRank_v2    | 80.00%             | 97.79%            | 6.90             | 0.995   |
| **DistilBart** | **89.47%**         | **98.72%**        | **6.32**         | **0.9983** |
| DistilBart_v2 | 71.43%             | 96.79%            | 5.39             | 0.988   |
| Pegasus       | 81.82%             | 97.75%            | 5.43             | 0.9318  |
| Pegasus_v2    | 71.43%             | 98.85%            | 8.25             | 0.997   |

**Metric definitions:**
- **Key-Entity Overlap:** Share of important entities retained in the summary.
- **Compression Ratio:** Degree of text reduction (higher % = more compressed); all models exceed 96%.
- **Redundancy Check:** Lower values mean less repetition; TextRank shows very high redundancy (60.62).
- **FactCC:** Factual consistency of the summary (closer to 1.0 is better).

**Key takeaways:**
- **DistilBart** balances high Key-Entity Overlap (89.47%), strong compression (98.72%), low redundancy (6.32), and excellent FactCC (0.9983).
- Extractive models (TextRank, LexRank) reach 100% Key-Entity Overlap in their base versions but suffer from high redundancy (TextRank) or lower FactCC (LexRank).
- V2 variants improve FactCC for TextRank and LexRank but reduce Key-Entity Overlap.

---

## Codebase Structure

```
fia-infringements/
├── Documents/                    # Source PDFs (by year: 2020–2024)
│   └── {year}_inf_profile/
├── pre_proc_op/                  # Preprocessed text (no_header_, no_comp_time_, no_footer_)
│   └── {year}/
├── ground_truth/                 # LLM-generated reference summaries
│   ├── gpt-4o-mini.txt
│   └── groq-llama.txt
├── OP_SUMMARIES/                 # Consolidated summaries (cleaned, for evaluation)
├── textrank_results/             # TextRank outputs (original)
├── textrank_results_v2/          # TextRank V2 outputs
├── lexrank_results/              # LexRank outputs (original)
├── lexrank_results_v2/           # LexRank V2 outputs
├── distilbart_results/           # DistilBART outputs (original)
├── distilbart_results_v2/        # DistilBART V2 outputs
│
├── eval/                         # Evaluation notebooks and outputs
│   ├── eval_textrank_lexrank.ipynb   # ROUGE evaluation vs ground truth
│   ├── eval_metrics.ipynb            # Intrinsic metrics (KE overlap, compression, redundancy)
│   ├── eval_ground_truth.ipynb       # Ground truth document analysis
│   ├── rouge_evaluation_results.csv
│   ├── rouge_v2_evaluation_results.csv
│   ├── ground_truth_evaluation_results.csv
│   └── intrinsic_metrics_results.csv
│
├── 3_filter_merc_to_txt.ipynb    # PDF → TXT (Mercedes-only, strict filter)
├── pre_proc_merc.ipynb           # Preprocessing pipeline
├── op_formatting.ipynb           # Extract and clean overall summaries → OP_SUMMARIES
├── textrank_merc.ipynb           # TextRank summarization (original)
├── textrank_merc_v2.ipynb        # TextRank V2 (tuned parameters)
├── lexrank_merc.ipynb            # LexRank summarization (original)
├── lexrank_merc_v2.ipynb         # LexRank V2 (tuned parameters)
├── distilbart_merc.ipynb         # DistilBART abstractive summarization (original)
├── distilbart_merc_v2.ipynb      # DistilBART V2 (tuned parameters)
└── desc_analysis_merc.ipynb      # Descriptive analysis and visualizations
```

---

## Preprocessing Pipeline

1. **Header removal** – Content before the `"No / Driver"` keyword is removed.
2. **Competitor/time removal** – Patterns like `Competitor [Team] Time [HH:MM]` are stripped.
3. **Footer removal** – Appeal and signature sections (from `"Competitors are reminded..."` onward) are removed.

Articles (a, an, the) are preserved to maintain grammatical coherence for summarization.

---

## Summarization Models

| Model      | Type        | Description |
|-----------|-------------|-------------|
| **TextRank** | Extractive | Graph-based ranking of sentences using TF-IDF and cosine similarity. |
| **LexRank**  | Extractive | Similar graph-based approach with centrality-based sentence selection. |
| **DistilBART** | Abstractive | Transformer-based model (`sshleifer/distilbart-cnn-12-6`, ~260MB). |
| **Pegasus**   | Abstractive | Large transformer model for abstractive summarization. |

V2 notebooks use modified parameters (e.g. similarity thresholds, sentence counts) to test different summarization behaviors.

---

## Requirements & Usage

### Dependencies

- Python 3.x
- `PyPDF2` – PDF extraction
- `pandas`, `numpy` – Data handling
- `nltk` – Tokenization, stopwords, stemming
- `scikit-learn` – TF-IDF, cosine similarity
- `networkx` – TextRank/LexRank graph construction
- `transformers`, `torch` – DistilBART / Pegasus
- `rouge-score` – ROUGE evaluation
- `matplotlib`, `seaborn` – Visualizations

### Recommended Workflow

1. Run **`3_filter_merc_to_txt.ipynb`** to extract Mercedes-specific documents from PDFs.
2. Run **`pre_proc_merc.ipynb`** to preprocess and produce `no_footer_` files.
3. Run summarization notebooks (`textrank_merc.ipynb`, `lexrank_merc.ipynb`, etc.).
4. Run **`op_formatting.ipynb`** to build `OP_SUMMARIES/`.
5. Run notebooks in **`eval/`** to compute metrics and generate CSVs.

---

## License

See repository for license information.
