# 📄 GPT Explore — AI/ML Research Paper Review System

A Python-based pipeline that **collects, processes, indexes, and reviews AI/ML research papers** using semantic embeddings and an LLM (GPT-4o-mini). Given a new research PDF, the system finds related papers from a local corpus and generates a structured comparative literature review.

---

## 🎯 Purpose

This project automates the **literature review process** for AI/ML research. It helps researchers quickly identify:
- **Exact duplicate** papers (via SHA-256 hash)
- **Semantically similar** existing work (via vector similarity search)
- **Research gaps, methodological comparisons, and novelty assessment** (via LLM analysis)

---

## 🗂️ Project Structure

```
gpt_explore/
├── dataset/
│   ├── papers/          # Downloaded PDF files (100 papers)
│   ├── extracted_text/  # Plain text extracted from PDFs
│   ├── chunks/          # Overlapping word chunks (JSON) for each paper
│   ├── sections/        # Structured section-level extraction (JSON)
│   └── metadata/
│       ├── papers.csv              # Master metadata for all papers
│       └── review_reports/         # Generated review reports (JSON + MD)
├── scripts/
│   ├── download_papers.py      # Step 1: Download 100 AI/ML papers
│   ├── extract_text.py         # Step 2: Extract text from PDFs
│   ├── section_extract.py      # Step 3: Identify & extract paper sections
│   ├── create_chunks.py        # Step 4: Split text into overlapping chunks
│   ├── build_vector_index.py   # Step 5: Embed chunks & store in ChromaDB
│   ├── embedding_service.py    # Wrapper around SentenceTransformer
│   ├── vector_service.py       # Wrapper around ChromaDB
│   ├── compare_new_paper.py    # Step 6a: Similarity search (no LLM)
│   ├── review_paper.py         # Step 6b: Full LLM-assisted review
│   ├── generate_hashes.py      # Utility: SHA-256 hash generation
│   └── test_similarity.py      # Utility: Similarity testing
└── storage/
    └── vector_db/              # Persistent ChromaDB vector database
```

---

## ⚙️ Pipeline Overview

```
PDF Papers ──▶ Text Extraction ──▶ Section Parsing ──▶ Chunking
                                                           │
                                                    Embedding (MiniLM)
                                                           │
                                                      ChromaDB Index
                                                           │
                              New Paper PDF ──▶ Query & Retrieve ──▶ LLM Review
```

### Step-by-step

| Step | Script | Description |
|------|--------|-------------|
| 1 | `download_papers.py` | Queries **OpenAlex API** for 100 open-access AI/ML papers (machine learning, deep learning, NLP, CV, RL, GenAI, LLMs) and downloads PDFs |
| 2 | `extract_text.py` | Uses **PyMuPDF (fitz)** to extract raw text from each PDF |
| 3 | `section_extract.py` | Uses regex patterns to detect and extract standard academic sections (Abstract, Introduction, Methodology, Results, Conclusion, etc.) |
| 4 | `create_chunks.py` | Splits text into **700-word overlapping chunks** (100-word overlap) and saves them as JSON per paper |
| 5 | `build_vector_index.py` | Encodes chunks using **`sentence-transformers/all-MiniLM-L6-v2`** and upserts into **ChromaDB** with cosine distance |
| 6a | `compare_new_paper.py` | Given a new PDF, checks for exact duplicates (SHA-256), chunks the text, queries ChromaDB, and outputs a similarity-ranked JSON report |
| 6b | `review_paper.py` | Full LLM workflow: extracts research profile → retrieves candidates → LLM relevance filtering → GPT-4o-mini comparative review → saves JSON + Markdown report |

---

## 🔬 Core Technologies

| Technology | Role |
|-----------|------|
| **OpenAlex API** | Open-access academic paper discovery |
| **PyMuPDF (`fitz` / `pymupdf`)** | PDF text extraction |
| **SentenceTransformers** (`all-MiniLM-L6-v2`) | Semantic text embeddings |
| **ChromaDB** | Persistent local vector database (cosine similarity) |
| **OpenAI API** (`gpt-4o-mini`) | LLM-assisted research profiling & comparative analysis |
| **SHA-256 Hashing** | Exact duplicate detection at byte level |

---

## 📦 Dataset

- **100 AI/ML research papers** collected from OpenAlex (published 2018–present)
- Covers topics: Machine Learning, Deep Learning, Artificial Intelligence, Computer Vision, NLP, Reinforcement Learning, Generative AI, Large Language Models
- Each paper stored with: PDF file, extracted text, section JSON, chunk JSON, and metadata row in `papers.csv`

---

## 🚀 Usage

### 1. Quick Similarity Check (No LLM required)
```bash
cd scripts
python compare_new_paper.py
# Enter the path to your PDF when prompted
```
**Output:** JSON report with top-5 related papers ranked by cosine similarity score.

### 2. Full LLM-Assisted Review
```bash
# Set your OpenAI API key first
$env:OPENAI_API_KEY = "your-api-key"

cd scripts
python review_paper.py path/to/your_paper.pdf
```
**Output:** Both a JSON report and a Markdown report with:
- Research profile (title, problem, methodology, contributions)
- Candidate papers retrieved from the corpus
- LLM relevance screening
- Comparative literature review

### 3. Local-only mode (skip LLM calls)
```bash
python review_paper.py path/to/your_paper.pdf --local-only
```

---

## 📊 Report Output

Both scripts save reports to `dataset/metadata/review_reports/`:

- `review_<timestamp>.json` — Full structured report
- `review_<timestamp>.md` — Human-readable Markdown review
- `latest_review_report.json` / `latest_review_report.md` — Always-current latest report

---

## ⚠️ Limitations

- Similarity scores are **embedding-based retrieval signals**, not plagiarism or originality percentages
- Only the **locally indexed corpus** (100 papers) is searched — broader literature may be missed
- Novelty **cannot be conclusively established** from a small corpus
- Scanned PDFs (image-only) require OCR and are not supported
- LLM review requires a valid `OPENAI_API_KEY` environment variable

---

## 🔧 Key Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `embedding_service.py` |
| LLM model | `gpt-4o-mini` (env: `OPENAI_MODEL`) | `review_paper.py` |
| Chunk size | 700 words | `create_chunks.py` |
| Chunk overlap | 100 words | `create_chunks.py` |
| Top-K retrieval | 40 chunks / 5 papers | `review_paper.py` |
| ChromaDB distance | Cosine | `vector_service.py` |
| Target paper count | 100 | `download_papers.py` |

---

*Generated on 2026-10-12*
