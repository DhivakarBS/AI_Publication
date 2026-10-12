# ResearchAI — Project Review Executive Summary

**Presentation Target:** Final-Year Project Review (Thursday, 15 October 2026)
**Document Status:** Final Verified Technical Summary
**Audit Scope:** Read-Only Verification of Production Code, Tests, and Benchmark Gate

---

## 1. Project Objective

**ResearchAI** automates the literature review and comparative analysis process for academic research papers in artificial intelligence and machine learning.

Given an input paper PDF, the system extracts structured technical facts, retrieves related literature from an indexed local corpus (100 papers), filters candidate citations, and generates an evidence-grounded comparative review that highlights shared methodologies, baseline comparisons, dataset compatibilities, and quantitative trade-offs.

---

## 2. Major Enhancements Achieved

Since the initial baseline, the system has undergone five major architectural enhancements:

1. **Deterministic Technical Fact Extraction:** Added Stage 1 extraction for algorithm families and variants, explicit experimental metrics, standard datasets, hardware, software, and hyperparameters with 1-based physical page provenance.
2. **Extraction Quality Engineering:** Built robust filters rejecting table-scrap floating decimals, mathematical formula constants (e.g., F-score constant 2), generic dataset quantifier phrases (e.g., "several datasets"), and bibliography citations. Disambiguated learning rate (LR) from Logistic Regression and enforced case-sensitive Vision Transformer (ViT) matching.
3. **5-Category Algorithm-Usage Classification:** Designed a contextual classification engine distinguishing whether an algorithm is the paper's *proposed method*, an *experimental baseline*, *background / related work*, *references only*, or *uncertain*.
4. **Method-Grounded Comparative Review:** Fixed the P0 flaw where algorithm overlap falsely implied shared methodology. Enforced that shared method claims require active usage in both papers, while imposing strict comparability rules preventing invalid metric rankings (e.g., Accuracy vs F1, or across different datasets).
5. **Rigorous Benchmark Gate & Curation:** Created a 73-example verified gold benchmark with 100% genuine source passages, verified physical PDF page numbers, and isolated unresolvable records. Expanded test coverage to **47 passing unit and benchmark tests**.

---

## 3. Current Classifier Capabilities & Taxonomy

The production algorithm-usage classifier is **100% deterministic, rule-based, and offline-capable**:

| Usage Category | Definition | Contextual Cues | Role in Comparative Review |
|---|---|---|---|
| **`proposed_method`** | Novel contribution or core architecture | `"we propose"`, `"our model"`, methodology section | Core method overlap claims |
| **`experiment_or_baseline`** | Tested baseline or empirical target | `"baseline"`, `"compared with"`, evaluation verbs | Active comparative claims |
| **`background_or_related_work`** | Prior work, literature, or survey context | `"et al."`, `[1]`, related work section, survey doc | Relegated to background; no active claim |
| **`references_only`** | Bibliographic citation only | Reference section headers, bracket citations | Completely suppressed |
| **`uncertain`** | Ambiguous or neutral mention | Fallback when cues are absent | Flagged under missing info / gaps |

---

## 4. Independently Verified Benchmark Results

The primary benchmark gate evaluates the deterministic classifier over **73 verified examples** (excluding two unresolvable records from the original 75-example pool):

| Metric | Primary Verified Gate (73 Examples) | Full Traceability (75 Examples) | Original Baseline (v1.0) |
|---|---|---|---|
| **Overall Accuracy** | **63.01%** (46 / 73) | 64.00% (48 / 75) | 56.00% (42 / 75) |
| **Macro Precision** | **0.7367** | 0.7485 | 0.6282 |
| **Macro Recall** | **0.6319** | 0.6469 | 0.5562 |
| **Macro F1-Score** | **0.6237** | 0.6388 | 0.5650 |
| **Weighted F1-Score** | **0.6579** | 0.6585 | 0.5779 |
| **Active-Method FPR** | **6.98%** (3 / 43 non-active) | 6.67% (3 / 45 non-active) | 15.22% (7 / 46 non-active) |
| **False Active Claims** | **3** (`ALGO_BENCH_037, 067, 069`) | 3 | 7 |
| **Missed Active Methods**| **16** (residual failure mode) | 16 | 17 |
| **Inference Latency** | **0.07 ms / example** (CPU) | 0.07 ms / example | 0.07 ms / example |

### Per-Class Breakdown (73 Verified Examples)
- **`references_only`:** 100% Precision, 100% Recall, 1.0000 F1 (16/16)
- **`experiment_or_baseline`:** 100% Precision, 31.25% Recall, 0.4762 F1 (5/16)
- **`proposed_method`:** 75.00% Precision, 64.29% Recall, 0.6923 F1 (9/14)
- **`background_or_related_work`:** 73.33% Precision, 57.89% Recall, 0.6471 F1 (11/19)
- **`uncertain`:** 20.00% Precision, 62.50% Recall, 0.3030 F1 (5/8)

---

## 5. Key Limitations & Known Failure Modes

1. **High Missed Active Rate (Recall Bottleneck):** While the classifier achieves 100% precision on baselines, it misses 68.75% of them (11/16) because academic papers frequently list comparison models without explicit trigger phrases like `"used as a baseline"`.
2. **Catalog Boundness:** Only algorithm families and variants present in `ALGORITHM_CATALOG` are detected. Completely novel or uncataloged acronyms cannot be classified.
3. **Conservative Catch-All Sink:** 34.2% of predictions fall into `uncertain`, reflecting strict safety thresholds designed to prevent false positive claims.
4. **Table & Figure Layout Blindness:** Baseline models embedded in complex multi-column grid tables or image graphics without prose descriptions are not captured.

---

## 6. Implemented versus Planned Status

| Component | Status | Implementation Details |
|---|---|---|
| **Deterministic Technical Extractor** | **Implemented & In Production** | Fully operational in `scripts/technical_extractor.py` |
| **5-Category Algorithm Classifier** | **Implemented & In Production** | Fully operational in `scripts/technical_extractor.py` |
| **Offline Comparative Review** | **Implemented & In Production** | Fully operational in `scripts/review_paper.py` |
| **73-Example Verified Benchmark Gate** | **Implemented & In Production** | Fully operational in `benchmarks/algorithm_usage/v1/` |
| **Automated Unit & Regression Tests** | **Implemented & In Production** | **47 passing tests** (`tests/`) running in 0.71s |
| **Pretrained NLI Classifier (`nli-MiniLM2-L6-H768`)** | **Planned / Evaluation Harness Only** | Evaluation adapter implemented; model weights **not** downloaded; **zero** production changes |
| **Open-Vocabulary Novel Algorithm NER** | **Planned / Proposed** | Future phase |
| **Table Layout & Multi-Column Parser** | **Planned / Proposed** | Future phase |

---

## 7. Two-Minute Review Script for Project Reviewers

> *"Good morning, members of the review committee.*
>
> *In AI research literature, one of the most persistent failure modes of automated review tools is treating an algorithm mention as proof that a paper implemented that algorithm. For example, if a vision paper merely cites AlphaZero in its related work, standard keyword and vector retrieval systems often hallucinate that both papers share the same methodology.*
>
> *To solve this, we developed ResearchAI's deterministic Stage 1 technical extractor and contextual algorithm-usage classifier. Our engine analyzes sentence-level evidence, section-heading context, and attribution cues to classify every algorithm into five distinct categories: proposed method, experimental baseline, background, references only, or uncertain.*
>
> *Crucially, we established a strict method usage rule: our system never claims two papers share a methodology unless both papers actively implement or benchmark that algorithm. Furthermore, we enforce strict metric comparability rules, preventing invalid comparisons like ranking Accuracy against F1-score or comparing models across different datasets.*
>
> *We evaluated this system against an independently audited, 73-example gold benchmark drawn from genuine corpus papers with verified physical PDF page numbers. The deterministic classifier achieves 63.01% overall accuracy, perfect 100% precision on references and baselines, and limits the critical Active-Method False Positive Rate to just 6.98%, running entirely offline in under one-tenth of a millisecond per example.*
>
> *Our test suite has 47 passing tests verifying these regression protections. We have also prepared an evaluation harness to test pretrained Natural Language Inference models as a potential enhancement to recover implicit baselines. However, we have maintained a strict, defensible engineering baseline: any future model must match our low 6.98% false positive rate before it is permitted in production.*
>
> *Thank you, and we welcome your questions."*

---

## 8. Verification & Execution Reference

```powershell
# 1. Run complete test suite (47 passed in ~0.71s)
python -m unittest discover -s tests

# 2. Run verified benchmark evaluation (73 shared verified examples)
python benchmarks/algorithm_usage/v1/evaluate_benchmark.py --skip-nli

# 3. Run production review pipeline in local-only mode
python scripts/review_paper.py dataset/papers/PAPER_033.pdf --local-only
```
