#!/usr/bin/env python3
"""
ResearchAI — Algorithm Usage Classification Benchmark Evaluator
================================================================
Reproducible benchmark evaluating the production deterministic classifier
against the pretrained NLI cross-encoder model:
    cross-encoder/nli-MiniLM2-L6-H768

Evaluation only — does not modify production code or require internet access.
"""

import sys
import os
import json
import time
import argparse
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to sys.path so we can import production modules
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.technical_extractor import (
    classify_algorithm_usage,
    USAGE_PROPOSED_METHOD,
    USAGE_EXPERIMENT_OR_BASELINE,
    USAGE_BACKGROUND_OR_RELATED_WORK,
    USAGE_REFERENCES_ONLY,
    USAGE_UNCERTAIN
)

# Standard five taxonomy classes
CLASSES = [
    USAGE_PROPOSED_METHOD,
    USAGE_EXPERIMENT_OR_BASELINE,
    USAGE_BACKGROUND_OR_RELATED_WORK,
    USAGE_REFERENCES_ONLY,
    USAGE_UNCERTAIN
]

ACTIVE_CLASSES = {USAGE_PROPOSED_METHOD, USAGE_EXPERIMENT_OR_BASELINE}
NON_ACTIVE_CLASSES = {USAGE_BACKGROUND_OR_RELATED_WORK, USAGE_REFERENCES_ONLY, USAGE_UNCERTAIN}

# NLI Hypothesis Templates
NLI_HYPOTHESES = {
    USAGE_PROPOSED_METHOD: "This algorithm is proposed, introduced, or developed as the main method of this paper.",
    USAGE_EXPERIMENT_OR_BASELINE: "This algorithm is used as an experimental baseline or benchmark comparison in this paper.",
    USAGE_BACKGROUND_OR_RELATED_WORK: "This algorithm is discussed as prior work, background, or literature in this paper.",
    USAGE_REFERENCES_ONLY: "This algorithm appears only as a bibliographic citation or reference entry in this paper.",
    USAGE_UNCERTAIN: "The text does not provide sufficient evidence to determine how this algorithm is used in this paper."
}


# ============================================================
# 1. DETERMINISTIC CLASSIFIER ADAPTER
# ============================================================

class DeterministicClassifierAdapter:
    """Adapter wrapping the existing production rule-based classifier."""

    def __init__(self):
        self.name = "deterministic_rule_based_classifier"

    def predict(self, record: Dict[str, Any]) -> Tuple[str, str]:
        """
        Runs classify_algorithm_usage using the benchmark record fields.
        Returns (predicted_label, evidence).
        """
        evidence = record.get("evidence_sentence", "")
        section = record.get("section", "unknown")
        title = record.get("paper_title", "")

        predicted_label = classify_algorithm_usage(
            snippet=evidence,
            section=section,
            full_sentence=evidence,
            paper_title=title
        )
        return predicted_label, evidence


# ============================================================
# 2. PRETRAINED NLI CROSS-ENCODER HARNESS
# ============================================================

class NLICrossEncoderClassifier:
    """
    Evaluator for cross-encoder/nli-MiniLM2-L6-H768.
    Supports offline local execution or injected/mocked scoring for testing.
    """

    MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

    def __init__(self, local_model=None, local_tokenizer=None, uncertainty_threshold: float = 0.40):
        self.name = self.MODEL_NAME
        self.uncertainty_threshold = uncertainty_threshold
        self.model = local_model
        self.tokenizer = local_tokenizer
        self.is_available = False
        self.availability_message = ""
        self.device = "cpu"
        self.load_time_sec = 0.0

        if local_model is not None and local_tokenizer is not None:
            self.is_available = True
            self.availability_message = "Injected / custom model provided"
        else:
            self._attempt_offline_load()

    def _attempt_offline_load(self):
        """Attempts to load weights strictly from local cache (offline)."""
        t0 = time.perf_counter()
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            # Must strictly use local_files_only=True to prevent network calls
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                local_files_only=True
            )
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.is_available = True
            self.load_time_sec = time.perf_counter() - t0
            self.availability_message = f"Loaded successfully from local cache on {self.device}"
        except Exception as e:
            self.is_available = False
            self.load_time_sec = 0.0
            self.availability_message = (
                f"Model weights not found in local Hugging Face cache ({e.__class__.__name__}: {str(e)}). "
                f"Evaluation marked as pending offline availability."
            )

    def predict(self, record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Performs NLI inference over all 5 hypothesis candidates.
        Premise: Evidence sentence + surrounding context + section metadata.
        """
        if not self.is_available:
            raise RuntimeError(f"NLI model unavailable: {self.availability_message}")

        import torch

        evidence = record.get("evidence_sentence", "")
        section = record.get("section", "unknown")
        context = record.get("surrounding_context", "")
        alg = record.get("algorithm_name", "algorithm")

        premise = f"Section: {section}. Text: {evidence} {context}".strip()

        pairs = []
        for cls in CLASSES:
            hypo = NLI_HYPOTHESES[cls].replace("This algorithm", f"The algorithm {alg}")
            pairs.append((premise, hypo))

        with torch.no_grad():
            features = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            if hasattr(features, "to"):
                features = features.to(self.device)
            elif isinstance(features, dict):
                features = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in features.items()}
            outputs = self.model(**features)
            logits = outputs.logits

            # Inspect output format: 3 labels for NLI cross-encoder: contradiction, neutral, entailment
            # Entailment is typically index 1 or 2 depending on model config
            # cross-encoder/nli-MiniLM2-L6-H768 maps: 0: contradiction, 1: neutral, 2: entailment (or label2)
            id2label = getattr(self.model.config, "id2label", {})
            entail_idx = 2
            for idx, label_name in id2label.items():
                if "entail" in label_name.lower():
                    entail_idx = int(idx)
                    break

            entail_scores = logits[:, entail_idx]
            probs = torch.softmax(entail_scores, dim=0).cpu().tolist()

        score_dict = {cls: probs[i] for i, cls in enumerate(CLASSES)}
        best_cls = max(score_dict, key=score_dict.get)
        best_prob = score_dict[best_cls]

        # Uncertainty policy: if highest probability is below threshold, mark as uncertain
        if best_prob < self.uncertainty_threshold and best_cls != USAGE_UNCERTAIN:
            pred_label = USAGE_UNCERTAIN
        else:
            pred_label = best_cls

        return pred_label, {"scores": score_dict, "max_prob": best_prob}


# ============================================================
# 3. METRIC COMPUTATION ENGINE
# ============================================================

def compute_metrics(
    records: List[Dict[str, Any]],
    predictions: List[str]
) -> Dict[str, Any]:
    """
    Computes overall accuracy, macro/weighted F1, per-class metrics,
    5x5 confusion matrix, and active-method false positive rate.
    """
    assert len(records) == len(predictions), "Mismatched record and prediction lengths"
    n = len(records)

    # 5x5 Confusion Matrix: rows = Gold, cols = Predicted
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    gold_counts = {c: 0 for c in CLASSES}
    pred_counts = {c: 0 for c in CLASSES}
    correct_counts = {c: 0 for c in CLASSES}

    correct_total = 0
    false_active_claims = []
    missed_active_methods = []
    uncertain_forced = []

    for rec, pred in zip(records, predictions):
        gold = rec["expected_label"]
        cm[gold][pred] += 1
        gold_counts[gold] += 1
        pred_counts[pred] += 1

        if gold == pred:
            correct_total += 1
            correct_counts[gold] += 1

        # Error tracking
        # False Active Method: Gold is Non-Active, Pred is Active
        if gold in NON_ACTIVE_CLASSES and pred in ACTIVE_CLASSES:
            false_active_claims.append({
                "example_id": rec["example_id"],
                "paper_id": rec["paper_id"],
                "algorithm_name": rec["algorithm_name"],
                "gold": gold,
                "pred": pred,
                "evidence": rec["evidence_sentence"],
                "section": rec["section"]
            })

        # Missed Active Method: Gold is Active, Pred is Non-Active
        if gold in ACTIVE_CLASSES and pred in NON_ACTIVE_CLASSES:
            missed_active_methods.append({
                "example_id": rec["example_id"],
                "paper_id": rec["paper_id"],
                "algorithm_name": rec["algorithm_name"],
                "gold": gold,
                "pred": pred,
                "evidence": rec["evidence_sentence"],
                "section": rec["section"]
            })

        # Uncertain Forced: Gold is Uncertain, Pred is Definitive
        if gold == USAGE_UNCERTAIN and pred != USAGE_UNCERTAIN:
            uncertain_forced.append({
                "example_id": rec["example_id"],
                "paper_id": rec["paper_id"],
                "algorithm_name": rec["algorithm_name"],
                "gold": gold,
                "pred": pred,
                "evidence": rec["evidence_sentence"]
            })

    # Per-class Precision, Recall, F1
    per_class = {}
    macro_prec = 0.0
    macro_rec = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0

    for c in CLASSES:
        support = gold_counts[c]
        prec = (correct_counts[c] / pred_counts[c]) if pred_counts[c] > 0 else 0.0
        rec = (correct_counts[c] / support) if support > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "predicted_count": pred_counts[c]
        }

        macro_prec += prec
        macro_rec += rec
        macro_f1 += f1
        weighted_f1 += f1 * (support / n)

    num_classes = len(CLASSES)
    macro_prec = round(macro_prec / num_classes, 4)
    macro_rec = round(macro_rec / num_classes, 4)
    macro_f1 = round(macro_f1 / num_classes, 4)
    weighted_f1 = round(weighted_f1, 4)
    accuracy = round(correct_total / n, 4)

    # Active Method FPR:
    # Denominator: All gold non-active examples (background + references + uncertain)
    total_gold_non_active = sum(gold_counts[c] for c in NON_ACTIVE_CLASSES)
    active_fpr = round(len(false_active_claims) / total_gold_non_active, 4) if total_gold_non_active > 0 else 0.0

    return {
        "overall": {
            "total_examples": n,
            "correct": correct_total,
            "accuracy": accuracy,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "active_method_false_positive_rate": active_fpr,
            "active_method_fpr_denominator": total_gold_non_active,
            "false_active_claims_count": len(false_active_claims),
            "missed_active_methods_count": len(missed_active_methods),
            "uncertain_forced_count": len(uncertain_forced)
        },
        "per_class": per_class,
        "confusion_matrix": cm,
        "errors": {
            "false_active_claims": false_active_claims,
            "missed_active_methods": missed_active_methods,
            "uncertain_forced": uncertain_forced
        }
    }


# ============================================================
# 4. BENCHMARK RUNNER
# ============================================================

def run_benchmark(
    dataset_path: Path,
    output_path: Path,
    skip_nli: bool = False,
    mock_nli_cls = None
) -> Dict[str, Any]:
    """Executes the benchmark evaluation on primary verified examples, with full-dataset traceability."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at: {dataset_path}")

    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    verified_records = [r for r in records if r.get("verification_status") != "unresolved"]
    unresolved_records = [r for r in records if r.get("verification_status") == "unresolved"]

    print(f"Loaded {len(records)} gold benchmark examples from: {dataset_path}")
    print(f"  Primary Verified Evaluation Set: {len(verified_records)} examples")
    print(f"  Unresolved Gaps Excluded from Primary Metric: {len(unresolved_records)} examples ({[r['example_id'] for r in unresolved_records]})")

    shared_verified_ids = [r["example_id"] for r in verified_records]

    # 1. Deterministic Classifier Evaluation (Primary: Verified Records)
    det_adapter = DeterministicClassifierAdapter()
    t0 = time.perf_counter()
    det_preds_verified = []
    for r in verified_records:
        pred, _ = det_adapter.predict(r)
        det_preds_verified.append(pred)
    det_time_sec = time.perf_counter() - t0

    det_metrics_verified = compute_metrics(verified_records, det_preds_verified)
    det_metrics_verified["runtime"] = {
        "total_inference_time_sec": round(det_time_sec, 6),
        "mean_inference_time_per_example_ms": round((det_time_sec / len(verified_records)) * 1000, 4) if verified_records else 0.0,
        "device": "cpu"
    }

    # Traceability: Also compute metrics on all 75 records
    det_preds_all = []
    for r in records:
        pred, _ = det_adapter.predict(r)
        det_preds_all.append(pred)
    det_metrics_all = compute_metrics(records, det_preds_all)

    # 2. NLI Model Evaluation (Strictly on the Shared Verified Evaluation Set)
    nli_results = {
        "status": "pending",
        "model_identifier": NLICrossEncoderClassifier.MODEL_NAME,
        "availability_notes": "",
        "metrics": None
    }

    if mock_nli_cls is not None:
        # Mocked/Injected NLI classifier for test harness
        nli_classifier = mock_nli_cls
    elif skip_nli:
        nli_classifier = None
        nli_results["status"] = "skipped"
        nli_results["availability_notes"] = "Explicitly skipped via --skip-nli"
    else:
        nli_classifier = NLICrossEncoderClassifier()

    if nli_classifier is not None:
        if nli_classifier.is_available:
            t0 = time.perf_counter()
            nli_preds = []
            for r in verified_records:
                pred, _ = nli_classifier.predict(r)
                nli_preds.append(pred)
            nli_time_sec = time.perf_counter() - t0

            nli_metrics = compute_metrics(verified_records, nli_preds)
            nli_metrics["runtime"] = {
                "model_load_time_sec": round(nli_classifier.load_time_sec, 4),
                "total_inference_time_sec": round(nli_time_sec, 4),
                "mean_inference_time_per_example_ms": round((nli_time_sec / len(verified_records)) * 1000, 2) if verified_records else 0.0,
                "device": nli_classifier.device
            }
            nli_results["status"] = "completed"
            nli_results["metrics"] = nli_metrics
            nli_results["availability_notes"] = nli_classifier.availability_message
        else:
            nli_results["status"] = "pending_offline_model_availability"
            nli_results["availability_notes"] = nli_classifier.availability_message
            nli_results["hypotheses"] = NLI_HYPOTHESES
            nli_results["hardware_requirements"] = {
                "expected_model_parameters": "~80M (6 layers, 768 hidden, 12 heads)",
                "fp32_weights_size": "~320 MB",
                "estimated_peak_memory_cpu": "~450 MB",
                "platform": platform.platform(),
                "python_version": platform.python_version()
            }

    # Save machine-readable output
    output_payload = {
        "benchmark_metadata": {
            "name": "ResearchAI Algorithm Usage Classification Benchmark",
            "version": "v1.1",
            "dataset_path": str(dataset_path),
            "num_verified_examples": len(verified_records),
            "num_total_examples": len(records),
            "num_unresolved_examples": len(unresolved_records),
            "unresolved_example_ids": [r["example_id"] for r in unresolved_records],
            "shared_verified_example_ids": shared_verified_ids,
            "classes": CLASSES,
            "active_classes": list(ACTIVE_CLASSES),
            "non_active_classes": list(NON_ACTIVE_CLASSES),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "os": platform.platform(),
                "python": sys.version.split()[0]
            }
        },
        "deterministic_classifier": {
            "status": "completed",
            "model_identifier": det_adapter.name,
            "metrics": det_metrics_verified,
            "traceability_full_dataset_metrics": det_metrics_all
        },
        "pretrained_nli_classifier": nli_results,
        "unresolved_records": [
            {
                "example_id": r["example_id"],
                "paper_id": r["paper_id"],
                "algorithm_name": r["algorithm_name"],
                "reason": r.get("unresolved_reason", "Absent from paper"),
                "verification_status": r.get("verification_status", "unresolved")
            }
            for r in unresolved_records
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark results to: {output_path}")

    return output_payload


def print_summary_report(results: Dict[str, Any]):
    """Prints a clear summary table of benchmark results."""
    det = results["deterministic_classifier"]["metrics"]
    det_ov = det["overall"]
    meta = results["benchmark_metadata"]

    print("\n" + "=" * 70)
    print("RESEARCHAI ALGORITHM USAGE BENCHMARK — PRIMARY VERIFIED REPORT")
    print("=" * 70)
    print(f"Evaluation Scope: Verified Examples Only ({meta['num_verified_examples']} examples)")
    print(f"Excluded Unresolved Gaps: {meta['num_unresolved_examples']} examples ({meta['unresolved_example_ids']})")
    print(f"Deterministic Accuracy: {det_ov['accuracy']:.2%} ({det_ov['correct']}/{det_ov['total_examples']})")
    print(f"Deterministic Macro F1: {det_ov['macro_f1']:.4f}")
    print(f"Deterministic Weighted F1: {det_ov['weighted_f1']:.4f}")
    print(f"Deterministic Inference Time: {det['runtime']['total_inference_time_sec']*1000:.2f} ms total ({det['runtime']['mean_inference_time_per_example_ms']:.4f} ms/example)")
    print(f"Active Method FPR: {det_ov['active_method_false_positive_rate']:.2%} ({det_ov['false_active_claims_count']}/{det_ov['active_method_fpr_denominator']})")
    print(f"False Active Claims: {det_ov['false_active_claims_count']}")
    print(f"Missed Active Methods: {det_ov['missed_active_methods_count']}")
    print(f"Uncertain Forced into Definitive: {det_ov['uncertain_forced_count']}")

    print("\nPer-Class Breakdown (Primary Verified Benchmark — 73 Examples):")
    print(f"{'Class':<30} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Support':<7}")
    print("-" * 65)
    for c in CLASSES:
        pc = det["per_class"][c]
        print(f"{c:<30} | {pc['precision']:<7.4f} | {pc['recall']:<7.4f} | {pc['f1_score']:<7.4f} | {pc['support']:<7}")

    print("\n5x5 Confusion Matrix (Rows = Gold, Columns = Predicted):")
    hdr = f"{'Gold \\ Pred':<25} | " + " | ".join([c[:8] for c in CLASSES])
    print(hdr)
    print("-" * len(hdr))
    for g in CLASSES:
        row = f"{g:<25} | " + " | ".join([f"{det['confusion_matrix'][g][p]:<8}" for p in CLASSES])
        print(row)

    # Full dataset reference
    det_all = results["deterministic_classifier"]["traceability_full_dataset_metrics"]["overall"]
    print("\n" + "-" * 70)
    print("FULL DATASET TRACEABILITY (75 Examples including 2 Unresolved Gaps):")
    print(f"Accuracy: {det_all['accuracy']:.2%} ({det_all['correct']}/{det_all['total_examples']}) | Macro F1: {det_all['macro_f1']:.4f} | Active FPR: {det_all['active_method_false_positive_rate']:.2%}")
    print("-" * 70)

    nli = results["pretrained_nli_classifier"]
    print("\n" + "=" * 70)
    print(f"Pretrained NLI Classifier ({NLICrossEncoderClassifier.MODEL_NAME}):")
    print(f"Status: {nli['status']}")
    print(f"Notes:  {nli['availability_notes']}")
    if nli.get("metrics"):
        nli_ov = nli["metrics"]["overall"]
        print(f"NLI Accuracy (on Shared 73 Verified Set): {nli_ov['accuracy']:.2%}")
        print(f"NLI Macro F1 (on Shared 73 Verified Set): {nli_ov['macro_f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Algorithm Usage Classification Benchmark")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/algorithm_usage/v1/gold_dataset.jsonl"),
        help="Path to gold dataset jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/algorithm_usage/v1/benchmark_results.json"),
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip NLI model evaluation"
    )

    args = parser.parse_args()
    res = run_benchmark(dataset_path=args.dataset, output_path=args.output, skip_nli=args.skip_nli)
    print_summary_report(res)
