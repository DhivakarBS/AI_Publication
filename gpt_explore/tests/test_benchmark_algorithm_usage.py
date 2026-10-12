"""
Unit Tests for Algorithm Usage Classification Benchmark
========================================================
Tests verify:
1. Gold dataset schema and class distributions.
2. Deterministic classifier adapter integration with production code.
3. Benchmark metric calculation engine (accuracy, F1, FPR, confusion matrix).
4. NLI classifier harness with mocked weights (100% offline, no internet access).
5. Reproducibility of benchmark evaluation artifacts.
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Import benchmark modules
from benchmarks.algorithm_usage.v1.evaluate_benchmark import (
    DeterministicClassifierAdapter,
    NLICrossEncoderClassifier,
    compute_metrics,
    run_benchmark,
    CLASSES,
    ACTIVE_CLASSES,
    NON_ACTIVE_CLASSES,
    NLI_HYPOTHESES,
    USAGE_PROPOSED_METHOD,
    USAGE_EXPERIMENT_OR_BASELINE,
    USAGE_BACKGROUND_OR_RELATED_WORK,
    USAGE_REFERENCES_ONLY,
    USAGE_UNCERTAIN
)


class TestBenchmarkDatasetAndAdapter(unittest.TestCase):
    """Validates the gold benchmark dataset and deterministic adapter."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = Path("benchmarks/algorithm_usage/v1/gold_dataset.jsonl")
        cls.results_path = Path("benchmarks/algorithm_usage/v1/benchmark_results.json")

        cls.records = []
        if cls.dataset_path.exists():
            cls.records = [
                json.loads(line) for line in cls.dataset_path.read_text(encoding="utf-8").strip().splitlines()
            ]

    def test_01_dataset_file_exists_and_size(self):
        """Verify the gold benchmark dataset exists and contains 50-100 examples."""
        self.assertTrue(self.dataset_path.exists(), "gold_dataset.jsonl must exist")
        self.assertGreaterEqual(len(self.records), 50, "Benchmark must contain at least 50 examples")
        self.assertEqual(len(self.records), 75, "Expected 75 total benchmark examples")
        verified = [r for r in self.records if r.get("verification_status") != "unresolved"]
        self.assertEqual(len(verified), 73, "Expected 73 primary verified examples")
        unresolved = [r for r in self.records if r.get("verification_status") == "unresolved"]
        self.assertEqual(len(unresolved), 2, "Expected 2 documented unresolved gaps")

    def test_02_dataset_schema_and_integrity(self):
        """Verify every record has all required fields and valid taxonomy labels."""
        required_keys = {
            "example_id", "paper_id", "paper_title", "algorithm_name",
            "evidence_sentence", "surrounding_context", "section", "page",
            "expected_label", "label_rationale", "source_file", "source_provenance"
        }
        for r in self.records:
            missing = required_keys - set(r.keys())
            self.assertEqual(len(missing), 0, f"Record {r.get('example_id')} missing keys: {missing}")
            self.assertIn(r["expected_label"], CLASSES, f"Invalid label: {r['expected_label']}")
            self.assertIsNotNone(r["page"], f"Page must not be None for {r['example_id']}")
            self.assertGreater(len(r["evidence_sentence"]), 10, f"Evidence sentence too short in {r['example_id']}")
            self.assertGreater(len(r["label_rationale"]), 10, f"Rationale too short in {r['example_id']}")

    def test_03_class_distribution_coverage(self):
        """Verify all five taxonomy categories have representation in the gold dataset."""
        label_counts = {c: 0 for c in CLASSES}
        for r in self.records:
            label_counts[r["expected_label"]] += 1

        for c in CLASSES:
            self.assertGreaterEqual(label_counts[c], 10, f"Class {c} must have at least 10 verified examples")

    def test_04_prioritized_papers_present(self):
        """Verify prioritized papers (PAPER_033, PAPER_068, PAPER_019, etc.) are present."""
        paper_ids = {r["paper_id"] for r in self.records}
        for pid in ["PAPER_033", "PAPER_068", "PAPER_019", "PAPER_067", "PAPER_081", "PAPER_013"]:
            self.assertIn(pid, paper_ids, f"Prioritized paper {pid} must be in benchmark")

    def test_05_deterministic_adapter_contract(self):
        """Verify deterministic adapter wraps production classify_algorithm_usage correctly."""
        adapter = DeterministicClassifierAdapter()
        test_rec = {
            "evidence_sentence": "We propose a novel deep neural network for network intrusion detection.",
            "section": "Introduction",
            "paper_title": "Network Intrusion Detection"
        }
        pred, ev = adapter.predict(test_rec)
        self.assertEqual(pred, USAGE_PROPOSED_METHOD)
        self.assertEqual(ev, test_rec["evidence_sentence"])


class TestBenchmarkMetricsEngine(unittest.TestCase):
    """Validates metric computation logic, confusion matrix, and active FPR."""

    def test_06_perfect_classification_metrics(self):
        """Verify metrics when predictions match gold labels 100%."""
        dummy_records = [
            {"example_id": f"EX_{i}", "expected_label": c, "evidence_sentence": "text", "section": "sec", "paper_id": "P1", "algorithm_name": "A"}
            for i, c in enumerate(CLASSES * 2)
        ]
        dummy_preds = [c for c in CLASSES * 2]

        m = compute_metrics(dummy_records, dummy_preds)
        self.assertEqual(m["overall"]["accuracy"], 1.0)
        self.assertEqual(m["overall"]["macro_f1"], 1.0)
        self.assertEqual(m["overall"]["active_method_false_positive_rate"], 0.0)
        self.assertEqual(m["overall"]["false_active_claims_count"], 0)
        self.assertEqual(m["overall"]["missed_active_methods_count"], 0)

    def test_07_false_active_claims_and_fpr_calculation(self):
        """Verify that predicting active method on non-active gold correctly triggers FPR."""
        # 2 active, 3 non-active
        records = [
            {"example_id": "1", "expected_label": USAGE_PROPOSED_METHOD, "evidence_sentence": "s1", "section": "s", "paper_id": "P", "algorithm_name": "A"},
            {"example_id": "2", "expected_label": USAGE_EXPERIMENT_OR_BASELINE, "evidence_sentence": "s2", "section": "s", "paper_id": "P", "algorithm_name": "A"},
            {"example_id": "3", "expected_label": USAGE_BACKGROUND_OR_RELATED_WORK, "evidence_sentence": "s3", "section": "s", "paper_id": "P", "algorithm_name": "A"},
            {"example_id": "4", "expected_label": USAGE_REFERENCES_ONLY, "evidence_sentence": "s4", "section": "s", "paper_id": "P", "algorithm_name": "A"},
            {"example_id": "5", "expected_label": USAGE_UNCERTAIN, "evidence_sentence": "s5", "section": "s", "paper_id": "P", "algorithm_name": "A"},
        ]
        # Predict record 3 (gold background) as proposed_method (False Active!)
        # Predict record 5 (gold uncertain) as experiment_or_baseline (False Active!)
        preds = [
            USAGE_PROPOSED_METHOD,
            USAGE_EXPERIMENT_OR_BASELINE,
            USAGE_PROPOSED_METHOD,  # FALSE ACTIVE
            USAGE_REFERENCES_ONLY,
            USAGE_EXPERIMENT_OR_BASELINE  # FALSE ACTIVE
        ]

        m = compute_metrics(records, preds)
        # Denominator: 3 non-active records (background, references, uncertain)
        # False active: 2 (records 3 and 5)
        # FPR = 2 / 3 = 0.6667
        self.assertEqual(m["overall"]["false_active_claims_count"], 2)
        self.assertEqual(m["overall"]["active_method_fpr_denominator"], 3)
        self.assertAlmostEqual(m["overall"]["active_method_false_positive_rate"], 2/3, places=4)

    def test_08_confusion_matrix_dimensions(self):
        """Verify confusion matrix has 5 rows and 5 columns corresponding to all classes."""
        records = [
            {"example_id": f"EX_{i}", "expected_label": c, "evidence_sentence": "text", "section": "sec", "paper_id": "P1", "algorithm_name": "A"}
            for i, c in enumerate(CLASSES)
        ]
        preds = [c for c in CLASSES]
        m = compute_metrics(records, preds)
        cm = m["confusion_matrix"]
        self.assertEqual(len(cm), 5)
        for row_label, cols in cm.items():
            self.assertIn(row_label, CLASSES)
            self.assertEqual(len(cols), 5)


class TestPretrainedNLIHarness(unittest.TestCase):
    """Validates NLI classifier harness with mocked weights and offline safety."""

    def test_09_nli_hypotheses_definitions(self):
        """Verify all five taxonomy categories have complete hypothesis templates."""
        self.assertEqual(len(NLI_HYPOTHESES), 5)
        for c in CLASSES:
            self.assertIn(c, NLI_HYPOTHESES)
            self.assertIn("algorithm", NLI_HYPOTHESES[c])

    def test_10_offline_protection_marks_pending_when_weights_absent(self):
        """Verify NLI classifier gracefully marks pending when weights are absent offline."""
        nli = NLICrossEncoderClassifier()
        if not nli.is_available:
            self.assertIn("pending", nli.availability_message.lower())
            with self.assertRaises(RuntimeError):
                nli.predict({"evidence_sentence": "Test sentence"})

    def test_11_mocked_nli_inference_and_uncertainty_threshold(self):
        """Verify NLI prediction policy and uncertainty threshold using mocked model outputs."""
        import torch

        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        mock_model.device = "cpu"

        # Case A: Clear entailment for proposed_method (index 0 in CLASSES)
        # 5 pairs: [proposed, experiment, background, references, uncertain]
        # Return logits where pair 0 has high entailment score
        logits_tensor = torch.tensor([
            [0.1, 0.1, 8.0],  # pair 0: proposed_method -> high entailment
            [0.1, 0.1, 1.0],  # pair 1: experiment
            [0.1, 0.1, 0.5],  # pair 2: background
            [0.1, 0.1, 0.2],  # pair 3: references
            [0.1, 0.1, 0.1]   # pair 4: uncertain
        ])
        mock_output = MagicMock()
        mock_output.logits = logits_tensor
        mock_model.return_value = mock_output

        nli = NLICrossEncoderClassifier(local_model=mock_model, local_tokenizer=mock_tokenizer, uncertainty_threshold=0.40)
        self.assertTrue(nli.is_available)

        pred, info = nli.predict({
            "evidence_sentence": "We propose AlphaZero for mastering chess.",
            "section": "Introduction",
            "surrounding_context": "",
            "algorithm_name": "AlphaZero"
        })
        self.assertEqual(pred, USAGE_PROPOSED_METHOD)
        self.assertGreater(info["max_prob"], 0.8)

        # Case B: Low confidence across all classes triggers uncertain policy
        flat_logits = torch.tensor([
            [0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0]
        ])
        mock_output_flat = MagicMock()
        mock_output_flat.logits = flat_logits
        mock_model.return_value = mock_output_flat

        # Max prob will be 1/5 = 0.20 < threshold (0.40) -> falls back to uncertain
        pred_flat, info_flat = nli.predict({
            "evidence_sentence": "Neural networks process vectors.",
            "section": "Discussion",
            "surrounding_context": "",
            "algorithm_name": "Neural Network"
        })
        self.assertEqual(pred_flat, USAGE_UNCERTAIN)


class TestBenchmarkArtifactsAndReproducibility(unittest.TestCase):
    """Validates the generated benchmark evaluation artifacts."""

    def test_12_benchmark_results_json_structure(self):
        """Verify benchmarks/algorithm_usage/v1/benchmark_results.json has required schema."""
        results_file = Path("benchmarks/algorithm_usage/v1/benchmark_results.json")
        self.assertTrue(results_file.exists(), "benchmark_results.json must exist")
        data = json.loads(results_file.read_text(encoding="utf-8"))

        self.assertIn("benchmark_metadata", data)
        self.assertIn("deterministic_classifier", data)
        self.assertIn("pretrained_nli_classifier", data)

        det = data["deterministic_classifier"]["metrics"]
        self.assertEqual(det["overall"]["total_examples"], 73)
        self.assertAlmostEqual(det["overall"]["accuracy"], 0.63, places=2)
        self.assertIn("confusion_matrix", det)
        self.assertIn("per_class", det)
        self.assertIn("runtime", det)

        # Verify traceability of full dataset metrics
        det_full = data["deterministic_classifier"]["traceability_full_dataset_metrics"]
        self.assertEqual(det_full["overall"]["total_examples"], 75)
        self.assertAlmostEqual(det_full["overall"]["accuracy"], 0.64, places=2)

        nli = data["pretrained_nli_classifier"]
        self.assertIn(nli["status"], ["pending_offline_model_availability", "completed", "skipped"])


if __name__ == "__main__":
    unittest.main()
