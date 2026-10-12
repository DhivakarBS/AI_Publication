"""
test_stage1_technical_extractor.py
Focused tests for ResearchAI Stage 1: Technical Extraction.

Verifies:
1. Recognition of YOLO and YOLOv8n as general algorithm family and specific variant.
2. Extraction of explicitly stated metric-value pairs.
3. Handling of values with and without percentage symbols.
4. Preservation of ambiguous '91.3 Index Terms' without inventing a metric or unit.
5. Graceful handling of missing sections and empty text.
6. Compatibility with existing report structure and local-only workflow.
"""

import os
import sys
import unittest
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from technical_extractor import (
    extract_technical_facts,
    parse_paper_sections,
    extract_algorithms,
    extract_techniques,
    extract_datasets,
    extract_experimental_setup,
    extract_metrics_and_results,
    extract_contributions,
    extract_limitations,
    extract_ambiguities_and_warnings
)


class TestTechnicalExtractor(unittest.TestCase):

    def test_1_algorithm_family_and_variant_recognition(self):
        """Verify recognition of YOLO as family and YOLOv8n as specific variant."""
        text = (
            "The YOLOv8n model was trained for object detection. "
            "In comparison with standard YOLO architectures, the lightweight variant reduces latency."
        )
        facts = extract_technical_facts(full_text=text)
        algorithms = facts["algorithms"]

        # Find YOLO family and YOLOv8n variant
        yolo_family = next((a for a in algorithms if a["family"] == "YOLO" and a["variant"] is None), None)
        yolov8n_variant = next((a for a in algorithms if a["variant"] == "YOLOv8n"), None)

        self.assertIsNotNone(yolov8n_variant, "YOLOv8n variant should be recognized")
        self.assertEqual(yolov8n_variant["family"], "YOLO")
        self.assertEqual(yolov8n_variant["variant"], "YOLOv8n")
        self.assertEqual(yolov8n_variant["category"], "object_detection")

        self.assertIsNotNone(yolo_family, "General YOLO family should also be recognized")
        self.assertEqual(yolo_family["family"], "YOLO")

        # Verify legacy methods_mentioned contains both
        self.assertIn("yolo", facts["methods_mentioned"])
        self.assertIn("yolov8n", facts["methods_mentioned"])

    def test_2_explicit_metric_value_extraction(self):
        """Verify extraction of explicitly stated metric-value pairs with confidence."""
        text = (
            "On the benchmark evaluation, the proposed model achieved an accuracy of 94.5% "
            "and reached 92.1% precision on the validation set."
        )
        facts = extract_technical_facts(full_text=text)
        results = facts["metrics_and_results"]

        acc_res = next((r for r in results if r["metric_name"] == "accuracy"), None)
        prec_res = next((r for r in results if r["metric_name"] == "precision"), None)

        self.assertIsNotNone(acc_res, "Accuracy result must be extracted")
        self.assertEqual(acc_res["raw_value"], "94.5")
        self.assertEqual(acc_res["unit"], "%")
        self.assertEqual(acc_res["status"], "confirmed")

        self.assertIsNotNone(prec_res, "Precision result must be extracted")
        self.assertEqual(prec_res["raw_value"], "92.1")
        self.assertEqual(prec_res["unit"], "%")
        self.assertEqual(prec_res["status"], "confirmed")

        # Verify legacy fields
        self.assertIn("accuracy", facts["metrics_mentioned"])
        self.assertIn("precision", facts["metrics_mentioned"])
        self.assertIn(94.5, facts["accuracy_values_found_percent"])

    def test_3_values_with_and_without_percentage_symbols(self):
        """Verify handling of metrics with and without percentage symbols."""
        text = (
            "The detector achieved an mAP of 0.872 and mAP@0.5 of 0.915. "
            "The F1-score reached 0.89, while recall was 88.4% and IoU was 76.5%."
        )
        facts = extract_technical_facts(full_text=text)
        results = facts["metrics_and_results"]

        map_res = next((r for r in results if r["metric_name"] == "map"), None)
        map50_res = next((r for r in results if r["metric_name"] == "map@0.5"), None)
        f1_res = next((r for r in results if r["metric_name"] == "f1-score"), None)
        recall_res = next((r for r in results if r["metric_name"] == "recall"), None)
        iou_res = next((r for r in results if r["metric_name"] == "iou"), None)

        # Values without percentage
        self.assertIsNotNone(map_res)
        self.assertEqual(map_res["raw_value"], "0.872")
        self.assertIsNone(map_res["unit"], "mAP without % should have unit None")

        self.assertIsNotNone(map50_res)
        self.assertEqual(map50_res["raw_value"], "0.915")
        self.assertIsNone(map50_res["unit"], "mAP@0.5 without % should have unit None")

        self.assertIsNotNone(f1_res)
        self.assertEqual(f1_res["raw_value"], "0.89")
        self.assertIsNone(f1_res["unit"])

        # Values with percentage
        self.assertIsNotNone(recall_res)
        self.assertEqual(recall_res["raw_value"], "88.4")
        self.assertEqual(recall_res["unit"], "%")

        self.assertIsNotNone(iou_res)
        self.assertEqual(iou_res["raw_value"], "76.5")
        self.assertEqual(iou_res["unit"], "%")

    def test_4_preservation_of_ambiguous_91_3_index_terms(self):
        """
        Critical test: Text contains '91.3' immediately before 'Index Terms'.
        Must NOT classify as accuracy, mAP, or 91.3%.
        Must record as an ambiguous performance value.
        """
        raw_abstract = (
            "—In forest-adjacent forest boundaries, human-wildlife conflict has become a major problem. "
            "Wild animals regularly enter residential and agricultural areas, which may have an impact on livestock "
            "security, agricultural productivity, and human safety. For continuous, real-time monitoring, "
            "conventional surveillance techniques like motion sensors, camera traps, and manual patrolling are inadequate. "
            "In order to monitor wildlife in real time, this article suggests a Wildlife Intrusion Detection and Early Warning "
            "System that combines edge computing, deep learning, and IoT-based alerting. The YOLOv8n model was trained with "
            "wild and non-wild life annotated images and deployed in Raspberry Pi with camera for detection. Upon detection, "
            "the system triggers a Telegram-based instant notification along with the captured image and stores the detection "
            "record locally on the Raspberry Pi. Experimental results demonstrate strong detection performance with 91.3 "
            "Index Terms—- Wildlife detection; YOLOv8n; Raspberry Pi; edge devices; object detection; Telegram alert; "
            "human–wildlife conflict; deep learning I."
        )

        pages = [{"page": 1, "text": raw_abstract}]
        facts = extract_technical_facts(pages=pages)

        # 1. 91.3 must NOT be classified as accuracy or mAP
        self.assertNotIn(91.3, facts["accuracy_values_found_percent"], "91.3 must NOT be in accuracy_values_found_percent")
        self.assertNotIn("accuracy", facts["metrics_mentioned"], "accuracy must NOT be invented")

        # 2. 91.3 must be in metrics_and_results as ambiguous
        results = facts["metrics_and_results"]
        ambiguous_91_3 = next((r for r in results if r["raw_value"] == "91.3"), None)

        self.assertIsNotNone(ambiguous_91_3, "91.3 should be recorded in metrics_and_results")
        self.assertIsNone(ambiguous_91_3["metric_name"], "metric_name must be None for 91.3")
        self.assertIsNone(ambiguous_91_3["unit"], "unit must be None (not '%') for 91.3")
        self.assertEqual(ambiguous_91_3["status"], "ambiguous")
        self.assertIn("Index Terms", ambiguous_91_3["explanation"])
        self.assertEqual(ambiguous_91_3["page"], 1)

        # 3. Warning must be registered
        warnings = facts["ambiguities_and_warnings"]
        has_ambig_warning = any(
            w["type"] == "ambiguous_performance_value" and "91.3" in w["description"]
            for w in warnings
        )
        self.assertTrue(has_ambig_warning, "Warning for ambiguous 91.3 must be recorded")

        # 4. Techniques, hardware, and dataset must be extracted accurately from this text
        hardware_names = [h["name"] for h in facts["experimental_setup"]["hardware"]]
        self.assertIn("Raspberry Pi", hardware_names)

        tech_names = [t["name"] for t in facts["techniques_and_architectures"]]
        self.assertIn("edge computing", tech_names)
        self.assertIn("deep learning", tech_names)

        # Custom dataset description extracted
        dataset_descs = [d["name"] for d in facts["datasets"]]
        self.assertTrue(
            any("wild and non-wild life annotated images" in d for d in dataset_descs),
            "Custom annotated images dataset should be captured"
        )

    def test_5_graceful_handling_missing_sections_and_empty_text(self):
        """Verify graceful degradation on empty, None, or incomplete text."""
        # Case A: None and empty strings
        facts_empty = extract_technical_facts(full_text="")
        self.assertEqual(facts_empty["algorithms"], [])
        self.assertEqual(facts_empty["metrics_and_results"], [])
        self.assertEqual(facts_empty["methods_mentioned"], [])
        self.assertTrue(any(w["type"] == "empty_input" for w in facts_empty["ambiguities_and_warnings"]))

        facts_none = extract_technical_facts(pages=None, full_text=None)
        self.assertEqual(facts_none["algorithms"], [])

        # Case B: Incomplete text with missing sections
        fragment = "We introduce a new method. Experiments are left for future work."
        facts_fragment = extract_technical_facts(full_text=fragment)
        warnings = facts_fragment["ambiguities_and_warnings"]
        warning_types = [w["type"] for w in warnings]

        self.assertIn("missing_section", warning_types)
        self.assertIn("no_algorithms_detected", warning_types)

    def test_6_report_schema_and_local_only_compatibility(self):
        """Verify schema compatibility with existing review report structure."""
        text = (
            "Abstract\n"
            "This paper explores ResNet-50 for image classification on ImageNet. "
            "Our model achieved an accuracy of 76.1% with batch size 32 on NVIDIA RTX 3090 GPU.\n"
        )
        facts = extract_technical_facts(full_text=text)

        # Simulate the report structure in review_paper.py
        report = {
            "created_at": "2026-10-12T14:00:00",
            "input_file": "test_paper.pdf",
            "sha256": "fakehash123",
            "exact_duplicate": None,
            "title_candidate": "Test Paper Title",
            "abstract": "Test Abstract",
            "new_paper_facts": facts,
            "retrieved_chunk_count": 5,
            "candidate_papers": [],
            "research_profile": None,
            "relevance_assessment": None,
            "llm_model": "gpt-4o-mini"
        }

        # Check required fields of new_paper_facts
        npf = report["new_paper_facts"]
        self.assertIn("algorithms", npf)
        self.assertIn("techniques_and_architectures", npf)
        self.assertIn("datasets", npf)
        self.assertIn("experimental_setup", npf)
        self.assertIn("metrics_and_results", npf)
        self.assertIn("contributions", npf)
        self.assertIn("limitations", npf)
        self.assertIn("ambiguities_and_warnings", npf)
        self.assertIn("methods_mentioned", npf)
        self.assertIn("metrics_mentioned", npf)
        self.assertIn("accuracy_values_found_percent", npf)

        # Verify that facts extraction worked completely without LLM
        self.assertGreater(len(npf["algorithms"]), 0)
        self.assertGreater(len(npf["datasets"]), 0)
        self.assertGreater(len(npf["metrics_and_results"]), 0)

    def test_7_fix_false_10_78_recall_pairing(self):
        """
        Verify that tabular scrap 10.78 preceding 'recall and precision results are not listed'
        is NOT extracted as confirmed recall.
        Verify that authors' actual reported recall and precision are extracted.
        """
        text = (
            "Total 125881 17802 89.22 92.97 89.22 90.76 10.78 recall and precision results are not listed "
            "but the bar charts show them to be around 69% and 83% respectively. "
            "Our model has produced superior results by offering f-score of 87.37%, recall of 85.42% and precision of 100.00%."
        )
        facts = extract_technical_facts(full_text=text)
        results = facts["metrics_and_results"]

        # 1. 10.78 must NOT be extracted as recall
        recall_10_78 = next(
            (r for r in results if r["metric_name"] == "recall" and r["raw_value"] == "10.78"),
            None
        )
        self.assertIsNone(recall_10_78, "False 10.78 recall pairing must NOT be confirmed as recall")

        # 2. Authors' confirmed metrics must be extracted with attribution == 'proposed_work'
        actual_recall = next(
            (r for r in results if r["metric_name"] == "recall" and r["raw_value"] == "85.42"),
            None
        )
        self.assertIsNotNone(actual_recall, "85.42% recall should be extracted")
        self.assertEqual(actual_recall["attribution"], "proposed_work")

        actual_precision = next(
            (r for r in results if r["metric_name"] == "precision" and r["raw_value"] == "100.00"),
            None
        )
        self.assertIsNotNone(actual_precision, "100.00% precision should be extracted")
        self.assertEqual(actual_precision["attribution"], "proposed_work")

        actual_fscore = next(
            (r for r in results if r["metric_name"] == "f1-score" and r["raw_value"] == "87.37"),
            None
        )
        self.assertIsNotNone(actual_fscore, "87.37% f1-score should be extracted")
        self.assertEqual(actual_fscore["attribution"], "proposed_work")

    def test_8_bibliography_exclusion(self):
        """
        Verify that algorithms and software mentioned only in the bibliography/references
        section do NOT become paper-specific technical facts.
        """
        text = (
            "Abstract\n"
            "We propose a novel neural architecture for cybersecurity monitoring.\n"
            "5. Methodology\n"
            "The proposed NDAE model is trained on network traffic.\n"
            "6. Results\n"
            "Our model achieved an accuracy of 97.85%.\n"
            "References\n"
            "[1] B. Dong and X. Wang, 'Comparison deep learning method to traditional methods.'\n"
            "[2] S. Hou, 'Deep4MalDroid: A Framework Based on Linux Kernel System Call Graphs.'\n"
            "[3] T. Smith, 'Classification using Random Forest and Docker containers in healthcare.'"
        )
        facts = extract_technical_facts(full_text=text)

        # Software mentioned only in references must not be extracted
        sw_names = [s["name"] for s in facts["experimental_setup"]["software_and_frameworks"]]
        self.assertNotIn("Linux", sw_names, "Linux from reference [2] must NOT be extracted")
        self.assertNotIn("Docker", sw_names, "Docker from reference [3] must NOT be extracted")

        # Algorithms mentioned only in references must not be extracted
        alg_names = [a["name"] for a in facts["algorithms"]]
        self.assertNotIn("Random Forest", alg_names, "Random Forest from reference [3] must NOT be extracted")

        # Proposed model accuracy should still be captured
        acc_results = [r for r in facts["metrics_and_results"] if r["metric_name"] == "accuracy"]
        self.assertEqual(len(acc_results), 1)
        self.assertEqual(acc_results[0]["raw_value"], "97.85")

    def test_9_kdd_cup_99_and_nsl_kdd_recognition(self):
        """
        Verify recognition of KDD Cup '99 and NSL-KDD benchmark datasets.
        """
        text = (
            "For our evaluations we have utilised the benchmark KDD Cup ’99 and NSL-KDD datasets "
            "and achieved very promising results."
        )
        facts = extract_technical_facts(full_text=text)
        ds_names = [d["name"] for d in facts["datasets"]]

        self.assertIn("KDD Cup '99", ds_names, "KDD Cup '99 dataset must be recognized")
        self.assertIn("NSL-KDD", ds_names, "NSL-KDD dataset must be recognized")

    def test_10_fact_deduplication(self):
        """
        Verify that facts mentioned repeatedly across different pages are deduplicated
        while preserving distinct page numbers and evidence.
        """
        pages = [
            {"page": 1, "text": "We introduce YOLOv8n for real-time edge processing."},
            {"page": 3, "text": "During training, YOLOv8n achieves high throughput."},
            {"page": 5, "text": "In summary, YOLOv8n demonstrated robust performance."}
        ]
        facts = extract_technical_facts(pages=pages)
        yolov8n_entries = [a for a in facts["algorithms"] if a.get("variant") == "YOLOv8n"]

        self.assertEqual(len(yolov8n_entries), 1, "YOLOv8n should appear exactly once in algorithms")
        entry = yolov8n_entries[0]
        self.assertEqual(entry["pages"], [1, 3, 5], "All distinct page numbers must be preserved")
        self.assertEqual(entry["page"], 1, "Primary page number should be the first page")

    def test_11_metric_attribution_proposed_vs_prior_work(self):
        """
        Verify that proposed-work metrics are accurately distinguished from cited prior-work metrics.
        """
        text = (
            "Tang et al. [23] claim that their Deep Neural Network (DNN) approach achieved an accuracy of 75.75% "
            "when performing a 5-class classification of the NSL-KDD dataset. "
            "This result is lower than our achieved accuracy of 85.42%."
        )
        facts = extract_technical_facts(full_text=text)
        results = facts["metrics_and_results"]

        prior_res = next((r for r in results if r["raw_value"] == "75.75"), None)
        proposed_res = next((r for r in results if r["raw_value"] == "85.42"), None)

        self.assertIsNotNone(prior_res, "Prior work accuracy 75.75% should be extracted")
        self.assertEqual(prior_res["attribution"], "prior_work", "75.75% should be attributed to prior_work")
        self.assertEqual(prior_res["status"], "confirmed")

        self.assertIsNotNone(proposed_res, "Proposed model accuracy 85.42% should be extracted")
        self.assertEqual(proposed_res["attribution"], "proposed_work", "85.42% should be attributed to proposed_work")
        self.assertEqual(proposed_res["status"], "confirmed")

    def test_12_llm_prompt_grounding_integration(self):
        """
        Verify that new_paper_facts are passed into LLM prompt payloads
        in review_paper.py without making any real API calls.
        """
        from unittest.mock import MagicMock
        from review_paper import summarize_facts_for_llm, extract_research_profile, analyze_comparison

        sample_facts = {
            "algorithms": [{"name": "YOLOv8n", "family": "YOLO", "variant": "YOLOv8n", "category": "detection", "page": 1}],
            "datasets": [{"name": "NSL-KDD", "description": "Benchmark dataset", "page": 2}],
            "metrics_and_results": [{"metric_name": "accuracy", "raw_value": "85.42", "unit": "%", "attribution": "proposed_work", "status": "confirmed", "raw_text": "accuracy of 85.42%"}],
            "experimental_setup": {"hardware": [{"name": "GPU"}], "software_and_frameworks": [{"name": "TensorFlow"}], "hyperparameters": []},
            "contributions": [{"statement": "Proposed stacked NDAE"}],
            "limitations": []
        }

        # 1. Verify summarize_facts_for_llm
        summary = summarize_facts_for_llm(sample_facts)
        self.assertEqual(len(summary["algorithms"]), 1)
        self.assertEqual(summary["algorithms"][0]["variant"], "YOLOv8n")
        self.assertEqual(summary["metrics_and_results"][0]["attribution"], "proposed_work")

        # 2. Verify extract_research_profile payload grounding
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = '{"title": "Test Paper", "research_area": "Security"}'
        mock_client.responses.create.return_value = mock_response

        profile = extract_research_profile(
            mock_client,
            "Test Title",
            "Test Abstract",
            "Test Introduction",
            technical_facts=sample_facts
        )
        self.assertEqual(profile["title"], "Test Paper")
        # Check that responses.create was called with deterministic_technical_facts in input JSON
        call_kwargs = mock_client.responses.create.call_args[1]
        input_payload = call_kwargs["input"]
        self.assertIn("deterministic_technical_facts", input_payload)
        self.assertIn("YOLOv8n", input_payload)

        # 3. Verify analyze_comparison payload grounding
        mock_comp_response = MagicMock()
        mock_comp_response.output_text = "# Comparative Analysis Report"
        mock_client.responses.create.return_value = mock_comp_response

        selected_papers = [{"paper_id": "P001", "title": "Prior Paper", "evidence": "Some evidence"}]
        comp_report = analyze_comparison(
            mock_client,
            profile,
            selected_papers,
            technical_facts=sample_facts
        )
        self.assertEqual(comp_report, "# Comparative Analysis Report")
        call_kwargs = mock_client.responses.create.call_args[1]
        comp_payload = call_kwargs["input"]
        self.assertIn("new_paper_verified_facts", comp_payload)
        self.assertIn("proposed_work", comp_payload)

    def test_13_vit_acronym_matching_and_vital_exclusion(self):
        """
        Verify that common words like 'vital' or 'vitamin' do NOT get extracted as ViT,
        while genuine Vision Transformer mentions (ViT, ViT-B/16, Vision Transformer) are detected.
        """
        # Case A: False-positive trap with 'vital'
        fp_text = (
            "Deep reinforcement learning plays a vital role in network security and cyber defense. "
            "It is vital to monitor all network packets to prevent inevitable attacks."
        )
        facts_fp = extract_technical_facts(full_text=fp_text)
        alg_names_fp = [a.get("variant") or a.get("name") for a in facts_fp["algorithms"]]
        self.assertNotIn("ViT", alg_names_fp, "'vital' must NOT be extracted as ViT")
        self.assertNotIn("vital", [a.get("name", "").lower() for a in facts_fp["algorithms"]])
        vit_matches = [a for a in facts_fp["algorithms"] if a.get("family") == "Transformer"]
        self.assertEqual(len(vit_matches), 0, "No Transformer should be extracted from 'vital' text")

        # Case B: Genuine ViT and Vision Transformer mentions
        tp_text = (
            "We evaluate ViT-B/16 and the Vision Transformer architecture on ImageNet. "
            "Additionally, Swin Transformer is used as a baseline backbone."
        )
        facts_tp = extract_technical_facts(full_text=tp_text)
        transformer_algs = [a for a in facts_tp["algorithms"] if a.get("family") == "Transformer"]
        variants = [a.get("variant") for a in transformer_algs]
        self.assertIn("ViT-B/16", variants, "ViT-B/16 must be extracted")
        self.assertIn("Vision Transformer", variants, "Vision Transformer must be extracted")
        self.assertTrue(any("Swin" in (v or "") for v in variants), "Swin Transformer must be extracted")

    def test_14_lr_hyperparameter_vs_logistic_regression_result(self):
        """
        Verify that classifier result labels like 'LR: 93.13%' are NOT extracted as learning rates,
        while genuine learning rate hyperparameters are correctly captured.
        """
        # Case A: Invalid - Logistic Regression accuracy result
        res_text = (
            "Our experimental comparison shows: DT: 99.93% , KNN: 99.93% , LR: 93.13% , SVM: 90.76% and NB: 79.52%."
        )
        facts_res = extract_technical_facts(full_text=res_text)
        hp_res = facts_res["experimental_setup"]["hyperparameters"]
        lr_hp_res = [h for h in hp_res if h.get("name") == "learning_rate"]
        self.assertEqual(len(lr_hp_res), 0, "LR: 93.13% must NOT be extracted as learning rate hyperparameter")

        # Also test with percentage word
        res_text_word = "Logistic Regression (LR) achieved accuracy of 93.13 percent."
        facts_res_word = extract_technical_facts(full_text=res_text_word)
        lr_hp_word = [h for h in facts_res_word["experimental_setup"]["hyperparameters"] if h.get("name") == "learning_rate"]
        self.assertEqual(len(lr_hp_word), 0, "LR accuracy percent must NOT be extracted as learning rate")

        # Case B: Valid - genuine learning rate configurations
        hp_text = (
            "Models were optimized using Adam with learning rate = 0.001 and batch size 64. "
            "For fine-tuning, we used lr=0.0001 and base learning rate of 1e-4."
        )
        facts_hp = extract_technical_facts(full_text=hp_text)
        hp_valid = facts_hp["experimental_setup"]["hyperparameters"]
        lr_valid_vals = [h.get("value") for h in hp_valid if h.get("name") == "learning_rate"]
        self.assertIn("0.001", lr_valid_vals, "learning rate = 0.001 must be extracted")
        self.assertIn("0.0001", lr_valid_vals, "lr=0.0001 must be extracted")
        self.assertIn("1e-4", lr_valid_vals, "base learning rate of 1e-4 must be extracted")

        # Context with explicit LR = 0.01 hyperparameter
        hp_text_lr_caps = "Training hyperparameters: LR = 0.01, epochs = 100."
        facts_caps = extract_technical_facts(full_text=hp_text_lr_caps)
        caps_vals = [h.get("value") for h in facts_caps["experimental_setup"]["hyperparameters"] if h.get("name") == "learning_rate"]
        self.assertIn("0.01", caps_vals, "LR = 0.01 should be accepted as learning rate")

    def test_15_game_playing_rl_algorithms(self):
        """
        Verify that AlphaZero, AlphaGo Zero, AlphaGo, and MuZero are recognized
        under the Reinforcement Learning family without generating broad false positives.
        """
        text = (
            "In this work, we study AlphaZero: a general reinforcement learning algorithm that masters chess, shogi, "
            "and Go through self-play. It eliminates domain-specific human knowledge from AlphaGo Zero. "
            "Furthermore, MuZero extends this framework to visual domains without knowing game dynamics."
        )
        facts = extract_technical_facts(full_text=text)
        rl_algs = [a for a in facts["algorithms"] if a.get("family") == "Reinforcement Learning"]
        variants = [a.get("variant") for a in rl_algs]

        self.assertIn("AlphaZero", variants, "AlphaZero must be recognized as RL variant")
        self.assertIn("AlphaGo Zero", variants, "AlphaGo Zero must be recognized as RL variant")
        self.assertIn("MuZero", variants, "MuZero must be recognized as RL variant")

        # Broad false-positive check on common non-algorithm phrases
        non_alg_text = "The system achieved zero errors and alpha parameter of 0.5 in round zero."
        facts_non_alg = extract_technical_facts(full_text=non_alg_text)
        non_alg_rl = [a for a in facts_non_alg["algorithms"] if a.get("family") == "Reinforcement Learning"]
        self.assertEqual(len(non_alg_rl), 0, "No RL algorithms should match generic 'zero' or 'alpha'")


if __name__ == "__main__":
    unittest.main(verbosity=2)

