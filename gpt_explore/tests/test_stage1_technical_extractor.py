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
    classify_algorithm_usage,
    extract_techniques,
    extract_datasets,
    is_valid_custom_dataset,
    extract_experimental_setup,
    extract_metrics_and_results,
    extract_contributions,
    extract_limitations,
    extract_ambiguities_and_warnings
)
from review_paper import (
    is_bibliography_chunk,
    group_and_rank_papers,
    extract_candidate_facts,
    enrich_candidate_papers_with_facts,
    generate_local_relevance_assessment,
    generate_local_comparative_review,
    format_local_comparative_review_markdown
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

    def test_16_reject_formula_constant_fscore_equation(self):
        """
        Verify that formula constants and metric definitions such as
        'F-score = 2 · Precision · Recall / (Precision + Recall)'
        are NOT extracted as experimental results, while genuine metrics are preserved.
        """
        # Case A: Metric definition formula with constant 2
        formula_text = (
            "We evaluate our system using standard metrics:\n"
            "Accuracy = TP + TN / (TP + TN + FP + FN) (6)\n"
            "Precision = TP / (TP + FP) (7)\n"
            "Recall = TP / (TP + FN) (8)\n"
            "F-score = 2 · Precision · Recall / (Precision + Recall) (10)\n"
            "The F-score measures the harmonic mean of precision and recall."
        )
        facts_formula = extract_technical_facts(full_text=formula_text)
        results = facts_formula["metrics_and_results"]

        # Ensure constant 2 is not extracted as f1-score
        f1_constant_2 = next((r for r in results if r["metric_name"] == "f1-score" and r["raw_value"] == "2"), None)
        self.assertIsNone(f1_constant_2, "Formula constant 2 must NOT be extracted as an F1-score result")

        # Case B: Genuine reported F1-score values
        genuine_text = (
            "Our model has produced superior results by offering f-score of 87.37% and recall of 85.42%. "
            "On test set B, the model achieves an F1-score of 0.89."
        )
        facts_genuine = extract_technical_facts(full_text=genuine_text)
        genuine_results = facts_genuine["metrics_and_results"]

        f1_87 = next((r for r in genuine_results if r["metric_name"] == "f1-score" and r["raw_value"] == "87.37"), None)
        self.assertIsNotNone(f1_87, "Genuine f-score 87.37% must be extracted")
        self.assertEqual(f1_87["attribution"], "proposed_work")

        f1_089 = next((r for r in genuine_results if r["metric_name"] == "f1-score" and r["raw_value"] == "0.89"), None)
        self.assertIsNotNone(f1_089, "Genuine F1-score 0.89 must be extracted")

    def test_17_section_heading_identification(self):
        """
        Verify that multi-line headings (section number on line 1, heading title on line 2)
        and expanded heading variants are recognized rather than defaulting to 'unknown'.
        """
        pages = [
            {
                "page": 1,
                "text": "1\nINTRODUCTION\nNetwork Intrusion Detection Systems play a vital role."
            },
            {
                "page": 2,
                "text": "3\nEXISTING WORK\nSeveral existing works study shallow classifiers."
            },
            {
                "page": 4,
                "text": "4\nPROPOSED METHODOLOGY\nOur NDAE model achieves superior dimensionality reduction."
            },
            {
                "page": 6,
                "text": "5\nEVALUATION & RESULTS\nWe evaluate our proposed solution on benchmark data.\n5.1\nDatasets\nEvaluated on benchmark KDD Cup '99 and NSL-KDD."
            },
            {
                "page": 8,
                "text": "6\nDISCUSSION\nOur evaluations show that our proposed solution achieves 85.42% accuracy."
            },
            {
                "page": 9,
                "text": "7\nCONCLUSION & FUTURE WORK\nIn conclusion, our work succeeds.\nREFERENCES\n[1] Reference one."
            }
        ]
        from technical_extractor import parse_paper_sections
        segments = parse_paper_sections(pages=pages)

        sec_by_page = {s["page"]: s["section"] for s in segments}
        self.assertEqual(sec_by_page[1], "introduction")
        self.assertEqual(sec_by_page[2], "related_work")
        self.assertEqual(sec_by_page[4], "methodology")

        # Page 6 contains results and datasets
        p6_sections = [s["section"] for s in segments if s["page"] == 6]
        self.assertIn("results", p6_sections)
        self.assertIn("dataset", p6_sections)

        self.assertEqual(sec_by_page[8], "discussion")

    def test_18_local_research_profile_generation(self):
        """
        Verify that generate_local_research_profile produces a structured,
        evidence-grounded profile without LLM calls, populating key results and schema fields.
        """
        from review_paper import generate_local_research_profile

        sample_facts = {
            "algorithms": [{"name": "NDAE", "family": "Autoencoder", "variant": "NDAE", "category": "deep_learning"}],
            "techniques_and_architectures": [{"name": "deep learning", "category": "deep_learning"}],
            "datasets": [{"name": "NSL-KDD", "description": "Benchmark dataset"}],
            "metrics_mentioned": ["accuracy", "recall", "precision", "f1-score"],
            "metrics_and_results": [
                {
                    "metric_name": "accuracy",
                    "raw_value": "85.42",
                    "unit": "%",
                    "status": "confirmed",
                    "attribution": "proposed_work",
                    "page": 8
                }
            ],
            "contributions": [{"statement": "Proposed stacked NDAE architecture"}],
            "limitations": [{"statement": "Evaluated only on offline benchmarks"}],
            "ambiguities_and_warnings": []
        }

        profile = generate_local_research_profile(
            title_candidate="A Deep Learning Approach to Network Intrusion Detection",
            abstract="Network Intrusion Detection Systems play a crucial role in defending networks.",
            introduction="Deep learning is garnering significant interest.",
            technical_facts=sample_facts
        )

        self.assertIsNotNone(profile)
        self.assertIn("title", profile)
        self.assertEqual(profile["title"], "A Deep Learning Approach to Network Intrusion Detection")
        self.assertIn("NSL-KDD", profile["data_and_dataset"])
        self.assertIn("NDAE", profile["proposed_methodology"])
        self.assertIn("accuracy: 85.42%", profile["key_results"])
        self.assertEqual(len(profile["limitations_stated"]), 1)
        self.assertEqual(profile["limitations_stated"][0], "Evaluated only on offline benchmarks")

    def test_19_self_retrieval_exclusion(self):
        """
        Verify that the input paper is excluded from the comparative candidate shortlist
        using its paper ID, preserving the retrieval index and model.
        """
        from review_paper import group_and_rank_papers, identify_source_paper_id
        from pathlib import Path

        # 1. Test source paper identification
        source_id = identify_source_paper_id(
            Path("dataset/papers/PAPER_033_A Deep Learning Approach to Network Intrusion Detection.pdf"),
            file_hash="dummyhash"
        )
        self.assertEqual(source_id, "PAPER_033")

        # 2. Test candidate exclusion in group_and_rank_papers
        simulated_chunks = [
            {"paper_id": "PAPER_033", "title": "Paper 033", "chunk_id": "c1", "section_type": "text", "original_heading": "", "text": "Self chunk", "distance": 0.05},
            {"paper_id": "PAPER_022", "title": "Paper 022", "chunk_id": "c2", "section_type": "text", "original_heading": "", "text": "Other chunk", "distance": 0.25},
            {"paper_id": "PAPER_081", "title": "Paper 081", "chunk_id": "c3", "section_type": "text", "original_heading": "", "text": "RL chunk", "distance": 0.35}
        ]

        title_lookup = {"PAPER_033": "Paper 033", "PAPER_022": "Paper 022", "PAPER_081": "Paper 081"}
        candidates = group_and_rank_papers(simulated_chunks, title_lookup, exclude_paper_id="PAPER_033")

        cand_ids = [c["paper_id"] for c in candidates]
        self.assertNotIn("PAPER_033", cand_ids, "Input paper PAPER_033 must NOT be included in comparative candidates")
        self.assertIn("PAPER_022", cand_ids)
        self.assertIn("PAPER_081", cand_ids)

    def test_20_generic_dataset_phrase_rejection(self):
        """
        Verify that generic quantifier phrases such as 'a limited amount of datasets',
        'several datasets', 'multiple datasets', 'various datasets', and 'a few datasets'
        are rejected, while genuine custom datasets remain extractable.
        """
        # 1. Direct validation helper tests
        generic_phrases = [
            "a limited amount of datasets",
            "a limited amount of dataset",
            "several datasets",
            "several dataset",
            "multiple datasets",
            "multiple dataset",
            "various datasets",
            "various dataset",
            "a few datasets",
            "a few dataset",
            "different datasets",
            "unseen datasets",
            "dataset",
            "datasets"
        ]
        for phrase in generic_phrases:
            self.assertFalse(
                is_valid_custom_dataset(phrase),
                f"Generic phrase '{phrase}' must be rejected by is_valid_custom_dataset"
            )

        genuine_names = [
            "wild and non-wild life annotated images",
            "10,000 chest X-ray images",
            "clinical trial patient records",
            "real-world traffic audio recordings"
        ]
        for name in genuine_names:
            self.assertTrue(
                is_valid_custom_dataset(name),
                f"Genuine custom dataset '{name}' must be accepted by is_valid_custom_dataset"
            )

        # 2. End-to-end extraction tests
        text = (
            "The proposed method was evaluated on a limited amount of datasets and several datasets. "
            "In practice, models trained on multiple datasets often fail on various datasets. "
            "However, when evaluated on wild and non-wild life annotated images, the model showed high accuracy."
        )
        segments = [{"text": text, "page": 1, "section": "evaluation"}]
        extracted = extract_datasets(segments)
        extracted_names = [d["name"] for d in extracted]

        # Verify none of the generic phrases were extracted
        for generic in ["a limited amount of dataset", "a limited amount of datasets", "several datasets", "multiple datasets", "various datasets"]:
            self.assertNotIn(generic, extracted_names)

        # Verify genuine dataset is extracted
        self.assertTrue(
            any("wild and non-wild life annotated images" in name for name in extracted_names),
            "Genuine custom dataset must be extracted"
        )

    def test_21_bibliography_chunk_exclusion_and_related_work_preservation(self):
        """
        Verify that bibliography chunks are excluded while legitimate related-work
        prose that cites papers is preserved.
        """
        # Case A: Chunk with references section metadata
        bib_chunk_meta = {
            "paper_id": "PAPER_022",
            "section_type": "references",
            "original_heading": "References",
            "text": "[1] J. Smith. Title. IEEE Trans, 2020.",
            "distance": 0.3
        }
        self.assertTrue(is_bibliography_chunk(bib_chunk_meta), "Metadata-marked references chunk must be identified as bibliography")

        # Case B: Chunk with bibliography text fallback (dense citations, no narrative prose)
        bib_chunk_text = {
            "paper_id": "PAPER_022",
            "section_type": "unknown",
            "original_heading": "",
            "text": (
                "[46] Soroush Fatemifar et al. 2019. Spoofing attack detection. In ICASSP. IEEE, 8464-8468.\n"
                "[47] Kamran Ghasedi et al. 2017. Deep clustering via joint autoencoders. In ICCV. 5736-5745.\n"
                "[48] Izhak Golan and Ran El-Yaniv. 2018. Deep anomaly detection. In NeurIPS. 445-455."
            ),
            "distance": 0.4
        }
        self.assertTrue(is_bibliography_chunk(bib_chunk_text), "Dense citation reference list must be identified as bibliography")

        # Case C: Legitimate related work prose that cites papers
        related_work_chunk = {
            "paper_id": "PAPER_022",
            "section_type": "related_work",
            "original_heading": "2. Related Work",
            "text": (
                "In related work, several deep learning methods have been proposed for anomaly detection. "
                "Smith et al. [1] proposed an autoencoder-based model. Later, Wang et al. [2] demonstrated "
                "that adversarial training improves detection accuracy on NSL-KDD. In this paper, our approach "
                "differs by incorporating self-supervised representations."
            ),
            "distance": 0.35
        }
        self.assertFalse(is_bibliography_chunk(related_work_chunk), "Legitimate related-work prose citing papers must NOT be flagged as bibliography")

        # Case D: group_and_rank_papers excludes bibliography chunk and keeps related work
        chunks = [bib_chunk_meta, bib_chunk_text, related_work_chunk]
        ranked = group_and_rank_papers(chunks, {"PAPER_022": "Deep Learning for Anomaly Detection"})
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["paper_id"], "PAPER_022")
        self.assertEqual(len(ranked[0]["evidence"]), 1)
        self.assertIn("In related work", ranked[0]["evidence"][0]["text"])

    def test_22_candidate_count_and_source_paper_exclusion(self):
        """
        Verify that group_and_rank_papers continues returning up to 5 eligible candidate papers
        when available and strictly excludes the source input paper.
        """
        chunks = []
        # Create chunks for 7 papers, including source paper PAPER_033 and bibliography chunks
        for i in range(1, 8):
            pid = f"PAPER_0{i:02d}" if i != 33 else "PAPER_033"
            # Add a bibliography chunk
            chunks.append({
                "paper_id": pid,
                "title": f"Paper {pid}",
                "chunk_id": f"{pid}_CHUNK_001",
                "section_type": "references",
                "original_heading": "References",
                "text": "[1] Author. Title. 2020.",
                "distance": 0.1 * i
            })
            # Add a legitimate technical chunk
            chunks.append({
                "paper_id": pid,
                "title": f"Paper {pid}",
                "chunk_id": f"{pid}_CHUNK_002",
                "section_type": "methodology",
                "original_heading": "Methodology",
                "text": f"This paper {pid} proposes a deep neural architecture evaluated on benchmark datasets.",
                "distance": 0.1 * i + 0.05
            })

        title_lookup = {f"PAPER_0{i:02d}": f"Title {i}" for i in range(1, 8)}
        ranked = group_and_rank_papers(chunks, title_lookup, exclude_paper_id="PAPER_033")

        cand_ids = [p["paper_id"] for p in ranked]
        self.assertNotIn("PAPER_033", cand_ids, "Source paper must be excluded")
        self.assertLessEqual(len(ranked[:5]), 5, "Candidate list must return up to 5 papers")
        self.assertTrue(len(ranked) >= 5, "At least 5 eligible candidate papers must be returned when available")

    def test_23_candidate_facts_isolation_and_provenance(self):
        """
        Verify candidate facts remain strictly associated with the candidate paper,
        never bleed into input paper facts, and candidate metrics are labeled with candidate attribution.
        """
        candidate = {
            "paper_id": "PAPER_022",
            "title": "Deep Learning for Anomaly Detection",
            "evidence": [
                {
                    "chunk_id": "PAPER_022_CHUNK_005",
                    "text": "The Autoencoder model achieved Accuracy: 94.2% on NSL-KDD."
                }
            ]
        }
        facts = extract_candidate_facts(candidate)
        self.assertEqual(facts["paper_id"], "PAPER_022")
        self.assertIn(facts["evidence_provenance"], ("full_paper_text", "full_paper_sections", "retrieved_chunks_partial"))

        # Verify candidate metrics are isolated and not marked as input paper proposed_work
        for m in facts.get("metrics_and_results", []):
            self.assertEqual(m["provenance_paper_id"], "PAPER_022")
            self.assertNotEqual(m.get("attribution"), "proposed_work", "Candidate metric attribution must never be conflated with input paper proposed_work")

    def test_24_partial_evidence_identification(self):
        """
        Verify that candidate papers without full text are extracted from evidence chunks
        and clearly marked as partial evidence.
        """
        candidate = {
            "paper_id": "PAPER_NONEXISTENT_999",
            "title": "Hypothetical External Paper",
            "evidence": [
                {
                    "chunk_id": "CHUNK_001",
                    "text": "The ResNet model achieved an accuracy of 88.5%."
                }
            ]
        }
        facts = extract_candidate_facts(candidate)
        self.assertEqual(facts["evidence_provenance"], "retrieved_chunks_partial")
        self.assertEqual(facts["evidence_scope"], "partial_evidence")
        self.assertEqual(facts["paper_id"], "PAPER_NONEXISTENT_999")
        self.assertTrue(any(a["family"] == "ResNet" for a in facts["algorithms"]))

    def test_25_metric_comparability_rules_and_differentiation(self):
        """
        Verify strict metric comparability rules:
        1. Accuracy is not compared directly with F1-score.
        2. Methods on different datasets are not ranked directly.
        3. Missing candidate metrics remain unknown rather than fabricated.
        """
        profile = {
            "title": "Input Paper",
            "research_area": "Network Intrusion Detection",
            "research_problem": "Intrusion detection",
            "proposed_methodology": "CNN",
            "data_and_dataset": "NSL-KDD"
        }
        input_facts = {
            "algorithms": [{"name": "CNN", "family": "CNN"}],
            "datasets": [{"name": "NSL-KDD"}],
            "experimental_setup": {"hardware": [], "software_and_frameworks": [], "hyperparameters": []},
            "metrics_and_results": [
                {
                    "metric_name": "F1-score",
                    "raw_value": "91.2",
                    "unit": "%",
                    "status": "confirmed",
                    "attribution": "proposed_work",
                    "dataset": "NSL-KDD"
                }
            ]
        }

        # Candidate A: Evaluated on same dataset but reports Accuracy (incompatible metric types)
        cand_a = {
            "paper_id": "PAPER_A",
            "title": "Candidate A",
            "evidence": [{"chunk_id": "C1", "text": "Achieved accuracy 93.5% on NSL-KDD."}],
            "technical_facts": {
                "algorithms": [{"name": "SVM", "family": "SVM"}],
                "datasets": [{"name": "NSL-KDD"}],
                "experimental_setup": {"hardware": [], "software_and_frameworks": [], "hyperparameters": []},
                "metrics_and_results": [
                    {
                        "metric_name": "Accuracy",
                        "raw_value": "93.5",
                        "unit": "%",
                        "status": "confirmed",
                        "attribution": "candidate_proposed_work",
                        "dataset": "NSL-KDD"
                    }
                ],
                "evidence_scope": "full_evidence"
            }
        }

        # Candidate B: Evaluated on different dataset
        cand_b = {
            "paper_id": "PAPER_B",
            "title": "Candidate B",
            "evidence": [{"chunk_id": "C2", "text": "Achieved F1-score 89.0% on ImageNet."}],
            "technical_facts": {
                "algorithms": [{"name": "ResNet", "family": "ResNet"}],
                "datasets": [{"name": "ImageNet"}],
                "experimental_setup": {"hardware": [], "software_and_frameworks": [], "hyperparameters": []},
                "metrics_and_results": [
                    {
                        "metric_name": "F1-score",
                        "raw_value": "89.0",
                        "unit": "%",
                        "status": "confirmed",
                        "attribution": "candidate_proposed_work",
                        "dataset": "ImageNet"
                    }
                ],
                "evidence_scope": "full_evidence"
            }
        }

        # Candidate C: Missing quantitative metrics
        cand_c = {
            "paper_id": "PAPER_C",
            "title": "Candidate C",
            "evidence": [{"chunk_id": "C3", "text": "Survey on intrusion detection."}],
            "technical_facts": {
                "algorithms": [{"name": "Random Forest", "family": "Random Forest"}],
                "datasets": [{"name": "NSL-KDD"}],
                "experimental_setup": {"hardware": [], "software_and_frameworks": [], "hyperparameters": []},
                "metrics_and_results": [],
                "evidence_scope": "full_evidence"
            }
        }

        review = generate_local_comparative_review(
            profile=profile,
            technical_facts=input_facts,
            candidate_papers=[cand_a, cand_b, cand_c]
        )

        metrics_comp = {m["candidate_id"]: m for m in review["metrics_and_results_comparison"]}

        # Check Candidate A (Accuracy vs F1-score)
        self.assertFalse(metrics_comp["PAPER_A"]["comparable"], "Accuracy must NOT be directly compared with F1-score")
        self.assertIn("Incompatible metric types", metrics_comp["PAPER_A"]["comparability_reason"])

        # Check Candidate B (NSL-KDD vs ImageNet)
        self.assertFalse(metrics_comp["PAPER_B"]["comparable"], "Different datasets must NOT permit direct numerical ranking")
        self.assertIn("different datasets", metrics_comp["PAPER_B"]["comparability_reason"])

        # Check Candidate C (Missing metrics)
        self.assertFalse(metrics_comp["PAPER_C"]["comparable"], "Candidate with missing metrics must NOT be comparable")
        self.assertEqual(metrics_comp["PAPER_C"]["status"], "insufficient_evidence")
        self.assertIn("Missing candidate quantitative metrics", metrics_comp["PAPER_C"]["comparability_reason"])

    def test_26_deterministic_local_comparative_review_structure(self):
        """
        Verify generate_local_comparative_review produces structured output with all 7 dimensions
        and explicit supported/partial_evidence/insufficient_evidence labels.
        """
        profile = {
            "title": "Deep Intrusion Detection",
            "research_area": "Network Intrusion Detection & Cybersecurity",
            "research_problem": "Detecting network attacks using deep models",
            "proposed_methodology": "Autoencoder",
            "data_and_dataset": "NSL-KDD, KDD Cup '99"
        }
        facts = {
            "algorithms": [{"name": "Autoencoder", "family": "Autoencoder"}],
            "datasets": [{"name": "NSL-KDD"}, {"name": "KDD Cup '99"}],
            "experimental_setup": {
                "hardware": [{"name": "NVIDIA GPU"}],
                "software_and_frameworks": [{"name": "PyTorch"}],
                "hyperparameters": [{"name": "lr", "value": "0.001"}]
            },
            "metrics_and_results": [
                {
                    "metric_name": "Accuracy",
                    "raw_value": "95.2",
                    "unit": "%",
                    "status": "confirmed",
                    "attribution": "proposed_work"
                }
            ]
        }
        cand = {
            "paper_id": "PAPER_022",
            "title": "Deep Learning for Anomaly Detection: A Review",
            "evidence": [
                {
                    "chunk_id": "PAPER_022_CHUNK_005",
                    "text": "Autoencoders are widely used for anomaly detection."
                }
            ],
            "technical_facts": {
                "algorithms": [{"name": "Autoencoder", "family": "Autoencoder"}, {"name": "SVM", "family": "SVM"}],
                "datasets": [{"name": "NSL-KDD"}],
                "experimental_setup": {"hardware": [], "software_and_frameworks": [], "hyperparameters": []},
                "metrics_and_results": [],
                "evidence_scope": "full_evidence"
            }
        }

        rel = generate_local_relevance_assessment(profile, [cand])
        self.assertEqual(len(rel["matches"]), 1)
        self.assertIn(rel["matches"][0]["relevance"], ("high", "medium"))

        comp = generate_local_comparative_review(profile, facts, [cand], rel)

        # Check required fields
        for field in [
            "status",
            "summary",
            "research_task_overlap",
            "methodological_comparison",
            "dataset_comparison",
            "experimental_setup_comparison",
            "metrics_and_results_comparison",
            "substantive_claims",
            "missing_information_and_uncertainties"
        ]:
            self.assertIn(field, comp, f"Field '{field}' must be present in comparative review")

        # Verify explicit status labels
        self.assertIn(comp["status"], ("supported", "partial_evidence", "insufficient_evidence"))
        self.assertIn(comp["research_task_overlap"][0]["status"], ("supported", "partial_evidence", "insufficient_evidence"))

        # Check Markdown formatting
        md = format_local_comparative_review_markdown(comp)
        self.assertIn("### 1. Research Task & Problem Overlap", md)
        self.assertIn("### 2. Methodological & Architectural Comparison", md)
        self.assertIn("### 3. Dataset & Benchmark Comparison", md)
        self.assertIn("### 4. Experimental Setup Differences", md)
        self.assertIn("### 5. Quantitative Results & Comparability Analysis", md)
        self.assertIn("### 6. Evidence-Supported Comparative Claims", md)
        self.assertIn("### 7. Missing Information & Uncertainties", md)

    def test_27_insufficient_evidence_handling(self):
        """
        Verify that when candidate papers are empty or evidence is unavailable,
        generate_local_comparative_review explicitly reports insufficient_evidence.
        """
        profile = {"title": "Paper", "research_area": "Area"}
        facts = {"algorithms": [], "datasets": [], "experimental_setup": {}, "metrics_and_results": []}
        comp = generate_local_comparative_review(profile, facts, [])

        self.assertEqual(comp["status"], "insufficient_evidence")
        self.assertIn("zero eligible candidate papers", comp["missing_information_and_uncertainties"][0])


    def test_28_algorithm_usage_proposed_method(self):
        """Verify an algorithm explicitly described as the proposed method is classified as proposed_method."""
        text = "In this paper, we propose a novel AlphaZero framework for mastering discrete board games through self-play."
        facts = extract_technical_facts(full_text=text)
        a = next((x for x in facts["algorithms"] if x.get("variant") == "AlphaZero" or x.get("name") == "AlphaZero" or x.get("family") == "AlphaZero"), None)
        self.assertIsNotNone(a, "AlphaZero must be extracted")
        self.assertEqual(a["usage"], "proposed_method")

    def test_29_algorithm_usage_experiment_or_baseline(self):
        """Verify an algorithm used as an experimental baseline is classified as experiment_or_baseline."""
        text = "We evaluate our proposed neural network against an SVM baseline model. The SVM was trained for comparison."
        facts = extract_technical_facts(full_text=text)
        a = next((x for x in facts["algorithms"] if x["family"] == "SVM"), None)
        self.assertIsNotNone(a, "SVM must be extracted")
        self.assertEqual(a["usage"], "experiment_or_baseline")

    def test_30_algorithm_usage_background_or_related_work(self):
        """Verify an algorithm mentioned only in related work or prior literature is classified as background_or_related_work."""
        text = "In related work, previous studies by Silver et al. proposed AlphaZero to learn chess and shogi from scratch."
        facts = extract_technical_facts(full_text=text)
        a = next((x for x in facts["algorithms"] if x.get("variant") == "AlphaZero" or x.get("name") == "AlphaZero" or x.get("family") == "AlphaZero"), None)
        self.assertIsNotNone(a, "AlphaZero must be extracted")
        self.assertEqual(a["usage"], "background_or_related_work")

    def test_31_algorithm_usage_references_only(self):
        """Verify an algorithm appearing only in references is not claimed as a method used by the paper."""
        text = "References\n[1] Silver, D., et al. Mastering Chess and Shogi with AlphaZero. Science, vol. 362, pp. 1140-1144, 2018."
        facts = extract_technical_facts(full_text=text)
        a = next((x for x in facts["algorithms"] if x.get("variant") == "AlphaZero" or x.get("name") == "AlphaZero" or x.get("family") == "AlphaZero"), None)
        if a is not None:
            self.assertEqual(a["usage"], "references_only")

        # Verify that comparative review never asserts an active method claim for references-only algorithms
        input_facts = {
            "algorithms": [{"name": "AlphaZero", "family": "AlphaZero", "usage": "proposed_method", "evidence": "We propose AlphaZero"}],
            "datasets": [],
            "experimental_setup": {},
            "metrics_and_results": []
        }
        cand = {
            "paper_id": "PAPER_REF",
            "title": "Unrelated Survey Paper",
            "evidence": [{"chunk_id": "c1", "text": text}],
            "technical_facts": facts
        }
        comp = generate_local_comparative_review({"title": "Input", "research_area": "RL"}, input_facts, [cand])
        shared_active = comp["methodological_comparison"][0]["shared_algorithms"]
        self.assertEqual(shared_active, [], "References-only mention must never become a shared active method")

    def test_32_algorithm_usage_uncertain(self):
        """Verify ambiguous or unknown contextual evidence yields uncertain usage."""
        text = "Notable architectures include ResNet."
        facts = extract_technical_facts(full_text=text)
        a = next((x for x in facts["algorithms"] if x["family"] == "ResNet"), None)
        self.assertIsNotNone(a, "ResNet must be extracted")
        self.assertEqual(a["usage"], "uncertain")

    def test_33_algorithm_overlap_alone_does_not_establish_task_overlap(self):
        """Verify that sharing algorithm mentions does not force research-task overlap."""
        profile = {
            "title": "A general reinforcement learning algorithm that masters chess, shogi, and Go",
            "research_area": "Game-Playing Artificial Intelligence",
            "research_problem": "Mastering chess, shogi, and Go without human domain knowledge",
            "proposed_methodology": "Self-play reinforcement learning with AlphaZero"
        }
        cand = {
            "paper_id": "PAPER_067",
            "title": "Deep Reinforcement Learning for Autonomous Driving: A Survey",
            "evidence": [{"chunk_id": "c1", "text": "AlphaZero [65] proposed self-play models. In this survey we review deep reinforcement learning for autonomous driving vehicles."}],
            "technical_facts": {
                "algorithms": [{"name": "AlphaZero", "family": "AlphaZero", "usage": "background_or_related_work"}],
                "datasets": [],
                "experimental_setup": {},
                "metrics_and_results": []
            }
        }
        rel = generate_local_relevance_assessment(profile, [cand])
        match = rel["matches"][0]
        self.assertFalse(match["same_research_problem"], "Autonomous driving survey cannot share specific research problem with board games")
        self.assertIn("Survey/background source", match["reason"])

    def test_34_shared_method_claim_requires_actual_use_evidence_from_both_papers(self):
        """Verify a definitive shared-method claim requires actual-use evidence (proposed_method or experiment_or_baseline) from both papers."""
        profile = {"title": "Input", "research_area": "RL"}
        facts_input = {
            "algorithms": [{"name": "AlphaZero", "family": "AlphaZero", "usage": "proposed_method", "evidence": "We propose AlphaZero."}],
            "datasets": [],
            "experimental_setup": {},
            "metrics_and_results": []
        }
        cand = {
            "paper_id": "PAPER_067",
            "title": "Autonomous Driving Survey",
            "evidence": [{"chunk_id": "c1", "text": "AlphaZero was introduced by prior researchers."}],
            "technical_facts": {
                "algorithms": [{"name": "AlphaZero", "family": "AlphaZero", "usage": "background_or_related_work", "evidence": "AlphaZero was introduced by prior researchers."}],
                "datasets": [],
                "experimental_setup": {},
                "metrics_and_results": []
            }
        }
        comp = generate_local_comparative_review(profile, facts_input, [cand])
        m_comp = comp["methodological_comparison"][0]
        self.assertEqual(m_comp["shared_algorithms"], [], "Cannot claim shared active algorithm when candidate is only background mention")
        self.assertIn("AlphaZero", m_comp["background_algorithm_mentions"])

        # Check substantive claims
        claims = comp["substantive_claims"]
        active_claims = [c for c in claims if "actively employ AlphaZero" in c["claim"]]
        self.assertEqual(len(active_claims), 0, "No active employment claim should be emitted")
        bg_claims = [c for c in claims if "mentions AlphaZero in background or related work" in c["claim"]]
        self.assertEqual(len(bg_claims), 1, "Must emit background mention claim instead")

    def test_35_survey_mentioning_alphazero_not_alphazero_implementation(self):
        """Verify that a survey mentioning AlphaZero does not automatically become an AlphaZero implementation."""
        survey_text = (
            "Deep Reinforcement Learning for Autonomous Driving: A Survey\n"
            "AlphaZero [65], developed by the same team, proposed a general framework for self-play models. "
            "AlphaZero taught itself from scratch how to master chess, shogi, and Go."
        )
        cand_facts = extract_technical_facts(
            full_text=survey_text,
            paper_title="Deep Reinforcement Learning for Autonomous Driving: A Survey"
        )
        az = next((a for a in cand_facts["algorithms"] if a.get("variant") == "AlphaZero" or a.get("name") == "AlphaZero" or a.get("family") == "AlphaZero"), None)
        self.assertIsNotNone(az, "AlphaZero must be extracted from survey text")
        self.assertEqual(az["usage"], "background_or_related_work", "Survey mention of AlphaZero must be classified as background_or_related_work")

        cand = {
            "paper_id": "PAPER_067",
            "title": "Deep Reinforcement Learning for Autonomous Driving: A Survey",
            "evidence": [{"chunk_id": "c1", "text": survey_text}],
            "technical_facts": cand_facts
        }
        input_facts = {
            "algorithms": [{"name": "AlphaZero", "family": "AlphaZero", "usage": "proposed_method", "evidence": "We present AlphaZero."}],
            "datasets": [],
            "experimental_setup": {},
            "metrics_and_results": []
        }
        comp = generate_local_comparative_review(
            {"title": "Mastering Chess with AlphaZero", "research_area": "RL"},
            input_facts,
            [cand]
        )
        self.assertEqual(comp["methodological_comparison"][0]["shared_algorithms"], [])
        self.assertIn("AlphaZero", comp["methodological_comparison"][0]["background_algorithm_mentions"])
        survey_claims = [c for c in comp["substantive_claims"] if c.get("category") == "survey_context"]
        self.assertEqual(len(survey_claims), 1, "Must recognize survey paper role")


if __name__ == "__main__":
    unittest.main(verbosity=2)

