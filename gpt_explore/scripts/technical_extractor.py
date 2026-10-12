"""
technical_extractor.py
Deterministic, evidence-backed technical fact extraction for AI/ML research papers.

Extracts structured technical facts from paper text and page information without external LLM calls:
- Algorithms and specific variants (e.g., YOLO and YOLOv8n)
- Techniques, model architectures, and methodological approaches
- Dataset names and custom dataset descriptions
- Experimental setup (hardware, software, hyperparameters)
- Evaluation metric names and numerical results (with ambiguity detection)
- Contributions and stated limitations
- Evidence snippets, page numbers, and source sections
- Ambiguities and extraction warnings
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================
# SECTION PATTERNS
# ============================================================

SECTION_PATTERNS = {
    "abstract": [
        r"^\s*(?:abstract|summary)\b"
    ],
    "introduction": [
        r"^\s*(?:\d+[\.\s]+)?introduction\b",
        r"^\s*i\.\s+introduction\b"
    ],
    "related_work": [
        r"^\s*(?:\d+[\.\s]+)?(?:related\s+work|related\s+works|literature\s+review|literature\s+survey|background)\b",
        r"^\s*ii\.\s+related\s+work\b"
    ],
    "methodology": [
        r"^\s*(?:\d+[\.\s]+)?(?:methodology|method|methods|proposed\s+method|proposed\s+approach|materials?\s+and\s+methods?|system\s+design|architecture|system\s+model)\b",
        r"^\s*iii\.\s+(?:methodology|proposed\s+system)\b"
    ],
    "dataset": [
        r"^\s*(?:\d+[\.\s]+)?(?:datasets?|data\s+collection|data\s+description|data\s+source|experimental\s+data)\b"
    ],
    "experiments": [
        r"^\s*(?:\d+[\.\s]+)?(?:experiments?|experimental\s+setup|implementation(?:\s+details)?|evaluations?)\b",
        r"^\s*iv\.\s+(?:experimental\s+setup|experiments?)\b"
    ],
    "results": [
        r"^\s*(?:\d+[\.\s]+)?(?:results?|experimental\s+results?|results?\s+and\s+discussion|performance\s+analysis)\b",
        r"^\s*v\.\s+(?:experimental\s+results?|results?)\b"
    ],
    "discussion": [
        r"^\s*(?:\d+[\.\s]+)?discussion\b"
    ],
    "limitations": [
        r"^\s*(?:\d+[\.\s]+)?(?:limitations?|limitations?\s+and\s+future\s+work)\b"
    ],
    "conclusion": [
        r"^\s*(?:\d+[\.\s]+)?(?:conclusions?|conclusion\s+and\s+future\s+work|concluding\s+remarks)\b",
        r"^\s*vi\.\s+conclusions?\b"
    ],
    "references": [
        r"^\s*(?:\d+[\.\s]+|[IVXLCDM]+[\.\s]+)?(?:references?|bibliography)\b",
        r"\b(?:references?|bibliography)\s*\[1\]",
        r"^\s*\[?(?:references?|bibliography)\]?\s*$"
    ]
}

BIBLIOGRAPHY_SECTIONS = {"references", "bibliography"}


# ============================================================
# ALGORITHM & VARIANT CATALOG
# ============================================================

# Format: (family, variant_pattern, category, family_pattern)
ALGORITHM_CATALOG = [
    # YOLO family & variants
    {
        "family": "YOLO",
        "category": "object_detection",
        "family_pattern": r"\bYOLO\b",
        "variant_pattern": r"\b(YOLOv?(?:1[0-2]|[1-9])(?:[a-z0-9_\-]+)?|PP-YOLO(?:v[1-2])?|YOLOX|YOLOR|YOLO-World)\b"
    },
    # ResNet family & variants
    {
        "family": "ResNet",
        "category": "computer_vision",
        "family_pattern": r"\bResNet\b",
        "variant_pattern": r"\b(ResNet[-_]?(?:18|34|50|101|152)[a-z]?|ResNeXt(?:[-_]?\d+)?)\b"
    },
    # VGG family
    {
        "family": "VGG",
        "category": "computer_vision",
        "family_pattern": r"\bVGG\b",
        "variant_pattern": r"\b(VGG[-_]?(?:16|19)[a-z]?)\b"
    },
    # MobileNet family
    {
        "family": "MobileNet",
        "category": "lightweight_vision",
        "family_pattern": r"\bMobileNet\b",
        "variant_pattern": r"\b(MobileNet(?:[-_]?v[1-3])?(?:[-_]?(?:small|large))?)\b"
    },
    # EfficientNet family
    {
        "family": "EfficientNet",
        "category": "computer_vision",
        "family_pattern": r"\bEfficientNet\b",
        "variant_pattern": r"\b(EfficientNet(?:[-_]?B[0-7])?)\b"
    },
    # DenseNet family
    {
        "family": "DenseNet",
        "category": "computer_vision",
        "family_pattern": r"\bDenseNet\b",
        "variant_pattern": r"\b(DenseNet[-_]?(?:121|161|169|201))\b"
    },
    # R-CNN / Faster R-CNN / Mask R-CNN
    {
        "family": "R-CNN",
        "category": "object_detection",
        "family_pattern": r"\bR[-_]?CNN\b",
        "variant_pattern": r"\b(Faster\s+R[-_]?CNN|Mask\s+R[-_]?CNN|Fast\s+R[-_]?CNN)\b"
    },
    # Segmentation models: U-Net, nnU-Net
    {
        "family": "U-Net",
        "category": "image_segmentation",
        "family_pattern": r"\bU[-_]?Net\b",
        "variant_pattern": r"\b(nnU[-_]?Net|Attention\s+U[-_]?Net|3D\s+U[-_]?Net)\b"
    },
    # General CNN
    {
        "family": "CNN",
        "category": "deep_learning",
        "family_pattern": r"\b(CNN|CNNs|Convolutional\s+Neural\s+Networks?)\b",
        "variant_pattern": None
    },
    # Transformer & Vision Transformer
    {
        "family": "Transformer",
        "category": "deep_learning",
        "family_pattern": r"\b(Transformer|Transformers)\b",
        "variant_pattern": r"\b((?-i:ViT|VIT)(?:[-_][A-Za-z0-9/]+)?|(?i:Vision\s+Transformers?|Swin(?:[-_\s]?Transformer)?|DeiT))\b"
    },
    # BERT family
    {
        "family": "BERT",
        "category": "nlp",
        "family_pattern": r"\bBERT\b",
        "variant_pattern": r"\b(RoBERTa|DistilBERT|DeBERTa|BioBERT|SciBERT|ALBERT|ELECTRA|BERT[-_]?(?:base|large))\b"
    },
    # GPT family
    {
        "family": "GPT",
        "category": "llm",
        "family_pattern": r"\bGPT\b",
        "variant_pattern": r"\b(GPT[-_]?(?:4o[-_]?mini|4o|4|3\.5|3|2))\b"
    },
    # LLaMA family
    {
        "family": "LLaMA",
        "category": "llm",
        "family_pattern": r"\bLLaMA\b",
        "variant_pattern": r"\b(LLaMA[-_]?(?:3|2|1)?(?:[-_]?[0-9]+[bB])?)\b"
    },
    # RNN, LSTM, GRU
    {
        "family": "RNN",
        "category": "deep_learning",
        "family_pattern": r"\b(RNN|Recurrent\s+Neural\s+Networks?)\b",
        "variant_pattern": r"\b(BiLSTM|LSTM|BiGRU|GRU)\b"
    },
    # Traditional ML: SVM
    {
        "family": "SVM",
        "category": "machine_learning",
        "family_pattern": r"\b(SVM|Support\s+Vector\s+Machines?)\b",
        "variant_pattern": r"\b(Linear\s+SVM|RBF\s+SVM)\b"
    },
    # Random Forest
    {
        "family": "Random Forest",
        "category": "machine_learning",
        "family_pattern": r"\b(Random\s+Forests?|RF)\b",
        "variant_pattern": None
    },
    # Decision Tree
    {
        "family": "Decision Tree",
        "category": "machine_learning",
        "family_pattern": r"\b(Decision\s+Trees?|CART|ID3|C4\.5)\b",
        "variant_pattern": None
    },
    # Gradient Boosting
    {
        "family": "Gradient Boosting",
        "category": "machine_learning",
        "family_pattern": r"\b(Gradient\s+Boosting|GBM)\b",
        "variant_pattern": r"\b(XGBoost|LightGBM|CatBoost|AdaBoost)\b"
    },
    # Reinforcement Learning
    {
        "family": "Reinforcement Learning",
        "category": "reinforcement_learning",
        "family_pattern": r"\b(Reinforcement\s+Learning|RL)\b",
        "variant_pattern": r"\b(DQN|Double\s+DQN|PPO|DDPG|SAC|A2C|A3C|TD3|Q[-_]?learning|AlphaZero|AlphaGo\s+Zero|AlphaGo|MuZero)\b"
    }
]


# ============================================================
# TECHNIQUES & METHODOLOGICAL APPROACHES
# ============================================================

TECHNIQUE_PATTERNS = [
    ("edge computing", "system_paradigm", r"\bedge\s+computing\b"),
    ("IoT-based alerting", "system_paradigm", r"\bIoT[-_\s]based\s+(?:alerting|alerts?|notification|monitoring)\b"),
    ("object detection", "computer_vision", r"\bobject\s+detection\b"),
    ("semantic segmentation", "computer_vision", r"\bsemantic\s+segmentation\b"),
    ("instance segmentation", "computer_vision", r"\binstance\s+segmentation\b"),
    ("image classification", "computer_vision", r"\bimage\s+classification\b"),
    ("feature extraction", "methodology", r"\bfeature\s+extraction\b"),
    ("transfer learning", "training_methodology", r"\btransfer\s+learning\b"),
    ("fine-tuning", "training_methodology", r"\bfine[-_\s]tuning\b"),
    ("data augmentation", "data_preprocessing", r"\bdata\s+augmentation\b"),
    ("cross-validation", "evaluation_methodology", r"\b(?:k[-_\s]fold\s+)?cross[-_\s]validation\b"),
    ("semi-supervised learning", "learning_paradigm", r"\bsemi[-_\s]supervised\s+learning\b"),
    ("self-supervised learning", "learning_paradigm", r"\bself[-_\s]supervised\s+learning\b"),
    ("contrastive learning", "learning_paradigm", r"\bcontrastive\s+learning\b"),
    ("attention mechanism", "model_architecture", r"\battention\s+mechanisms?\b"),
    ("model compression", "efficiency", r"\bmodel\s+compression\b"),
    ("pruning", "efficiency", r"\b(?:network|weight|model)?\s*pruning\b"),
    ("quantization", "efficiency", r"\bquantization\b"),
    ("deep learning", "general_paradigm", r"\bdeep\s+learning\b"),
    ("edge intelligence", "system_paradigm", r"\bedge\s+intelligence\b"),
    ("real-time monitoring", "system_application", r"\breal[-_\s]time\s+monitoring\b"),
    ("retrieval-augmented generation", "nlp", r"\b(?:retrieval[-_\s]augmented\s+generation|RAG)\b"),
    ("in-context learning", "nlp", r"\bin[-_\s]context\s+learning\b")
]


# ============================================================
# DATASET CATALOG & PATTERNS
# ============================================================

STANDARD_DATASETS = [
    ("ImageNet", r"\bImageNet(?:[-_]?[0-9kK]+)?\b"),
    ("COCO", r"\b(?:MS\s+)?COCO\b"),
    ("Pascal VOC", r"\bPascal\s+VOC(?:\s*\d{4})?\b"),
    ("CIFAR-10", r"\bCIFAR[-_]?10\b"),
    ("CIFAR-100", r"\bCIFAR[-_]?100\b"),
    ("MNIST", r"\bMNIST\b"),
    ("Fashion-MNIST", r"\bFashion[-_]?MNIST\b"),
    ("Cityscapes", r"\bCityscapes\b"),
    ("KITTI", r"\bKITTI\b"),
    ("GLUE", r"\bGLUE\b"),
    ("SuperGLUE", r"\bSuperGLUE\b"),
    ("SQuAD", r"\bSQuAD(?:\s*[0-9\.]+)?\b"),
    ("Kaggle", r"\bKaggle(?:\s+dataset)?\b"),
    ("KDD Cup '99", r"\bKDD\s*Cup(?:\s*['’]?(?:99|1999))?\b|\bKDD[-_]?99\b"),
    ("NSL-KDD", r"\bNSL[-_\s]?KDD\b")
]

CUSTOM_DATASET_PATTERNS = [
    r"(?:trained\s+with|trained\s+on|evaluated\s+on|dataset\s+(?:of|with|containing|consisting\s+of)|collected\s+a\s+dataset\s+of)\s+([a-zA-Z0-9\s,\-–—]+?(?:annotated\s+images?|images?|samples?|recordings?|records?|photos?|dataset))",
    r"\b([a-zA-Z0-9\s\-]+?annotated\s+images)\b"
]


# ============================================================
# EXPERIMENTAL SETUP CATALOG
# ============================================================

HARDWARE_PATTERNS = [
    ("Raspberry Pi", r"\bRaspberry\s+Pi(?:\s*[0-9a-zA-Z]+)?\b"),
    ("camera", r"\b(?:digital\s+|RGB\s+|infrared\s+|IR\s+)?camera(?:s|\s+module)?\b"),
    ("motion sensor", r"\bmotion\s+sensors?\b"),
    ("camera trap", r"\bcamera\s+traps?\b"),
    ("edge device", r"\bedge\s+devices?\b"),
    ("Jetson", r"\b(?:NVIDIA\s+)?Jetson(?:\s+[a-zA-Z0-9]+)?\b"),
    ("GPU", r"\b(?:NVIDIA\s+)?(?:RTX\s*[0-9]+[a-zA-Z]*|GTX\s*[0-9]+[a-zA-Z]*|Tesla\s*[a-zA-Z0-9]+|A100|V100|T4|GPU|TPU)\b")
]

SOFTWARE_PATTERNS = [
    ("PyTorch", r"\bPyTorch\b"),
    ("TensorFlow", r"\bTensorFlow\b"),
    ("Keras", r"\bKeras\b"),
    ("OpenCV", r"\bOpenCV\b"),
    ("Scikit-learn", r"\bScikit[-_\s]learn\b"),
    ("Telegram alert", r"\bTelegram(?:[-_\s]based)?\s*(?:instant\s+)?(?:notification|alert|bot)?\b"),
    ("Docker", r"\bDocker\b"),
    ("Linux", r"\bLinux\b"),
    ("Ubuntu", r"\bUbuntu\b")
]

HYPERPARAMETER_PATTERNS = [
    ("learning_rate", r"\b(?:(?:initial\s+|base\s+)?learning[-_\s]?rate|(?-i:lr|LR(?=\s*=\s*[0-9])))\s*(?:=|:|is|of|set\s+to)?\s*([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)\b"),
    ("batch_size", r"\bbatch\s+size\s*(?:=|:|is|of)?\s*([0-9]+)\b"),
    ("epochs", r"\b([0-9]+)\s+epochs?\b"),
    ("optimizer", r"\boptimizer\s*(?:=|:|is|of)?\s*(AdamW|Adam|SGD|RMSprop)\b"),
    ("resolution", r"\b([0-9]{2,4}\s*[xX*]\s*[0-9]{2,4})\b")
]


# ============================================================
# METRICS DEFINITIONS
# ============================================================

METRIC_NAME_ALIASES = {
    "accuracy": ["accuracy", "acc", "classification accuracy", "detection accuracy", "top-1 accuracy", "top-5 accuracy"],
    "precision": ["precision", "average precision"],
    "recall": ["recall", "sensitivity"],
    "f1-score": ["f1-score", "f1 score", "f1", "f-score", "f1-measure"],
    "iou": ["iou", "intersection over union", "miou", "mean iou"],
    "map@0.5:0.95": ["map@0.5:0.95", "map@0.5-0.95", "map@.5:.95", "map@.5-.95", "map50-95", "map 0.5:0.95", "map@0.5 to 0.95"],
    "map@0.5": ["map@0.5", "map@.5", "map50", "map 0.5"],
    "map": ["mean average precision", "map"],
    "auc": ["auc", "roc-auc", "roc auc"]
}

# Inverted lookup for fast matching
CANONICAL_METRIC_MAP = {}
for canonical, aliases in METRIC_NAME_ALIASES.items():
    for alias in aliases:
        CANONICAL_METRIC_MAP[alias.lower()] = canonical


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    """Normalize whitespace and fix hyphenation split across lines."""
    if not text:
        return ""
    # Fix words split by hyphen and newline: e.g. "hu- \n man" -> "human"
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)
    # Fix numbers split across lines: e.g. "91.3 \n %" -> "91.3%"
    text = re.sub(r"(\d+(?:\.\d+)?)\s*\n\s*(%)", r"\1\2", text)
    # Normalize general whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_snippet(text: str, start: int, end: int, window: int = 100) -> str:
    """Extract context snippet around matching span."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].strip()
    # Normalize internal newlines to spaces for snippet display
    snippet = re.sub(r"\s+", " ", snippet)
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."
    return snippet


# ============================================================
# SECTION & PAGE PARSER
# ============================================================

def parse_paper_sections(
    pages: Optional[List[Dict[str, Any]]] = None,
    full_text: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Decompose paper pages/text into line-level or paragraph-level segments
    tagged with 1-based page number and detected section heading.
    Cleanly partitions and isolates bibliography/references sections.
    """
    if pages:
        page_items = pages
    elif full_text:
        page_items = [{"page": 1, "text": full_text}]
    else:
        return []

    segments = []
    current_section = "unknown"
    in_references = False

    for page_info in page_items:
        page_num = page_info.get("page", 1)
        raw_page_text = page_info.get("text", "")
        cleaned_page_text = clean_text(raw_page_text)

        # Break page into coherent sentences/paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw_page_text) if p.strip()]
        if not paragraphs:
            paragraphs = [cleaned_page_text] if cleaned_page_text else []

        for para in paragraphs:
            para_clean = clean_text(para)
            if not para_clean:
                continue

            # Check if this paragraph contains a section heading
            first_line = para_clean.split(".")[0][:80].strip()
            for sec_name, patterns in SECTION_PATTERNS.items():
                matched = False
                for pat in patterns:
                    if re.match(pat, first_line, re.IGNORECASE) or re.match(pat, para_clean[:80], re.IGNORECASE):
                        current_section = sec_name
                        if sec_name in BIBLIOGRAPHY_SECTIONS:
                            in_references = True
                        else:
                            in_references = False
                        matched = True
                        break
                if matched:
                    break

            # If not yet flagged as references, check if references header starts within this paragraph
            if not in_references:
                ref_match = re.search(
                    r"(?i)(?:^|\n|[.\s])(?:references?|bibliography)\s*(?:\[1\]|[:\-]|\n|$)",
                    para_clean
                )
                if ref_match:
                    split_idx = ref_match.start()
                    prefix_text = para_clean[:split_idx].strip()
                    ref_text = para_clean[split_idx:].strip()

                    if prefix_text:
                        segments.append({
                            "page": page_num,
                            "section": current_section,
                            "text": prefix_text
                        })

                    in_references = True
                    current_section = "references"
                    segments.append({
                        "page": page_num,
                        "section": "references",
                        "text": ref_text
                    })
                    continue

            if in_references:
                current_section = "references"

            segments.append({
                "page": page_num,
                "section": current_section,
                "text": para_clean
            })

    return segments


# ============================================================
# 1. ALGORITHMS & VARIANTS EXTRACTION
# ============================================================

def extract_algorithms(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract algorithm families and specific variants (e.g. YOLO vs YOLOv8n).
    Preserves evidence, page number, and section name.
    """
    results = []
    seen_on_page = set()

    for seg in segments:
        if seg.get("section", "").lower() in BIBLIOGRAPHY_SECTIONS:
            continue
        text = seg["text"]
        page = seg["page"]
        section = seg["section"]

        for item in ALGORITHM_CATALOG:
            family = item["family"]
            category = item["category"]
            family_pat = item["family_pattern"]
            variant_pat = item.get("variant_pattern")

            # Check for specific variant first
            if variant_pat:
                for match in re.finditer(variant_pat, text, re.IGNORECASE):
                    variant_name = match.group(1).strip()
                    # Filter out short or false positive matches
                    if len(variant_name) < 2:
                        continue

                    norm_key = variant_name.lower()
                    page_key = (norm_key, page)
                    if page_key not in seen_on_page:
                        seen_on_page.add(page_key)
                        snippet = extract_snippet(text, match.start(), match.end())
                        results.append({
                            "name": variant_name,
                            "family": family,
                            "variant": variant_name,
                            "category": category,
                            "evidence": snippet,
                            "page": page,
                            "section": section
                        })

            # Also check for general family mention
            for match in re.finditer(family_pat, text, re.IGNORECASE):
                fam_name = match.group(0).strip()
                norm_key = fam_name.lower()
                page_key = (norm_key, page)

                if page_key not in seen_on_page:
                    seen_on_page.add(page_key)
                    snippet = extract_snippet(text, match.start(), match.end())
                    results.append({
                        "name": family,
                        "family": family,
                        "variant": None,
                        "category": category,
                        "evidence": snippet,
                        "page": page,
                        "section": section
                    })

    return results


# ============================================================
# 2. TECHNIQUES & METHODOLOGICAL APPROACHES
# ============================================================

def extract_techniques(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract techniques, architectures, and approaches.
    """
    results = []
    seen_on_page = set()

    for seg in segments:
        if seg.get("section", "").lower() in BIBLIOGRAPHY_SECTIONS:
            continue
        text = seg["text"]
        page = seg["page"]
        section = seg["section"]

        for name, category, pattern in TECHNIQUE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                key = name.lower()
                page_key = (key, page)
                if page_key not in seen_on_page:
                    seen_on_page.add(page_key)
                    snippet = extract_snippet(text, match.start(), match.end())
                    results.append({
                        "name": name,
                        "category": category,
                        "evidence": snippet,
                        "page": page,
                        "section": section
                    })

    return results


# ============================================================
# 3. DATASETS & CUSTOM DESCRIPTIONS
# ============================================================

def extract_datasets(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract standard benchmark datasets and custom dataset descriptions.
    """
    results = []
    seen_on_page = set()

    for seg in segments:
        if seg.get("section", "").lower() in BIBLIOGRAPHY_SECTIONS:
            continue
        text = seg["text"]
        page = seg["page"]
        section = seg["section"]

        # 1. Standard benchmarks
        for name, pattern in STANDARD_DATASETS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                key = name.lower()
                page_key = (key, page)
                if page_key not in seen_on_page:
                    seen_on_page.add(page_key)
                    snippet = extract_snippet(text, match.start(), match.end())
                    results.append({
                        "name": name,
                        "description": f"Standard benchmark dataset: {name}",
                        "evidence": snippet,
                        "page": page,
                        "section": section
                    })

        # 2. Custom dataset descriptions
        for pat in CUSTOM_DATASET_PATTERNS:
            for match in re.finditer(pat, text, re.IGNORECASE):
                desc = match.group(1).strip()
                # Clean up punctuation and stop words
                desc = re.sub(r"^[—\-,\.]\s*", "", desc).strip()
                if len(desc) < 6 or len(desc) > 120:
                    continue

                key = desc.lower()
                page_key = (key, page)
                if page_key not in seen_on_page:
                    seen_on_page.add(page_key)
                    snippet = extract_snippet(text, match.start(), match.end())
                    results.append({
                        "name": desc,
                        "description": f"Custom dataset: {desc}",
                        "evidence": snippet,
                        "page": page,
                        "section": section
                    })

    return results


# ============================================================
# 4. EXPERIMENTAL SETUP & IMPLEMENTATION DETAILS
# ============================================================

def is_valid_learning_rate(match: re.Match, text: str, val_str: str) -> bool:
    """
    Validate that a matched learning rate candidate is a genuine hyperparameter
    and not a model result/metric (e.g. 'LR: 93.13%' where LR = Logistic Regression).
    """
    # 1. Reject if immediately followed by % or percent (percentages are evaluation metrics)
    post_text = text[match.end():match.end() + 15]
    if re.match(r"\s*(?:%|percent\b)", post_text, re.IGNORECASE):
        return False

    # 2. Parse float and verify reasonable magnitude (learning rates in ML are typically <= 1.0)
    try:
        val_f = float(val_str)
        if val_f <= 0 or val_f > 1.0:
            return False
    except (ValueError, TypeError):
        return False

    matched_span = match.group(0)
    # 3. If matched via abbreviation 'lr' or 'LR' rather than explicit 'learning rate'
    if not re.search(r"learning[-_\s]?rate", matched_span, re.IGNORECASE):
        pre_window = text[max(0, match.start() - 60):match.start()]
        post_window = text[match.end():min(len(text), match.end() + 60)]
        surrounding = pre_window + " " + post_window

        # Check for model / classifier abbreviations or metric labels
        classifier_cues = r"\b(?:logistic\s+regression|classifiers?|classification|accuracy|acc|precision|recall|f1|SVM|KNN|DT|NB|RF|MLP)\b"
        if re.search(classifier_cues, surrounding, re.IGNORECASE):
            if re.search(r"\bLR\b", matched_span):
                return False

    return True


def extract_experimental_setup(segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract hardware, software, frameworks, and hyperparameters.
    """
    hardware_list = []
    software_list = []
    hyperparams_list = []
    seen_hw_page = set()
    seen_sw_page = set()
    seen_hp_page = set()

    for seg in segments:
        if seg.get("section", "").lower() in BIBLIOGRAPHY_SECTIONS:
            continue
        text = seg["text"]
        page = seg["page"]
        section = seg["section"]

        # Hardware
        for name, pat in HARDWARE_PATTERNS:
            for match in re.finditer(pat, text, re.IGNORECASE):
                matched_str = match.group(0).strip()
                key = name.lower()
                page_key = (key, page)
                if page_key not in seen_hw_page:
                    seen_hw_page.add(page_key)
                    snippet = extract_snippet(text, match.start(), match.end())
                    hardware_list.append({
                        "name": name,
                        "detail": matched_str,
                        "evidence": snippet,
                        "page": page,
                        "section": section
                    })

        # Software
        for name, pat in SOFTWARE_PATTERNS:
            for match in re.finditer(pat, text, re.IGNORECASE):
                matched_str = match.group(0).strip()
                key = name.lower()
                page_key = (key, page)
                if page_key not in seen_sw_page:
                    seen_sw_page.add(page_key)
                    snippet = extract_snippet(text, match.start(), match.end())
                    software_list.append({
                        "name": name,
                        "detail": matched_str,
                        "evidence": snippet,
                        "page": page,
                        "section": section
                    })

        # Hyperparameters
        for name, pat in HYPERPARAMETER_PATTERNS:
            for match in re.finditer(pat, text, re.IGNORECASE):
                val = match.group(1).strip()
                if name == "learning_rate" and not is_valid_learning_rate(match, text, val):
                    continue
                key = f"{name}:{val}"
                page_key = (key, page)
                if page_key not in seen_hp_page:
                    seen_hp_page.add(page_key)
                    snippet = extract_snippet(text, match.start(), match.end())
                    hyperparams_list.append({
                        "name": name,
                        "value": val,
                        "evidence": snippet,
                        "page": page,
                        "section": section
                    })

    return {
        "hardware": hardware_list,
        "software_and_frameworks": software_list,
        "hyperparameters": hyperparams_list
    }


# ============================================================
# 5. METRIC-VALUE PAIRS & AMBIGUOUS RESULTS
# ============================================================

def resolve_metric_name(text_slice: str) -> Optional[str]:
    """Check if text contains any recognized evaluation metric."""
    lower = text_slice.lower()
    for alias, canonical in CANONICAL_METRIC_MAP.items():
        # Match as whole word or phrase
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, lower):
            return canonical
    return None


def is_metric_negated(sent_str: str, metric_alias: str, start: int = 0, end: int = 0) -> bool:
    """
    Check if a metric occurrence is negated or stated as unmeasured/unlisted.
    e.g., 'recall and precision results are not listed', 'accuracy was not reported'.
    """
    left = max(0, start - 80)
    right = min(len(sent_str), end + 80)
    window_text = sent_str[left:right]

    neg_pats = [
        r"\b(?:are|is|were|was)?\s*not\s+(?:listed|reported|provided|shown|given|measured|evaluated|available|disclosed|included|tested|published)\b",
        r"\b(?:no|without)\s+(?:explicit\s+)?(?:results?|data|values?|evaluation)\s+(?:are|is)?\s*(?:reported|listed|available|provided|given)?\b",
        r"\b(?:cannot|could\s+not)\s+be\s+(?:determined|measured|evaluated|found)\b",
        r"\bnot\s+stated\b",
        r"\bomitted\b"
    ]
    for np in neg_pats:
        if re.search(np, window_text, re.IGNORECASE):
            return True
    return False


def is_table_scrap_number(sent_str: str, match_start: int, val_raw: str, unit_raw: Optional[str], canonical: str) -> bool:
    """
    Detect if val_raw is actually the trailing cell of a tabular row
    rather than a legitimate metric quantifier.
    e.g., '... 89.22 90.76 10.78 recall and precision ...'
    """
    # 1. Preceded by another number (table column sequence)
    preceding_slice = sent_str[:match_start]
    if re.search(r"\b\d+(?:\.\d+)?\s+$", preceding_slice):
        return True

    # 2. Values > 1.0 without % symbol modifying accuracy, precision, recall, auc
    # without any connector noun (e.g. '10.78 recall' vs '0.87 recall' or '10.78% recall')
    try:
        val_float = float(val_raw)
        if unit_raw is None and val_float > 1.0 and canonical in ("accuracy", "precision", "recall", "auc", "roc-auc"):
            if not re.search(r"\b(?:rate|score|value|level)\b", sent_str[match_start:match_start + len(val_raw) + 25], re.IGNORECASE):
                return True
    except (ValueError, TypeError):
        pass

    return False


def determine_metric_attribution(
    sent_str: str,
    val_raw: str,
    canonical: Optional[str] = None,
    section: str = "unknown"
) -> str:
    """
    Determine whether a metric result is attributed to:
    - 'proposed_work': Authors' own model, proposed approach, or current paper results
    - 'prior_work': Baseline models, cited prior literature, or comparisons
    - 'uncertain': Ambiguous or unstated attribution
    """
    proposed_cues = [
        r"\b(?:our\s+(?:proposed\s+)?(?:model|approach|method|system|framework|work|architecture|results?|classifier|network|pipeline))\b",
        r"\b(?:we\s+(?:achieve|achieved|obtain|obtained|reach|reached|report|reported|propose|proposed|demonstrate|demonstrated|yield|yielded|managed))\b",
        r"\b(?:this\s+(?:paper|work|article|study)\s+(?:achieves?|achieved|proposes?|proposed|presents?|presented|yields?|demonstrates?))\b",
        r"\b(?:accomplished\s+by\s+our\s+model|our\s+achieved\s+accuracy|our\s+evaluations?\s+show|superior\s+results\s+by\s+offering)\b",
        r"\b(?:proposed\s+(?:model|approach|method|architecture)\s+(?:achieves?|achieved|reaches?|reached|yields?|yielded))\b"
    ]

    prior_cues = [
        r"(?:[A-Z][a-zA-Z]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z]+)?\s*)?\[\d+\]\s*(?:claim|claimed|achieve|achieved|report|reported|obtain|obtained|reach|reached|propose|proposed|show|showed|demonstrate|demonstrated)\b",
        r"(?:in\s+\[\d+\]|\[\d+\]\s*(?:claim|claimed|achieved?|reported?|showed?|obtain|obtained))\b",
        r"\b(?:authors?\s+in\s+\[\d+\]|authors?\s+claim|authors?\s+report)\b",
        r"\b(?:prior\s+work|previous\s+works?|earlier\s+studies?|existing\s+(?:methods?|models?|approaches?))\b",
        r"\b(?:baseline|compared\s+(?:with|to)\s+(?:the\s+work\s+in\s+)?\[\d+\])\b",
        r"\b(?:other\s+(?:deep\s+learning\s+)?methods\s+such\s+as\s+\[\d+\]|work\s+in\s+\[\d+\])\b",
        r"\b(?:state[-_\s]of[-_\s]the[-_\s]art\s+(?:by|in)\s+\[\d+\])\b"
    ]

    has_proposed = any(re.search(pat, sent_str, re.IGNORECASE) for pat in proposed_cues)
    has_prior = any(re.search(pat, sent_str, re.IGNORECASE) for pat in prior_cues)

    if has_proposed and not has_prior:
        return "proposed_work"
    if has_prior and not has_proposed:
        return "prior_work"
    if has_proposed and has_prior:
        # Comparative sentence: check proximity of val_raw to the nearest cue
        val_pos = sent_str.find(val_raw)
        if val_pos == -1:
            return "uncertain"

        min_prop_dist = float("inf")
        for pat in proposed_cues:
            for m in re.finditer(pat, sent_str, re.IGNORECASE):
                dist = min(abs(val_pos - m.start()), abs(val_pos - m.end()))
                if dist < min_prop_dist:
                    min_prop_dist = dist

        min_prior_dist = float("inf")
        for pat in prior_cues:
            for m in re.finditer(pat, sent_str, re.IGNORECASE):
                dist = min(abs(val_pos - m.start()), abs(val_pos - m.end()))
                if dist < min_prior_dist:
                    min_prior_dist = dist

        if min_prop_dist < min_prior_dist:
            return "proposed_work"
        elif min_prior_dist < min_prop_dist:
            return "prior_work"
        return "uncertain"

    # Default section context heuristics
    if section in ("results", "experiments", "abstract"):
        if re.search(r"\[\d+\]", sent_str):
            return "prior_work"
        return "proposed_work"
    elif section == "related_work":
        return "prior_work"

    return "uncertain"


def extract_metrics_and_results(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract metric-value pairs with contextual evidence.
    Supports common metrics: accuracy, precision, recall, F1-score, IoU, mAP, mAP@0.5, mAP@0.5:0.95.
    Preserves:
    - Raw numerical value and surrounding text.
    - Metric name ONLY when supported by text.
    - Unit or percentage ONLY when supported by text.
    - Page number and section.
    - Confidence/ambiguity status with brief explanation.
    - Attribution: proposed_work, prior_work, or uncertain.
    """
    results = []
    seen_keys = set()

    for seg in segments:
        if seg.get("section", "").lower() in BIBLIOGRAPHY_SECTIONS:
            continue
        text = seg["text"]
        page = seg["page"]
        section = seg["section"]

        # Sentence-by-sentence parsing to retain local context (rejoining on et al., e.g., i.e.)
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = []
        for s in raw_sentences:
            if sentences and re.search(r"\b(?:et\s+al|e\.g|i\.e|vs)\.$", sentences[-1].strip(), re.IGNORECASE):
                sentences[-1] = sentences[-1] + " " + s
            else:
                sentences.append(s)

        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            # ----------------------------------------------------
            # CASE A: Explicit Metric followed by Value
            # e.g., "accuracy of 94.5%", "mAP@0.5 of 0.87", "precision: 91.2%"
            # ----------------------------------------------------
            explicit_pattern_1 = (
                r"(?i)\b(accuracy|acc|precision|recall|f1?[-_\s]?score|f1|miou|iou|"
                r"intersection\s+over\s+union|map@0\.5[:\-]0\.95|map@\.5[:\-]\.95|"
                r"map50[-_:]95|map@0\.5|map@\.5|map50|mean\s+average\s+precision|map|"
                r"top[-_]?[15]\s+accuracy|auc|roc[-_]?auc)\b"
                r"(?:\s*(?:of|is|was|reached|achieved|at|:|=|≈|~)\s*|\s+)"
                r"(?:approx(?:imately)?\s*)?"
                r"([0-9]+(?:\.[0-9]+)?)"
                r"\s*(%|percent\b)?"
            )

            for match in re.finditer(explicit_pattern_1, sent_str):
                metric_raw = match.group(1).strip()
                val_raw = match.group(2).strip()
                unit_raw = "%" if match.group(3) else None

                canonical = CANONICAL_METRIC_MAP.get(metric_raw.lower(), metric_raw)

                # Check negation guard
                if is_metric_negated(sent_str, metric_raw, match.start(), match.end()):
                    continue

                snippet = extract_snippet(sent_str, match.start(), match.end(), window=80)
                attribution = determine_metric_attribution(sent_str, val_raw, canonical, section)

                key = (canonical, val_raw, unit_raw, page, attribution)
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append({
                        "raw_value": val_raw,
                        "raw_text": snippet,
                        "metric_name": canonical,
                        "unit": unit_raw,
                        "page": page,
                        "section": section,
                        "status": "confirmed",
                        "attribution": attribution,
                        "explanation": f"Explicit metric '{canonical}' detected with value {val_raw}{unit_raw or ''}."
                    })

            # ----------------------------------------------------
            # CASE B: Value followed by Metric
            # e.g., "94.5% accuracy", "0.85 mAP", "91.3% precision"
            # ----------------------------------------------------
            explicit_pattern_2 = (
                r"(?i)\b([0-9]+(?:\.[0-9]+)?)"
                r"\s*(%|percent\b)?\s+"
                r"(?:(?:of|in|for)\s+)?"
                r"\b(accuracy|acc|precision|recall|f1?[-_\s]?score|f1|miou|iou|"
                r"intersection\s+over\s+union|map@0\.5[:\-]0\.95|map@\.5[:\-]\.95|"
                r"map50[-_:]95|map@0\.5|map@\.5|map50|mean\s+average\s+precision|map|"
                r"top[-_]?[15]\s+accuracy|auc|roc[-_]?auc)\b"
            )

            for match in re.finditer(explicit_pattern_2, sent_str):
                val_raw = match.group(1).strip()
                unit_raw = "%" if match.group(2) else None
                metric_raw = match.group(3).strip()

                canonical = CANONICAL_METRIC_MAP.get(metric_raw.lower(), metric_raw)

                # Check negation and table scrap guards
                if is_metric_negated(sent_str, metric_raw, match.start(), match.end()):
                    continue
                if is_table_scrap_number(sent_str, match.start(), val_raw, unit_raw, canonical):
                    continue

                snippet = extract_snippet(sent_str, match.start(), match.end(), window=80)
                attribution = determine_metric_attribution(sent_str, val_raw, canonical, section)

                key = (canonical, val_raw, unit_raw, page, attribution)
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append({
                        "raw_value": val_raw,
                        "raw_text": snippet,
                        "metric_name": canonical,
                        "unit": unit_raw,
                        "page": page,
                        "section": section,
                        "status": "confirmed",
                        "attribution": attribution,
                        "explanation": f"Explicit metric '{canonical}' detected with value {val_raw}{unit_raw or ''}."
                    })

            # ----------------------------------------------------
            # CASE C: Ambiguous Performance Numbers
            # e.g., "detection performance with 91.3 Index Terms"
            # ----------------------------------------------------
            perf_patterns = [
                r"(?i)(?:performance|detection\s+performance|experimental\s+results?\s+(?:demonstrate|show|achieve)\s+(?:strong\s+)?performance)\s+(?:with|of|is|at|reaching)?\s*([0-9]+(?:\.[0-9]+)?)\s*(%|percent\b)?",
                r"(?i)(?:achieved?|attained?|scores?|scoring)\s+([0-9]+(?:\.[0-9]+)?)\s*(%|percent\b)?"
            ]

            for pat in perf_patterns:
                for match in re.finditer(pat, sent_str):
                    val_raw = match.group(1).strip()
                    unit_raw = "%" if match.group(2) else None

                    # Avoid capturing section numbers or years
                    try:
                        num_float = float(val_raw)
                        if 1950 <= num_float <= 2035 and num_float.is_integer() and not unit_raw:
                            continue
                    except ValueError:
                        pass

                    metric_found = resolve_metric_name(sent_str)
                    key_confirmed = (metric_found, val_raw, unit_raw, page, "proposed_work")
                    if metric_found and any(k[:4] == (metric_found, val_raw, unit_raw, page) for k in seen_keys):
                        continue

                    key_ambiguous = (None, val_raw, unit_raw, page, "uncertain")
                    if not any(k[:4] == (None, val_raw, unit_raw, page) for k in seen_keys):
                        seen_keys.add(key_ambiguous)
                        snippet = extract_snippet(sent_str, match.start(), match.end(), window=80)

                        subsequent_slice = sent_str[match.end():match.end() + 30].strip()
                        following_words = subsequent_slice.split()[:2]
                        following_label = " ".join(following_words) if following_words else ""

                        if re.search(r"^\s*index\s+terms\b", subsequent_slice, re.I):
                            expl = (
                                f"Ambiguous numerical performance value '{val_raw}' found immediately before "
                                f"'Index Terms' without explicit metric name or percentage symbol."
                            )
                        elif unit_raw:
                            expl = f"Percentage value '{val_raw}%' found without explicit evaluation metric name."
                        elif following_label:
                            expl = (
                                f"Numerical value '{val_raw}' appears in a performance context before '{following_label}' "
                                f"without an explicit evaluation metric name or unit."
                            )
                        else:
                            expl = f"Numerical value '{val_raw}' appears in a performance context without an explicit evaluation metric name or unit."

                        results.append({
                            "raw_value": val_raw,
                            "raw_text": snippet,
                            "metric_name": None,
                            "unit": unit_raw,
                            "page": page,
                            "section": section,
                            "status": "ambiguous",
                            "attribution": "uncertain",
                            "explanation": expl
                        })

    return results


# ============================================================
# 6. CONTRIBUTIONS & STATED LIMITATIONS
# ============================================================

def extract_contributions(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract author-stated contributions supported by source text.
    """
    contributions = []
    seen = set()

    contrib_patterns = [
        r"(?i)\b(?:our\s+(?:main\s+)?contributions?|this\s+paper\s+contributes?|the\s+contributions?\s+of\s+this\s+(?:paper|work)|we\s+make\s+the\s+following\s+contributions?)\b",
        r"(?i)\b(?:in\s+order\s+to\s+[^,\.]+,\s*this\s+article\s+suggests\s+a\s+[^,\.]+)\b",
        r"(?i)\b(?:this\s+(?:paper|article|work)\s+(?:proposes|presents|develops|introduces)\s+a\s+[^,\.]+)\b"
    ]

    for seg in segments:
        if seg.get("section", "").lower() in BIBLIOGRAPHY_SECTIONS:
            continue
        text = seg["text"]
        page = seg["page"]
        section = seg["section"]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            sent_str = sent.strip()
            if len(sent_str) < 30:
                continue

            for pat in contrib_patterns:
                if re.search(pat, sent_str):
                    norm = sent_str.lower()
                    if norm not in seen:
                        seen.add(norm)
                        contributions.append({
                            "statement": sent_str,
                            "evidence": sent_str,
                            "page": page,
                            "section": section
                        })
                    break

    return contributions


def extract_limitations(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract author-stated limitations and constraints supported by source text.
    """
    limitations = []
    seen = set()

    limitation_cues = [
        r"(?i)\b(?:limitations?|drawbacks?|shortcomings?)\b",
        r"(?i)\b(?:are\s+inadequate|is\s+inadequate|inadequate\s+for)\b",
        r"(?i)\b(?:computational\s+(?:overhead|cost|burden)|high\s+computational\s+cost)\b",
        r"(?i)\b(?:fails?\s+to|struggles?\s+with|limited\s+by|constraint|constrained\s+by)\b"
    ]

    for seg in segments:
        if seg.get("section", "").lower() in BIBLIOGRAPHY_SECTIONS:
            continue
        text = seg["text"]
        page = seg["page"]
        section = seg["section"]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            sent_str = sent.strip()
            if len(sent_str) < 30:
                continue

            is_limitation_section = section == "limitations"
            cue_matched = any(re.search(pat, sent_str) for pat in limitation_cues)

            if is_limitation_section or cue_matched:
                norm = sent_str.lower()
                if norm not in seen:
                    seen.add(norm)
                    limitations.append({
                        "statement": sent_str,
                        "evidence": sent_str,
                        "page": page,
                        "section": section
                    })

    return limitations


# ============================================================
# 7. AMBIGUITIES & EXTRACTION WARNINGS
# ============================================================

def extract_ambiguities_and_warnings(
    segments: List[Dict[str, Any]],
    metrics_and_results: List[Dict[str, Any]],
    algorithms: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compile warnings, ambiguous metrics, and extraction uncertainties.
    """
    warnings = []

    # 1. Ambiguous performance values
    for item in metrics_and_results:
        if item.get("status") == "ambiguous":
            warnings.append({
                "type": "ambiguous_performance_value",
                "description": item.get("explanation", "Ambiguous numerical value detected without confirmed metric."),
                "evidence": item.get("raw_text", ""),
                "page": item.get("page"),
                "section": item.get("section")
            })

    # 2. Check for missing critical sections
    sections_present = {seg["section"] for seg in segments if seg["section"] != "unknown"}
    if "methodology" not in sections_present:
        warnings.append({
            "type": "missing_section",
            "description": "Explicit methodology/proposed approach section heading was not detected.",
            "evidence": "",
            "page": None,
            "section": "methodology"
        })

    if "results" not in sections_present and "experiments" not in sections_present:
        warnings.append({
            "type": "missing_section",
            "description": "Explicit experimental results section heading was not detected.",
            "evidence": "",
            "page": None,
            "section": "results"
        })

    # 3. Warning if no algorithms or datasets were identifiable
    if not algorithms:
        warnings.append({
            "type": "no_algorithms_detected",
            "description": "No recognized algorithm or architectural variant was detected.",
            "evidence": "",
            "page": None,
            "section": None
        })

    if not datasets:
        warnings.append({
            "type": "no_datasets_detected",
            "description": "No explicit dataset names or dataset descriptions were detected.",
            "evidence": "",
            "page": None,
            "section": None
        })

    return warnings


# ============================================================
# DEDUPLICATION FUNCTIONS
# ============================================================

def deduplicate_algorithms(algorithms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = {}
    for item in algorithms:
        family = item.get("family", "")
        variant = item.get("variant")
        key = (family.lower(), variant.lower() if variant else None)
        if key not in grouped:
            c = dict(item)
            c["pages"] = [item["page"]] if item.get("page") is not None else []
            c["sections"] = [item["section"]] if item.get("section") else []
            c["all_evidence"] = [item["evidence"]] if item.get("evidence") else []
            grouped[key] = c
        else:
            e = grouped[key]
            if item.get("page") is not None and item["page"] not in e["pages"]:
                e["pages"].append(item["page"])
                e["pages"].sort()
            if item.get("section") and item["section"] not in e["sections"]:
                e["sections"].append(item["section"])
            if item.get("evidence") and item["evidence"] not in e["all_evidence"]:
                e["all_evidence"].append(item["evidence"])
                if len(item["evidence"]) > len(e.get("evidence", "")):
                    e["evidence"] = item["evidence"]
    return list(grouped.values())


def deduplicate_techniques(techniques: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = {}
    for item in techniques:
        key = item.get("name", "").lower()
        if key not in grouped:
            c = dict(item)
            c["pages"] = [item["page"]] if item.get("page") is not None else []
            c["sections"] = [item["section"]] if item.get("section") else []
            c["all_evidence"] = [item["evidence"]] if item.get("evidence") else []
            grouped[key] = c
        else:
            e = grouped[key]
            if item.get("page") is not None and item["page"] not in e["pages"]:
                e["pages"].append(item["page"])
                e["pages"].sort()
            if item.get("section") and item["section"] not in e["sections"]:
                e["sections"].append(item["section"])
            if item.get("evidence") and item["evidence"] not in e["all_evidence"]:
                e["all_evidence"].append(item["evidence"])
                if len(item["evidence"]) > len(e.get("evidence", "")):
                    e["evidence"] = item["evidence"]
    return list(grouped.values())


def deduplicate_datasets(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = {}
    for item in datasets:
        key = item.get("name", "").lower()
        if key not in grouped:
            c = dict(item)
            c["pages"] = [item["page"]] if item.get("page") is not None else []
            c["sections"] = [item["section"]] if item.get("section") else []
            c["all_evidence"] = [item["evidence"]] if item.get("evidence") else []
            grouped[key] = c
        else:
            e = grouped[key]
            if item.get("page") is not None and item["page"] not in e["pages"]:
                e["pages"].append(item["page"])
                e["pages"].sort()
            if item.get("section") and item["section"] not in e["sections"]:
                e["sections"].append(item["section"])
            if item.get("evidence") and item["evidence"] not in e["all_evidence"]:
                e["all_evidence"].append(item["evidence"])
                if len(item["evidence"]) > len(e.get("evidence", "")):
                    e["evidence"] = item["evidence"]
    return list(grouped.values())


def deduplicate_experimental_setup(setup: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    hw_grouped = {}
    for item in setup.get("hardware", []):
        key = item.get("name", "").lower()
        if key not in hw_grouped:
            c = dict(item)
            c["pages"] = [item["page"]] if item.get("page") is not None else []
            c["details"] = [item["detail"]] if item.get("detail") else []
            hw_grouped[key] = c
        else:
            e = hw_grouped[key]
            if item.get("page") is not None and item["page"] not in e["pages"]:
                e["pages"].append(item["page"])
                e["pages"].sort()
            if item.get("detail") and item["detail"] not in e["details"]:
                e["details"].append(item["detail"])

    sw_grouped = {}
    for item in setup.get("software_and_frameworks", []):
        key = item.get("name", "").lower()
        if key not in sw_grouped:
            c = dict(item)
            c["pages"] = [item["page"]] if item.get("page") is not None else []
            c["details"] = [item["detail"]] if item.get("detail") else []
            sw_grouped[key] = c
        else:
            e = sw_grouped[key]
            if item.get("page") is not None and item["page"] not in e["pages"]:
                e["pages"].append(item["page"])
                e["pages"].sort()
            if item.get("detail") and item["detail"] not in e["details"]:
                e["details"].append(item["detail"])

    hp_grouped = {}
    for item in setup.get("hyperparameters", []):
        key = (item.get("name", "").lower(), str(item.get("value", "")).lower())
        if key not in hp_grouped:
            c = dict(item)
            c["pages"] = [item["page"]] if item.get("page") is not None else []
            hp_grouped[key] = c
        else:
            e = hp_grouped[key]
            if item.get("page") is not None and item["page"] not in e["pages"]:
                e["pages"].append(item["page"])
                e["pages"].sort()

    return {
        "hardware": list(hw_grouped.values()),
        "software_and_frameworks": list(sw_grouped.values()),
        "hyperparameters": list(hp_grouped.values())
    }


def deduplicate_metrics_and_results(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = {}
    for item in metrics:
        key = (
            item.get("metric_name"),
            item.get("raw_value"),
            item.get("unit"),
            item.get("attribution", "uncertain")
        )
        if key not in grouped:
            c = dict(item)
            c["pages"] = [item["page"]] if item.get("page") is not None else []
            c["sections"] = [item["section"]] if item.get("section") else []
            grouped[key] = c
        else:
            e = grouped[key]
            if item.get("page") is not None and item["page"] not in e["pages"]:
                e["pages"].append(item["page"])
                e["pages"].sort()
            if item.get("section") and item["section"] not in e["sections"]:
                e["sections"].append(item["section"])
            if item.get("raw_text") and len(item["raw_text"]) > len(e.get("raw_text", "")):
                e["raw_text"] = item["raw_text"]
    return list(grouped.values())


# ============================================================
# MASTER FUNCTION: EXTRACT TECHNICAL FACTS
# ============================================================

def extract_technical_facts(
    pages: Optional[List[Dict[str, Any]]] = None,
    full_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for deterministic, evidence-backed technical fact extraction.

    Parameters:
    - pages: List of page dictionaries [{'page': 1, 'text': '...'}] (preferred for page attribution)
    - full_text: Raw paper text string (used when pages are not provided)

    Returns structured dictionary complying with new_paper_facts schema:
    - algorithms: Algorithms & variants with family hierarchy
    - techniques_and_architectures: Methodological approaches & paradigms
    - datasets: Standard benchmark datasets and custom descriptions
    - experimental_setup: Hardware, software, hyperparameters
    - metrics_and_results: Metric-value pairs & ambiguous performance values
    - contributions: Stated contributions supported by text
    - limitations: Stated limitations and constraints
    - ambiguities_and_warnings: Uncertainties, extraction warnings, ambiguous values
    - methods_mentioned: Legacy compatibility list of method/model names
    - metrics_mentioned: Legacy compatibility list of metric names
    - accuracy_values_found_percent: Legacy compatibility list of confirmed accuracy percentages
    """
    # Handle empty input gracefully
    if not pages and not (full_text and full_text.strip()):
        return {
            "algorithms": [],
            "techniques_and_architectures": [],
            "datasets": [],
            "experimental_setup": {
                "hardware": [],
                "software_and_frameworks": [],
                "hyperparameters": []
            },
            "metrics_and_results": [],
            "contributions": [],
            "limitations": [],
            "ambiguities_and_warnings": [
                {
                    "type": "empty_input",
                    "description": "No text or pages provided for technical fact extraction.",
                    "evidence": "",
                    "page": None,
                    "section": None
                }
            ],
            "methods_mentioned": [],
            "metrics_mentioned": [],
            "accuracy_values_found_percent": []
        }

    segments = parse_paper_sections(pages=pages, full_text=full_text)

    # 1. Extraction passes with deduplication
    algorithms = deduplicate_algorithms(extract_algorithms(segments))
    techniques = deduplicate_techniques(extract_techniques(segments))
    datasets = deduplicate_datasets(extract_datasets(segments))
    experimental_setup = deduplicate_experimental_setup(extract_experimental_setup(segments))
    metrics_and_results = deduplicate_metrics_and_results(extract_metrics_and_results(segments))
    contributions = extract_contributions(segments)
    limitations = extract_limitations(segments)

    # 2. Ambiguities and warnings
    warnings = extract_ambiguities_and_warnings(
        segments=segments,
        metrics_and_results=metrics_and_results,
        algorithms=algorithms,
        datasets=datasets
    )

    # 3. Legacy compatibility fields
    # methods_mentioned: unique lowercase names of algorithms & techniques
    methods_set = set()
    for alg in algorithms:
        methods_set.add(alg["name"].lower())
        if alg.get("family"):
            methods_set.add(alg["family"].lower())
    for tech in techniques:
        methods_set.add(tech["name"].lower())
    methods_mentioned = sorted(list(methods_set))

    # metrics_mentioned: unique canonical names of confirmed metrics
    metrics_set = set()
    accuracy_percents = []
    for item in metrics_and_results:
        m_name = item.get("metric_name")
        if m_name and item.get("status") == "confirmed":
            metrics_set.add(m_name.lower())
            if m_name.lower() == "accuracy" and item.get("unit") == "%":
                try:
                    accuracy_percents.append(float(item["raw_value"]))
                except (ValueError, TypeError):
                    pass
    metrics_mentioned = sorted(list(metrics_set))

    return {
        "algorithms": algorithms,
        "techniques_and_architectures": techniques,
        "datasets": datasets,
        "experimental_setup": experimental_setup,
        "metrics_and_results": metrics_and_results,
        "contributions": contributions,
        "limitations": limitations,
        "ambiguities_and_warnings": warnings,
        "methods_mentioned": methods_mentioned,
        "metrics_mentioned": metrics_mentioned,
        "accuracy_values_found_percent": accuracy_percents
    }
