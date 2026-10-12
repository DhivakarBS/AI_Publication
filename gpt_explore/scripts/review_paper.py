
import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pymupdf

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from technical_extractor import extract_technical_facts
except ImportError:
    from scripts.technical_extractor import extract_technical_facts


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PAPERS_DIR = ROOT / "dataset" / "papers"
VECTOR_DB_DIR = ROOT / "storage" / "vector_db"
METADATA_CSV = ROOT / "dataset" / "metadata" / "papers.csv"
REPORT_DIR = ROOT / "dataset" / "metadata" / "review_reports"
SECTIONS_DIR = ROOT / "dataset" / "sections"
EXTRACTED_TEXT_DIR = ROOT / "dataset" / "extracted_text"

COLLECTION_NAME = "ai_ml_papers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOP_K_CHUNKS = 40
TOP_PAPERS = 5
MAX_CHUNKS_PER_PAPER = 3

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_path):
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def find_exact_duplicate(pdf_path, file_hash):
    if not PAPERS_DIR.exists():
        return None

    for existing in PAPERS_DIR.glob("*.pdf"):
        try:
            if existing.resolve() == pdf_path.resolve():
                continue

            if sha256_file(existing) == file_hash:
                return existing.name

        except OSError:
            continue

    return None


def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def extract_title_abstract_intro(pages):
    """
    Heuristic extraction used to prepare input for the LLM.
    The LLM will help identify the actual title and research problem.
    """

    full_text = "\n".join(page["text"] for page in pages)

    lines = [
        line.strip()
        for line in full_text.splitlines()
        if line.strip()
    ]

    title_candidate = ""

    # Search early lines for a possible title.
    # Conference headers are not guaranteed to be excluded here;
    # the LLM receives the surrounding context to resolve ambiguity.
    for line in lines[:35]:
        line = normalize_text(line)

        if (
            15 <= len(line) <= 300
            and not re.match(
                r"^(abstract|keywords?|doi|arxiv)\b",
                line,
                re.IGNORECASE
            )
            and not re.match(r"^\d+\s+(volume|issue)\b", line, re.I)
        ):
            title_candidate = line
            break

    abstract = ""

    abstract_match = re.search(
        r"\babstract\b\s*[:.]?\s*(.*?)(?="
        r"\bkeywords?\b|\bintroduction\b)",
        full_text,
        re.IGNORECASE | re.DOTALL
    )

    if abstract_match:
        abstract = normalize_text(abstract_match.group(1))

    opening_text = "\n".join(
        page["text"] for page in pages[:4]
    )

    introduction_match = re.search(
        r"\bintroduction\b\s*(.*)",
        opening_text,
        re.IGNORECASE | re.DOTALL
    )

    if introduction_match:
        introduction = normalize_text(
            introduction_match.group(1)
        )[:8000]
    else:
        introduction = normalize_text(opening_text)[:8000]

    # Fallback if the abstract heading was not detected.
    if len(abstract) < 50:
        abstract = normalize_text(opening_text)[:3500]

    return (
        title_candidate or "Title not identified",
        abstract[:5000],
        introduction,
        full_text
    )


# ============================================================
# LLM CLIENT
# ============================================================

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it in PowerShell "
            "before running the LLM workflow."
        )

    return OpenAI(api_key=api_key)


def call_llm_json(client, instructions, input_data):
    response = client.responses.create(
        model=LLM_MODEL,
        instructions=instructions,
        input=json.dumps(input_data, ensure_ascii=False)
    )

    output = response.output_text.strip()

    # Handle JSON returned inside a Markdown code fence.
    if output.startswith("```"):
        output = output.split("\n", 1)[1]
        output = output.rsplit("```", 1)[0].strip()

    return json.loads(output)


def summarize_facts_for_llm(technical_facts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a clean, grounded summary of deterministic facts for LLM prompts."""
    if not technical_facts:
        return {}
    return {
        "algorithms": [
            {
                "name": alg.get("name"),
                "family": alg.get("family"),
                "variant": alg.get("variant"),
                "category": alg.get("category"),
                "pages": alg.get("pages", [alg.get("page")]),
            }
            for alg in technical_facts.get("algorithms", [])
        ],
        "datasets": [
            {
                "name": d.get("name"),
                "description": d.get("description"),
                "pages": d.get("pages", [d.get("page")]),
            }
            for d in technical_facts.get("datasets", [])
        ],
        "metrics_and_results": [
            {
                "metric_name": m.get("metric_name"),
                "raw_value": m.get("raw_value"),
                "unit": m.get("unit"),
                "attribution": m.get("attribution", "uncertain"),
                "status": m.get("status"),
                "evidence": m.get("raw_text"),
                "pages": m.get("pages", [m.get("page")]),
            }
            for m in technical_facts.get("metrics_and_results", [])
            if m.get("status") == "confirmed"
        ],
        "experimental_setup": {
            "hardware": [h.get("name") for h in technical_facts.get("experimental_setup", {}).get("hardware", [])],
            "software": [s.get("name") for s in technical_facts.get("experimental_setup", {}).get("software_and_frameworks", [])],
            "hyperparameters": [
                f"{hp.get('name')}={hp.get('value')}"
                for hp in technical_facts.get("experimental_setup", {}).get("hyperparameters", [])
            ],
        },
        "contributions": [c.get("statement") for c in technical_facts.get("contributions", [])[:5]],
        "limitations": [l.get("statement") for l in technical_facts.get("limitations", [])[:5]],
    }


# ============================================================
# STEP 1: RESEARCH PROFILE EXTRACTION
# ============================================================

def extract_research_profile(
    client, title_candidate, abstract, introduction, technical_facts=None
):
    paper_content = {
        "title_candidate": title_candidate,
        "abstract": abstract,
        "introduction": introduction[:8000]
    }
    if technical_facts:
        paper_content["deterministic_technical_facts"] = summarize_facts_for_llm(technical_facts)

    instructions = """
You are an academic research-paper analysis assistant.

Identify the actual paper title, not the conference name,
journal name, event theme, author affiliation, or publication venue.

Infer the research area from the supplied content. Do not use a
predefined list of domains.

Ground your analysis in the supplied deterministic technical facts (extracted
algorithms, datasets, experimental setup, and metrics with attribution).
Do not contradict the verified facts or hallucinate alternative metrics.

Return valid JSON with exactly these fields:

{
  "title": "actual title or uncertain",
  "research_area": "specific research area",
  "research_problem": "problem being addressed",
  "objectives": ["objective 1"],
  "proposed_methodology": "method proposed by the authors",
  "data_and_dataset": "dataset or input data, if identified",
  "evaluation_metrics": ["metrics explicitly mentioned"],
  "main_claims": ["claims made by the authors"],
  "claimed_contributions": ["claimed contributions"],
  "limitations_stated": ["explicitly stated limitations"],
  "uncertainties": ["uncertain or missing information"]
}

Do not invent missing information.
Distinguish author claims from experimentally demonstrated results.
If the document title cannot be identified reliably, say so.
"""

    return call_llm_json(
        client,
        instructions,
        paper_content
    )


def generate_local_research_profile(
    title_candidate: str,
    abstract: str,
    introduction: str,
    technical_facts: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a deterministic, evidence-grounded research profile from technical facts
    and paper text for local-only execution without external LLM calls.
    Follows the exact report schema and preserves evidence without hallucinating claims.
    """
    facts = technical_facts or {}

    # 1. Title
    title = title_candidate.strip() if title_candidate and title_candidate.strip() != "Title not identified" else "Title not identified"

    # 2. Research area inference
    algs = facts.get("algorithms", [])
    techs = facts.get("techniques_and_architectures", [])
    categories = {a.get("category") for a in algs if a.get("category")}
    categories.update(t.get("category") for t in techs if t.get("category"))

    text_sample = f"{title} {abstract[:500]}".lower()
    if "intrusion detection" in text_sample or "cyber" in text_sample or "nids" in text_sample:
        area = "Network Intrusion Detection & Cybersecurity"
    elif "autonomous driving" in text_sample or "vehicle" in text_sample:
        area = "Autonomous Systems & Intelligent Transportation"
    elif "plant disease" in text_sample or "agriculture" in text_sample:
        area = "Agricultural Computer Vision & Precision Farming"
    elif "medical" in text_sample or "clinical" in text_sample or "healthcare" in text_sample:
        area = "Medical AI & Healthcare Informatics"
    elif "reinforcement_learning" in categories:
        area = "Deep Reinforcement Learning"
    elif "object_detection" in categories or "computer_vision" in categories:
        area = "Computer Vision & Deep Learning"
    elif "nlp" in categories:
        area = "Natural Language Processing"
    elif categories:
        area = ", ".join(sorted(categories)).replace("_", " ").title()
    else:
        area = "Machine Learning & Artificial Intelligence"

    # 3. Research problem
    # Extract problem statement from abstract opening or introduction
    research_problem = "Uncertain / not explicitly extracted"
    if abstract:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if s.strip()]
        if sentences:
            research_problem = sentences[0]
            if len(sentences) > 1 and len(research_problem) < 100:
                research_problem += " " + sentences[1]

    # 4. Proposed methodology
    prop_algs = [a.get("variant") or a.get("name") for a in algs]
    prop_techs = [t.get("name") for t in techs[:3]]
    if prop_algs:
        proposed_methodology = ", ".join(prop_algs[:4])
        if prop_techs:
            proposed_methodology += f" utilizing {', '.join(prop_techs)}"
    else:
        proposed_methodology = "Uncertain / not explicitly identified"

    # 5. Datasets
    datasets = facts.get("datasets", [])
    if datasets:
        dataset_str = ", ".join(d.get("name") for d in datasets)
    else:
        dataset_str = "None identified"

    # 6. Evaluation metrics
    metrics_mentioned = facts.get("metrics_mentioned", [])
    if not metrics_mentioned:
        confirmed_m = {m.get("metric_name") for m in facts.get("metrics_and_results", []) if m.get("status") == "confirmed" and m.get("metric_name")}
        metrics_mentioned = sorted(confirmed_m)

    # 7. Objectives
    objectives = []
    contributions = facts.get("contributions", [])
    for c in contributions:
        stmt = c.get("statement", "").strip()
        if stmt and stmt not in objectives:
            objectives.append(stmt)
    if not objectives and abstract:
        for s in re.split(r"(?<=[.!?])\s+", abstract):
            if re.search(r"\b(?:we\s+(?:propose|introduce|present|aim)|this\s+paper\s+(?:proposes|introduces|presents))\b", s, re.IGNORECASE):
                objectives.append(s.strip())
                break

    # 8. Main claims & Key results
    main_claims = []
    key_results = []
    metrics_results = facts.get("metrics_and_results", [])
    for mr in metrics_results:
        if mr.get("status") == "confirmed" and mr.get("attribution") == "proposed_work":
            val_str = f"{mr.get('metric_name')}: {mr.get('raw_value')}{mr.get('unit') or ''}"
            claim_str = f"Achieved {val_str} (Page {mr.get('page')})" if mr.get("page") else f"Achieved {val_str}"
            if claim_str not in main_claims:
                main_claims.append(claim_str)
            if val_str not in key_results:
                key_results.append(val_str)

    # 9. Stated limitations
    limitations = [l.get("statement") for l in facts.get("limitations", []) if l.get("statement")]

    # 10. Uncertainties
    warnings = [w.get("description") for w in facts.get("ambiguities_and_warnings", []) if w.get("description")]

    return {
        "title": title,
        "research_area": area,
        "research_problem": research_problem,
        "objectives": objectives,
        "proposed_methodology": proposed_methodology,
        "data_and_dataset": dataset_str,
        "evaluation_metrics": metrics_mentioned,
        "main_claims": main_claims,
        "key_results": key_results,
        "claimed_contributions": [c.get("statement") for c in contributions if c.get("statement")],
        "limitations_stated": limitations,
        "uncertainties": warnings
    }


# ============================================================
# METADATA
# ============================================================

def load_paper_titles():
    titles = {}

    if not METADATA_CSV.exists():
        return titles

    with open(
        METADATA_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            paper_id = row.get("paper_id")
            title = row.get("title")

            if paper_id and title:
                titles[str(paper_id)] = title

    return titles


# ============================================================
# STEP 2: LOCAL SEMANTIC RETRIEVAL
# ============================================================

def retrieve_chunks(collection, model, query_text):
    count = collection.count()

    if count == 0:
        return []

    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_CHUNKS, count),
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        metadata = metadata or {}
        paper_id = str(metadata.get("paper_id", "")).strip()

        if not paper_id or not document:
            continue

        chunks.append({
            "paper_id": paper_id,
            "title": metadata.get("title", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "section_type": metadata.get("section_type", "unknown"),
            "original_heading": metadata.get("original_heading", ""),
            "text": document,
            "distance": float(distance)
        })

    return chunks


def identify_source_paper_id(pdf_path: Path, file_hash: str) -> Optional[str]:
    """
    Identify the source paper's stable paper_id (e.g. 'PAPER_033')
    using SHA-256 match in papers.csv, filename prefix, or exact duplicate detection.
    """
    # 1. Match against metadata CSV sha256 or filename
    if METADATA_CSV.exists():
        with open(METADATA_CSV, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("sha256") == file_hash:
                    return str(row.get("paper_id")).strip()
                if row.get("pdf_filename") and Path(row["pdf_filename"]).name.lower() == pdf_path.name.lower():
                    return str(row.get("paper_id")).strip()

    # 2. Match standard PAPER_XXX pattern in filename
    m = re.search(r"\b(PAPER_\d+)\b", pdf_path.name, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def is_bibliography_chunk(chunk: Dict[str, Any]) -> bool:
    """
    Determine if a chunk belongs to a references/bibliography section.
    Preserves legitimate related-work text that discusses and cites papers.
    """
    section_type = str(chunk.get("section_type") or "").strip().lower()
    if section_type in ("references", "bibliography", "works cited", "literature cited"):
        return True

    original_heading = str(chunk.get("original_heading") or "").strip().lower()
    if re.search(r"^(?:references|bibliography|works cited|literature cited)\b", original_heading, re.IGNORECASE):
        return True

    text = chunk.get("text") or ""
    if not text.strip():
        return False

    first_line = text.strip().split("\n", 1)[0].strip()
    if re.match(r"^(?:references|bibliography|works cited|literature cited)\b", first_line, re.IGNORECASE):
        return True

    # Conservative text fallback:
    # Multiple citation entries e.g. [1] Author..., [2] Author... or 1. Author...
    bracket_cites = re.findall(r"\[\d+\]\s+[A-Z]", text)
    numbered_cites = re.findall(r"(?:^|\n)\s*\d+\.\s+[A-Z]", text)
    total_cites = len(bracket_cites) + len(numbered_cites)

    if total_cites >= 2:
        pub_markers = len(re.findall(r"\b(?:in proc\.|proceedings|conference|journal|symposium|workshop|ieee|acm|springer|elsevier|vol\.|pp\.|pages|trans\.|arxiv|iccv|cvpr|icml|neurips|nips|iclr|aaai|ijcai|kdd|sigir|acl|emnlp|icassp|eccv)\b", text, re.IGNORECASE))
        # Legitimate related work prose has discussion verbs and discourse markers
        prose_discussion = len(re.findall(r"\b(?:we propose|we use|we demonstrate|our method|our approach|in this paper|outperforms|evaluated on|this study|the authors propose|proposed an?|introduced an?)\b", text, re.IGNORECASE))
        if pub_markers >= 1 and prose_discussion == 0:
            return True

    return False


def group_and_rank_papers(chunks, title_lookup, exclude_paper_id: Optional[str] = None):
    grouped = {}

    for chunk in chunks:
        if is_bibliography_chunk(chunk):
            continue
        pid = chunk["paper_id"]
        if exclude_paper_id and pid.upper() == exclude_paper_id.upper():
            continue
        grouped.setdefault(pid, []).append(chunk)

    papers = []

    for paper_id, paper_chunks in grouped.items():
        paper_chunks.sort(key=lambda item: item["distance"])

        selected_chunks = paper_chunks[:MAX_CHUNKS_PER_PAPER]

        title = (
            next(
                (
                    chunk.get("title")
                    for chunk in selected_chunks
                    if chunk.get("title")
                ),
                None
            )
            or title_lookup.get(paper_id)
            or "Title unavailable"
        )

        papers.append({
            "paper_id": paper_id,
            "title": title,
            "best_vector_distance": round(
                selected_chunks[0]["distance"], 5
            ),
            "evidence": [
                {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "section_type": chunk.get("section_type", ""),
                    "original_heading": chunk.get("original_heading", ""),
                    "text": chunk.get("text", "")[:3000]
                }
                for chunk in selected_chunks
            ]
        })

    papers.sort(key=lambda item: item["best_vector_distance"])

    return papers


# Global candidate facts cache to prevent redundant extractions within a run
CANDIDATE_FACTS_CACHE: Dict[str, Any] = {}


def extract_candidate_facts(
    candidate: Dict[str, Any],
    cache: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract structured technical facts for a candidate paper using the
    existing deterministic technical extractor.
    Uses full extracted candidate text when available; otherwise extracts from
    retrieved chunks and marks evidence provenance as partial.
    Caches facts by paper_id to avoid repeated extraction.
    """
    paper_id = candidate.get("paper_id", "")
    target_cache = cache if cache is not None else CANDIDATE_FACTS_CACHE
    if paper_id and paper_id in target_cache:
        return json.loads(json.dumps(target_cache[paper_id]))

    full_text = None
    provenance = "retrieved_chunks_partial"
    evidence_scope = "partial_evidence"

    # 1. Try dataset/extracted_text/
    if EXTRACTED_TEXT_DIR.exists():
        matches = list(EXTRACTED_TEXT_DIR.glob(f"{paper_id}_*.txt")) or list(EXTRACTED_TEXT_DIR.glob(f"{paper_id}.txt"))
        if matches and matches[0].is_file():
            try:
                full_text = matches[0].read_text(encoding="utf-8", errors="replace")
                provenance = "full_paper_text"
                evidence_scope = "full_evidence"
            except Exception:
                full_text = None

    # 2. Try dataset/sections/{paper_id}.json if extracted_text was not found
    if full_text is None and SECTIONS_DIR.exists():
        sec_path = SECTIONS_DIR / f"{paper_id}.json"
        if sec_path.is_file():
            try:
                sec_data = json.loads(sec_path.read_text(encoding="utf-8", errors="replace"))
                # Exclude references section to prevent bibliography contamination
                sec_parts = [
                    sec_data.get(k, "") for k in [
                        "title", "abstract", "introduction", "related_work",
                        "methodology", "dataset", "experiments", "results",
                        "discussion", "limitations", "conclusion"
                    ] if sec_data.get(k)
                ]
                if sec_parts:
                    full_text = "\n\n".join(sec_parts)
                    provenance = "full_paper_sections"
                    evidence_scope = "full_evidence"
            except Exception:
                full_text = None

    # 3. Fallback to retrieved evidence chunks
    if not full_text:
        evidence_chunks = candidate.get("evidence", [])
        chunk_texts = [e.get("text", "") for e in evidence_chunks if e.get("text")]
        full_text = "\n\n".join(chunk_texts)
        provenance = "retrieved_chunks_partial"
        evidence_scope = "partial_evidence"

    if full_text and full_text.strip():
        extracted = extract_technical_facts(
            full_text=full_text,
            paper_title=candidate.get("title")
        )
    else:
        extracted = {
            "algorithms": [],
            "techniques_and_architectures": [],
            "datasets": [],
            "experimental_setup": {"hardware": [], "software_and_frameworks": [], "hyperparameters": []},
            "metrics_and_results": [],
            "metrics_mentioned": [],
            "contributions": [],
            "limitations": [],
            "ambiguities_and_warnings": []
        }

    # Tag provenance and candidate attribution clearly
    extracted["paper_id"] = paper_id
    extracted["evidence_provenance"] = provenance
    extracted["evidence_scope"] = evidence_scope

    # Ensure candidate metrics are isolated and clearly labeled as candidate findings
    for m in extracted.get("metrics_and_results", []):
        m["provenance_paper_id"] = paper_id
        if m.get("attribution") == "proposed_work":
            m["attribution"] = "candidate_proposed_work"

    if paper_id:
        target_cache[paper_id] = json.loads(json.dumps(extracted))

    return extracted


def enrich_candidate_papers_with_facts(
    candidates: List[Dict[str, Any]],
    cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Enrich all candidate papers with structured technical facts.
    """
    for candidate in candidates:
        candidate["technical_facts"] = extract_candidate_facts(candidate, cache=cache)
    return candidates


# ============================================================
# STEP 3: LLM RELEVANCE FILTERING
# ============================================================

def select_relevant_papers(client, profile, candidate_papers):
    """
    Let the LLM assess whether retrieved papers address the same
    problem. It cannot inspect papers that local retrieval missed.
    """

    candidates = []

    for paper in candidate_papers:
        candidates.append({
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "evidence": paper["evidence"][:2]
        })

    instructions = """
You are screening candidate research papers for relevance.

Compare each candidate with the new paper's research problem,
target task, application, inputs, and intended outputs.

Do not rank a paper highly merely because it uses the same
algorithm, neural network, or evaluation metric.

Return valid JSON:

{
  "matches": [
    {
      "paper_id": "existing paper ID",
      "relevance": "high, medium, low, or uncertain",
      "same_research_problem": true,
      "reason": "evidence-based explanation",
      "missing_information": ["what cannot be verified"]
    }
  ],
  "overall_assessment": "summary of candidate relevance"
}

Only use the supplied evidence.
Do not invent titles or findings.
A lack of evidence is not proof that a paper is unrelated.
"""

    return call_llm_json(
        client,
        instructions,
        {
            "new_paper_profile": profile,
            "candidate_papers": candidates
        }
    )


# ============================================================
# STEP 4: EVIDENCE-BASED COMPARATIVE REVIEW
# ============================================================

def analyze_comparison(client, profile, selected_papers, technical_facts=None):
    evidence = []

    for paper in selected_papers:
        evidence.append({
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "relevance_assessment": paper.get(
                "relevance_assessment", {}
            ),
            "evidence": paper["evidence"]
        })

    instructions = """
You are preparing a preliminary academic literature review.

Compare the new paper with the supplied existing research papers.

Address:

1. Research problem and application overlap.
2. Existing methods versus the proposed method.
3. Dataset and experimental setup.
4. Reported accuracy and evaluation metrics.
5. Claimed contributions and whether the provided evidence supports them.
6. Potential methodological improvements.
7. Potential research gaps.
8. Experiments needed to validate the claims.
9. Limitations and missing evidence.

Rules:
- Ground comparative claims in the provided new_paper_verified_facts.
- Use the metric attributions (proposed_work vs prior_work) to ensure baseline
  or cited metrics are not falsely attributed to the new paper's proposed model.
- Do not invent results, citations, accuracy values, or datasets.
- Distinguish a claim made by the author from a result demonstrated
  by the supplied experimental evidence.
- Do not declare novelty proven or disprove novelty based on a
  limited candidate set.
- Explain when comparisons are unfair because datasets or metrics differ.
- If evidence is insufficient, state exactly what is missing.
- Cite the supplied paper IDs when discussing existing work.
- Return a readable Markdown report.
"""

    payload = {
        "new_paper_profile": profile,
        "related_papers": evidence
    }
    if technical_facts:
        payload["new_paper_verified_facts"] = summarize_facts_for_llm(technical_facts)

    response = client.responses.create(
        model=LLM_MODEL,
        instructions=instructions,
        input=json.dumps(
            payload,
            ensure_ascii=False
        )
    )

    return response.output_text


# ============================================================
# DETERMINISTIC LOCAL RELEVANCE AND COMPARATIVE REVIEW
# ============================================================

def generate_local_relevance_assessment(
    profile: Dict[str, Any],
    candidate_papers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Populate a deterministic local relevance assessment using task, method,
    dataset, and evidence overlap without presenting vector similarity as scientific probability.
    """
    matches = []
    input_area = (profile.get("research_area") or "").lower()
    input_prob = (profile.get("research_problem") or "").lower()
    input_method = (profile.get("proposed_methodology") or "").lower()
    input_data_str = (profile.get("data_and_dataset") or "").lower()

    # Collect meaningful domain tokens
    keywords = set()
    for text_source in [input_area, input_prob, input_method]:
        for token in re.findall(r"\b[a-z]{4,}\b", text_source):
            if token not in ("with", "using", "that", "this", "from", "were", "been", "have", "approach", "method", "proposed", "paper", "model"):
                keywords.add(token)

    high_count = 0
    med_count = 0

    for paper in candidate_papers:
        pid = paper.get("paper_id", "")
        title = paper.get("title", "")
        facts = paper.get("technical_facts", {})
        evidence_text = " ".join(e.get("text", "") for e in paper.get("evidence", [])).lower()
        cand_str = f"{title.lower()} {evidence_text}"

        cand_algs = {a.get("family", "").lower() for a in facts.get("algorithms", []) if a.get("family")}
        cand_algs.update(a.get("name", "").lower() for a in facts.get("algorithms", []) if a.get("name"))
        cand_datasets = {d.get("name", "").lower() for d in facts.get("datasets", []) if d.get("name")}

        shared_kw = [k for k in keywords if k in cand_str]

        # Check problem domain match and survey status
        is_survey = bool(re.search(r"\b(?:survey|review|overview)\b", title, re.IGNORECASE))
        domain_match = False
        relevance = "uncertain"
        same_problem = False
        reason = ""
        missing_info = []

        if ("intrusion detection" in input_area or "nids" in input_area) and ("intrusion detection" in cand_str or "nids" in cand_str or "kdd" in cand_str or "attack" in cand_str or "anomaly detection" in cand_str):
            domain_match = True
            same_problem = ("intrusion detection" in cand_str or "nids" in cand_str) and not is_survey
        elif ("reinforcement learning" in input_area or "game" in input_area or "chess" in input_prob or "board game" in input_prob):
            if "reinforcement learning" in cand_str or "chess" in cand_str or "shogi" in cand_str or "mcts" in cand_str or "self-play" in cand_str or "policy" in cand_str:
                domain_match = True
                # Distinguish board-game playing from autonomous driving / robotics:
                is_input_games = any(g in (input_area + " " + input_prob) for g in ["chess", "shogi", "go", "game", "board game"])
                is_cand_driving = any(d in cand_str for d in ["driving", "autonomous driving", "vehicle", "traffic"])
                if is_input_games and is_cand_driving:
                    same_problem = False
                elif is_input_games and any(g in cand_str for g in ["chess", "shogi", "go", "game", "board game"]):
                    same_problem = not is_survey
                else:
                    same_problem = False
        elif ("segmentation" in input_area or "biomedical" in input_area):
            if "segmentation" in cand_str or "u-net" in cand_str or "medical" in cand_str or "biomedical" in cand_str:
                domain_match = True
                same_problem = ("segmentation" in cand_str and ("medical" in cand_str or "biomedical" in cand_str)) and not is_survey
        elif len(shared_kw) >= 3:
            domain_match = True
            same_problem = False

        if is_survey:
            same_problem = False
            relevance = "medium" if domain_match else "low"
            if relevance == "medium":
                med_count += 1
            reason = f"Survey/background source: '{title}' synthesizes broader domain literature rather than directly competing on the specific research problem."
        elif domain_match:
            shared_datasets = [d for d in cand_datasets if d in input_data_str]
            if same_problem and (shared_datasets or len(shared_kw) >= 3):
                relevance = "high"
                high_count += 1
                reason = f"Shares primary research domain ({profile.get('research_area', 'domain')}) with direct task alignment and shared contextual keywords ({', '.join(shared_kw[:3])})."
            elif same_problem:
                relevance = "medium"
                med_count += 1
                reason = f"Shares core research problem within {profile.get('research_area', 'domain')}, with distinct experimental configuration."
            else:
                relevance = "medium" if len(shared_kw) >= 2 else "low"
                if relevance == "medium":
                    med_count += 1
                reason = f"Related broad domain ({profile.get('research_area', 'domain')}), but addresses a distinct specific research task."
        else:
            if len(shared_kw) >= 1:
                relevance = "low"
                reason = f"General domain or algorithmic overlap ({', '.join(shared_kw[:2])}), but distinct specific research problem."
            else:
                relevance = "uncertain"
                reason = "Insufficient evidence in retrieved candidate chunks to confirm research-problem alignment."

        if facts.get("evidence_scope") == "partial_evidence":
            missing_info.append("Candidate full experimental protocol (extracted from partial chunks)")
        if not facts.get("metrics_and_results"):
            missing_info.append("Confirmed quantitative baseline results in candidate paper")
        if not facts.get("datasets"):
            missing_info.append("Explicit candidate dataset metadata")

        matches.append({
            "paper_id": pid,
            "relevance": relevance,
            "same_research_problem": same_problem,
            "reason": reason,
            "missing_information": missing_info if missing_info else ["None identified"]
        })

    overall = (
        f"Deterministic local screening identified {high_count} high-relevance and {med_count} medium-relevance "
        f"candidate(s) among {len(candidate_papers)} retrieved papers based on research task, methodology, and evidence overlap."
    )

    return {
        "matches": matches,
        "overall_assessment": overall
    }


def generate_local_comparative_review(
    profile: Dict[str, Any],
    technical_facts: Dict[str, Any],
    candidate_papers: List[Dict[str, Any]],
    relevance_assessment: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a deterministic, evidence-grounded comparative review covering:
    1. Research-task overlap
    2. Shared and different algorithms or architectures
    3. Shared and different datasets
    4. Experimental-setup differences
    5. Metrics and results, keeping metric types, datasets, and attribution separate
    6. Evidence-supported similarities and differences
    7. Missing information and uncertainties
    """
    if not candidate_papers:
        return {
            "status": "insufficient_evidence",
            "summary": "No candidate papers available for comparative review.",
            "research_task_overlap": [],
            "methodological_comparison": [],
            "dataset_comparison": [],
            "experimental_setup_comparison": [],
            "metrics_and_results_comparison": [],
            "substantive_claims": [],
            "missing_information_and_uncertainties": ["Corpus retrieval yielded zero eligible candidate papers."]
        }

    input_title = profile.get("title", "Input Paper")
    input_area = profile.get("research_area", "Unspecified Area")
    input_algs = technical_facts.get("algorithms", [])
    priority_order = [
        "proposed_method",
        "experiment_or_baseline",
        "background_or_related_work",
        "references_only",
        "uncertain"
    ]
    input_alg_map = {}
    for a in input_algs:
        name = a.get("name") or a.get("family") or ""
        fam = a.get("family") or ""
        u = a.get("usage", "uncertain")
        ev = a.get("evidence", "")
        keys = set(filter(None, [name.lower(), fam.lower()]))
        for k in keys:
            if k not in input_alg_map or priority_order.index(u) < priority_order.index(input_alg_map[k]["usage"]):
                input_alg_map[k] = {
                    "display_name": a.get("variant") or a.get("name") or fam,
                    "family": fam,
                    "usage": u,
                    "evidence": ev,
                    "item": a
                }

    input_datasets = [d.get("name") for d in technical_facts.get("datasets", [])]
    input_setup = technical_facts.get("experimental_setup", {})
    input_metrics = [
        m for m in technical_facts.get("metrics_and_results", [])
        if m.get("status") == "confirmed"
    ]

    relevance_by_id = {}
    if relevance_assessment:
        for m in relevance_assessment.get("matches", []):
            relevance_by_id[m.get("paper_id")] = m

    task_overlap_list = []
    method_comp_list = []
    dataset_comp_list = []
    setup_comp_list = []
    metric_comp_list = []
    substantive_claims = []
    uncertainties = []

    for paper in candidate_papers:
        pid = paper.get("paper_id", "")
        title = paper.get("title", "")
        cand_facts = paper.get("technical_facts", {})
        cand_evidence = paper.get("evidence", [])
        primary_snippet = cand_evidence[0].get("text", "")[:300] if cand_evidence else "No snippet available."
        primary_chunk = cand_evidence[0].get("chunk_id", "") if cand_evidence else "unknown"
        rel_info = relevance_by_id.get(pid, {})
        is_survey = bool(re.search(r"\b(?:survey|review|overview)\b", title, re.IGNORECASE))

        # 1. Task overlap
        rel_label = rel_info.get("relevance", "uncertain")
        if is_survey:
            task_status = "partial_evidence"
            task_desc = f"Survey/background source: '{title}' reviews literature rather than proposing or directly competing on the specific research problem."
        elif rel_label == "high":
            task_status = "supported"
            task_desc = f"Candidate addresses overlapping problem in {input_area}: {rel_info.get('reason', 'Domain alignment identified.')}"
        elif rel_label == "medium":
            task_status = "partial_evidence"
            task_desc = f"Candidate addresses related problem space in {title}: {rel_info.get('reason', 'Partial task overlap.')}"
        elif rel_label == "low":
            task_status = "partial_evidence"
            task_desc = f"Candidate addresses distinct problem in {title}: {rel_info.get('reason', 'Domain difference noted.')}"
        else:
            task_status = "insufficient_evidence"
            task_desc = "Insufficient evidence to confirm specific research task overlap."

        task_overlap_list.append({
            "candidate_id": pid,
            "candidate_title": title,
            "status": task_status,
            "overlap_description": task_desc,
            "supporting_evidence": primary_snippet,
            "provenance": primary_chunk
        })

        # 2. Methodological comparison with usage classification
        cand_algs = cand_facts.get("algorithms", [])
        cand_alg_map = {}
        for ca in cand_algs:
            name = ca.get("name") or ca.get("family") or ""
            fam = ca.get("family") or ""
            u = ca.get("usage", "uncertain")
            ev = ca.get("evidence", "")
            keys = set(filter(None, [name.lower(), fam.lower()]))
            for k in keys:
                if k not in cand_alg_map or priority_order.index(u) < priority_order.index(cand_alg_map[k]["usage"]):
                    cand_alg_map[k] = {
                        "display_name": ca.get("variant") or ca.get("name") or fam,
                        "family": fam,
                        "usage": u,
                        "evidence": ev,
                        "item": ca
                    }

        common_keys = sorted(list(set(input_alg_map.keys()).intersection(set(cand_alg_map.keys()))))
        shared_active_algs = []
        background_algs = []
        uncertain_algs = []
        active_usages = {"proposed_method", "experiment_or_baseline"}

        for k in common_keys:
            u_inp = input_alg_map[k]["usage"]
            u_cand = cand_alg_map[k]["usage"]
            disp = cand_alg_map[k]["display_name"] or input_alg_map[k]["display_name"]

            # Exclude references_only from active and in-text claims
            if u_cand == "references_only" or u_inp == "references_only":
                continue

            if u_inp in active_usages and u_cand in active_usages:
                shared_active_algs.append(disp)
            elif u_cand == "background_or_related_work" or u_inp == "background_or_related_work":
                background_algs.append(disp)
            elif u_cand == "uncertain" or u_inp == "uncertain":
                uncertain_algs.append(disp)

        shared_active_algs = list(dict.fromkeys(shared_active_algs))
        background_algs = [a for a in dict.fromkeys(background_algs) if a not in shared_active_algs]
        uncertain_algs = [a for a in dict.fromkeys(uncertain_algs) if a not in shared_active_algs and a not in background_algs]

        cand_only_algs = [
            cand_alg_map[k]["display_name"]
            for k in cand_alg_map
            if k not in input_alg_map and cand_alg_map[k]["usage"] != "references_only"
        ]
        cand_only_algs = list(dict.fromkeys(cand_only_algs))

        input_only_algs = [
            input_alg_map[k]["display_name"]
            for k in input_alg_map
            if k not in cand_alg_map and input_alg_map[k]["usage"] != "references_only"
        ]
        input_only_algs = list(dict.fromkeys(input_only_algs))

        if shared_active_algs:
            method_status = "supported" if cand_facts.get("evidence_scope") == "full_evidence" else "partial_evidence"
        elif background_algs:
            method_status = "background_only"
        elif uncertain_algs:
            method_status = "uncertain"
        elif cand_only_algs:
            method_status = "distinct_methods"
        else:
            method_status = "insufficient_evidence"

        method_comp_list.append({
            "candidate_id": pid,
            "candidate_title": title,
            "status": method_status,
            "shared_algorithms": shared_active_algs,
            "background_algorithm_mentions": background_algs,
            "uncertain_algorithm_mentions": uncertain_algs,
            "candidate_distinct_algorithms": cand_only_algs[:5],
            "input_distinct_algorithms": input_only_algs[:5],
            "supporting_evidence": primary_snippet,
            "provenance": primary_chunk
        })

        # 3. Dataset comparison
        cand_datasets = [d.get("name") for d in cand_facts.get("datasets", [])]
        shared_ds = [d for d in cand_datasets if any(d.lower() == id_name.lower() for id_name in input_datasets)]
        cand_only_ds = [d for d in cand_datasets if not any(d.lower() == id_name.lower() for id_name in input_datasets)]

        if cand_datasets:
            ds_status = "supported" if cand_facts.get("evidence_scope") == "full_evidence" else "partial_evidence"
        else:
            ds_status = "insufficient_evidence"

        dataset_comp_list.append({
            "candidate_id": pid,
            "candidate_title": title,
            "status": ds_status,
            "shared_datasets": shared_ds,
            "candidate_distinct_datasets": cand_only_ds,
            "input_datasets": input_datasets,
            "supporting_evidence": primary_snippet,
            "provenance": primary_chunk
        })

        # 4. Experimental setup comparison
        cand_setup = cand_facts.get("experimental_setup", {})
        cand_hw = [h.get("name") for h in cand_setup.get("hardware", [])]
        cand_sw = [s.get("name") for s in cand_setup.get("software_and_frameworks", [])]
        cand_hp = [f"{hp.get('name')}={hp.get('value')}" for hp in cand_setup.get("hyperparameters", [])]

        input_hw = [h.get("name") for h in input_setup.get("hardware", [])]
        input_sw = [s.get("name") for s in input_setup.get("software_and_frameworks", [])]
        input_hp = [f"{hp.get('name')}={hp.get('value')}" for hp in input_setup.get("hyperparameters", [])]

        has_cand_setup = bool(cand_hw or cand_sw or cand_hp)
        has_input_setup = bool(input_hw or input_sw or input_hp)

        if has_cand_setup and has_input_setup:
            setup_status = "supported"
            setup_desc = f"Input setup (HW: {input_hw or 'N/A'}, SW: {input_sw or 'N/A'}) compared with candidate (HW: {cand_hw or 'N/A'}, SW: {cand_sw or 'N/A'})."
        elif has_cand_setup or has_input_setup:
            setup_status = "partial_evidence"
            setup_desc = f"Partial setup available: Input (HW: {input_hw or 'N/A'}, SW: {input_sw or 'N/A'}), Candidate (HW: {cand_hw or 'N/A'}, SW: {cand_sw or 'N/A'})."
        else:
            setup_status = "insufficient_evidence"
            setup_desc = "Neither input paper nor candidate paper reported identifiable hardware/hyperparameter setup in extracted evidence."

        setup_comp_list.append({
            "candidate_id": pid,
            "candidate_title": title,
            "status": setup_status,
            "comparison_notes": setup_desc,
            "supporting_evidence": primary_snippet,
            "provenance": primary_chunk
        })

        # 5. Metrics and results comparison
        # STRICT COMPARABILITY RULES:
        # Never compare accuracy directly with F1-score.
        # Never rank methods directly on different datasets.
        # Keep candidate metrics isolated with candidate provenance.
        cand_metrics = [
            m for m in cand_facts.get("metrics_and_results", [])
            if m.get("status") == "confirmed"
        ]

        comparable = False
        comp_reason = ""
        metric_status = "insufficient_evidence"

        if not cand_metrics:
            comp_reason = "Missing candidate quantitative metrics: candidate reports no confirmed experimental results in available evidence."
            metric_status = "insufficient_evidence"
        elif not input_metrics:
            comp_reason = "Missing input quantitative metrics: input paper reports no confirmed experimental results."
            metric_status = "insufficient_evidence"
        else:
            # Check dataset compatibility
            shared_metric_datasets = [d for d in shared_ds]
            if not shared_metric_datasets and (input_datasets or cand_datasets):
                comp_reason = (
                    f"Direct numerical ranking invalid: input evaluated on {', '.join(input_datasets) or 'unspecified'} "
                    f"while candidate evaluated on {', '.join(cand_datasets) or 'unspecified'} (different datasets)."
                )
                metric_status = "partial_evidence"
            else:
                # Check metric type compatibility
                input_types = {m.get("metric_name") for m in input_metrics}
                cand_types = {m.get("metric_name") for m in cand_metrics}
                compatible_types = input_types.intersection(cand_types)

                if compatible_types:
                    comparable = True
                    comp_reason = f"Metric types compatible ({', '.join(compatible_types)}). Direct comparison permitted subject to benchmark conditions."
                    metric_status = "supported"
                else:
                    comp_reason = (
                        f"Incompatible metric types: input reports {', '.join(input_types)} while candidate reports {', '.join(cand_types)}; "
                        f"cannot rank numerical values directly (e.g. Accuracy cannot be ranked against F1-score)."
                    )
                    metric_status = "partial_evidence"

        metric_comp_list.append({
            "candidate_id": pid,
            "candidate_title": title,
            "status": metric_status,
            "comparable": comparable,
            "comparability_reason": comp_reason,
            "input_metrics": [
                {
                    "metric_name": m.get("metric_name"),
                    "raw_value": m.get("raw_value"),
                    "unit": m.get("unit"),
                    "attribution": m.get("attribution", "uncertain"),
                    "dataset": m.get("dataset") or (input_datasets[0] if input_datasets else "unspecified")
                }
                for m in input_metrics[:5]
            ],
            "candidate_metrics": [
                {
                    "metric_name": m.get("metric_name"),
                    "raw_value": m.get("raw_value"),
                    "unit": m.get("unit"),
                    "attribution": m.get("attribution", "uncertain"),
                    "dataset": m.get("dataset") or (cand_datasets[0] if cand_datasets else "unspecified")
                }
                for m in cand_metrics[:5]
            ],
            "ranking_claim": None if not comparable else "Conditional comparison: verify train/test protocol before asserting superiority.",
            "provenance": primary_chunk
        })

        # 6. Substantive evidence-grounded claims grounded in actual usage
        # Rule 1: Shared active methods
        for alg in shared_active_algs:
            alg_k = alg.lower()
            cand_ev = cand_alg_map.get(alg_k, {}).get("evidence") or primary_snippet
            substantive_claims.append({
                "claim": f"{pid} and the input paper both actively employ {alg} (evidence confirms usage as active proposed method or experimental baseline in both papers).",
                "candidate_id": pid,
                "category": "method",
                "support_level": "supported",
                "evidence_snippet": cand_ev,
                "provenance": primary_chunk
            })

        # Rule 2: Background mentions
        for alg in background_algs:
            alg_k = alg.lower()
            cand_ev = cand_alg_map.get(alg_k, {}).get("evidence") or primary_snippet
            u_c = cand_alg_map.get(alg_k, {}).get("usage", "background_or_related_work")
            substantive_claims.append({
                "claim": f"{pid} mentions {alg} in background or related work (usage: {u_c}); it is not established as a shared active method with the input paper.",
                "candidate_id": pid,
                "category": "background_mention",
                "support_level": "supported",
                "evidence_snippet": cand_ev,
                "provenance": primary_chunk
            })

        # Rule 5: Survey context
        if is_survey:
            substantive_claims.append({
                "claim": f"{pid} is a survey paper ('{title}') synthesizing existing literature rather than proposing or experimentally benchmarking direct competing implementations.",
                "candidate_id": pid,
                "category": "survey_context",
                "support_level": "supported",
                "evidence_snippet": primary_snippet,
                "provenance": primary_chunk
            })

        # Rule 3 & 8: Uncertain algorithms preserved with uncertainty labels
        for alg in uncertain_algs:
            alg_k = alg.lower()
            u_i = input_alg_map.get(alg_k, {}).get("usage", "uncertain")
            u_c = cand_alg_map.get(alg_k, {}).get("usage", "uncertain")
            uncertainties.append(
                f"{pid}: Algorithm {alg} usage classification is uncertain (input: {u_i}, candidate: {u_c}); definitive shared-method claim withheld pending empirical confirmation."
            )

        # Datasets
        if shared_ds:
            substantive_claims.append({
                "claim": f"{pid} and the input paper both evaluate on the {', '.join(shared_ds)} dataset.",
                "candidate_id": pid,
                "category": "dataset",
                "support_level": "supported",
                "evidence_snippet": primary_snippet,
                "provenance": primary_chunk
            })
        elif cand_datasets:
            substantive_claims.append({
                "claim": f"{pid} evaluates on {', '.join(cand_datasets)}, whereas the input paper evaluates on {', '.join(input_datasets) or 'different data'}.",
                "candidate_id": pid,
                "category": "dataset",
                "support_level": "supported",
                "evidence_snippet": primary_snippet,
                "provenance": primary_chunk
            })

        # 7. Uncertainties for this candidate
        for missing in rel_info.get("missing_information", []):
            if missing and missing != "None identified":
                uncertainties.append(f"{pid}: {missing}")

    overall_status = "supported" if any(t["status"] == "supported" for t in task_overlap_list) else "partial_evidence"

    summary_text = (
        f"Comparative evaluation against {len(candidate_papers)} retrieved candidate(s) ({', '.join(p['paper_id'] for p in candidate_papers)}). "
        f"Task overlap is {overall_status}. Methodological and benchmark alignments are detailed below with explicit provenance."
    )

    return {
        "status": overall_status,
        "summary": summary_text,
        "research_task_overlap": task_overlap_list,
        "methodological_comparison": method_comp_list,
        "dataset_comparison": dataset_comp_list,
        "experimental_setup_comparison": setup_comp_list,
        "metrics_and_results_comparison": metric_comp_list,
        "substantive_claims": substantive_claims,
        "missing_information_and_uncertainties": uncertainties if uncertainties else [
            "Full experimental hyperparameters across all candidates require inspection of source code/appendices.",
            "Direct numerical performance rankings withheld due to varying benchmark protocols."
        ]
    }


def format_local_comparative_review_markdown(review: Dict[str, Any]) -> str:
    """Format structured comparative review as clean GitHub-style Markdown."""
    lines = [
        f"### Status: `{review.get('status', 'unknown')}`",
        "",
        review.get("summary", ""),
        "",
        "### 1. Research Task & Problem Overlap",
        ""
    ]

    for item in review.get("research_task_overlap", []):
        lines.append(f"- **{item['candidate_id']}** ({item.get('candidate_title', '')}) [`{item.get('status')}`]:")
        lines.append(f"  - {item.get('overlap_description', '')}")
        lines.append(f"  - *Evidence [{item.get('provenance', '')}]:* \"{item.get('supporting_evidence', '')[:200]}...\"")
        lines.append("")

    lines.extend([
        "### 2. Methodological & Architectural Comparison",
        ""
    ])
    for item in review.get("methodological_comparison", []):
        shared = ", ".join(item.get("shared_algorithms", [])) or "None (no verified shared active methods)"
        bg = ", ".join(item.get("background_algorithm_mentions", []))
        uncert = ", ".join(item.get("uncertain_algorithm_mentions", []))
        cand_diff = ", ".join(item.get("candidate_distinct_algorithms", [])) or "None identified"
        inp_diff = ", ".join(item.get("input_distinct_algorithms", [])) or "None identified"
        lines.append(f"- **{item['candidate_id']}** [`{item.get('status')}`]:")
        lines.append(f"  - Shared active methods: {shared}")
        if bg:
            lines.append(f"  - Background / related work mentions: {bg}")
        if uncert:
            lines.append(f"  - Uncertain usage mentions: {uncert}")
        lines.append(f"  - Candidate-specific: {cand_diff}")
        lines.append(f"  - Input-specific: {inp_diff}")
        lines.append("")

    lines.extend([
        "### 3. Dataset & Benchmark Comparison",
        ""
    ])
    for item in review.get("dataset_comparison", []):
        shared = ", ".join(item.get("shared_datasets", [])) or "None"
        cand_ds = ", ".join(item.get("candidate_distinct_datasets", [])) or "None"
        lines.append(f"- **{item['candidate_id']}** [`{item.get('status')}`]:")
        lines.append(f"  - Shared datasets: {shared}")
        lines.append(f"  - Candidate datasets: {cand_ds}")
        lines.append("")

    lines.extend([
        "### 4. Experimental Setup Differences",
        ""
    ])
    for item in review.get("experimental_setup_comparison", []):
        lines.append(f"- **{item['candidate_id']}** [`{item.get('status')}`]: {item.get('comparison_notes', '')}")
    lines.append("")

    lines.extend([
        "### 5. Quantitative Results & Comparability Analysis",
        ""
    ])
    for item in review.get("metrics_and_results_comparison", []):
        lines.append(f"- **{item['candidate_id']}** [`{item.get('status')}`] (Comparable: {item.get('comparable', False)}):")
        lines.append(f"  - Comparability verdict: {item.get('comparability_reason', '')}")
        if item.get("ranking_claim"):
            lines.append(f"  - Ranking claim: {item.get('ranking_claim')}")
        lines.append("")

    lines.extend([
        "### 6. Evidence-Supported Comparative Claims",
        ""
    ])
    claims = review.get("substantive_claims", [])
    if claims:
        for c in claims:
            lines.append(f"- [{c.get('support_level')}] **{c.get('candidate_id')}** ({c.get('category')}): {c.get('claim')}")
            lines.append(f"  - Provenance [{c.get('provenance')}]: \"{c.get('evidence_snippet', '')[:180]}...\"")
    else:
        lines.append("- No direct cross-paper claims could be reliably asserted from available evidence.")
    lines.append("")

    lines.extend([
        "### 7. Missing Information & Uncertainties",
        ""
    ])
    for u in review.get("missing_information_and_uncertainties", []):
        lines.append(f"- {u}")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# REPORT SAVING
# ============================================================

def save_reports(report, comparison_markdown=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = REPORT_DIR / f"review_{timestamp}.json"
    md_path = REPORT_DIR / f"review_{timestamp}.md"

    if comparison_markdown:
        if "comparative_review" not in report or report["comparative_review"] is None:
            report["comparative_review"] = comparison_markdown
        report["comparative_review_markdown"] = comparison_markdown if isinstance(comparison_markdown, str) else ""

    for path in [
        json_path,
        REPORT_DIR / "latest_review_report.json"
    ]:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

    lines = [
        "# AI-Assisted Research Paper Review",
        "",
        f"- **Input:** {report['input_file']}",
        f"- **SHA-256:** {report['sha256']}",
        ""
    ]

    if report.get("new_paper_facts"):
        lines.extend([
            "## Technical facts",
            "",
            "```json",
            json.dumps(report["new_paper_facts"], indent=2, ensure_ascii=False),
            "```",
            ""
        ])

    lines.extend([
        "## Research profile",
        "",
        "```json",
        json.dumps(report["research_profile"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Retrieved candidate papers",
        ""
    ])

    for paper in report["candidate_papers"]:
        lines.extend([
            f"### {paper['title']}",
            f"- Paper ID: {paper['paper_id']}",
            f"- Best vector distance: {paper['best_vector_distance']}",
            ""
        ])

    if report.get("relevance_assessment"):
        lines.extend([
            "## Relevance screening",
            "",
            "```json",
            json.dumps(
                report["relevance_assessment"],
                indent=2,
                ensure_ascii=False
            ),
            "```",
            ""
        ])

    if comparison_markdown:
        md_text = comparison_markdown if isinstance(comparison_markdown, str) else format_local_comparative_review_markdown(comparison_markdown)
        lines.extend([
            "## Comparative research review",
            "",
            md_text,
            ""
        ])

    lines.extend([
        "## Limitations",
        "",
        "- The review is limited to the supplied paper and retrieved corpus.",
        "- A retrieval miss can exclude a relevant paper from comparison.",
        "- Novelty cannot be conclusively established from a small corpus.",
        "- Experimental claims require verification against source evidence."
    ])

    markdown = "\n".join(lines)

    for path in [
        md_path,
        REPORT_DIR / "latest_review_report.md"
    ]:
        with open(path, "w", encoding="utf-8") as file:
            file.write(markdown)

    return json_path, md_path


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-assisted research paper review with local retrieval"
    )

    parser.add_argument("pdf", help="Path to the research PDF")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Extract and retrieve locally; skip all LLM calls"
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("\n1. Extracting PDF text...")
    pages = extract_pdf(pdf_path)

    full_text = "\n".join(page["text"] for page in pages)

    if len(full_text.strip()) < 100:
        raise ValueError(
            "Insufficient text extracted. The PDF may need OCR."
        )

    title_candidate, abstract, introduction, _ = (
        extract_title_abstract_intro(pages)
    )

    file_hash = sha256_file(pdf_path)

    print("2. Checking exact duplicates...")
    duplicate = find_exact_duplicate(pdf_path, file_hash)

    print("3. Extracting technical facts locally...")
    technical_facts = extract_technical_facts(
        pages=pages, full_text=full_text
    )

    print("4. Loading local embedding model and ChromaDB...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    db_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    try:
        collection = db_client.get_collection(COLLECTION_NAME)
    except Exception as error:
        raise RuntimeError(
            f"Cannot open ChromaDB collection '{COLLECTION_NAME}'. "
            f"Check your existing index. Details: {error}"
        )

    print("5. Retrieving candidate papers locally...")
    query_text = (
        f"Title candidate:\n{title_candidate}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Research introduction:\n{introduction[:6000]}"
    )

    retrieved_chunks = retrieve_chunks(
        collection, model, query_text
    )

    source_paper_id = identify_source_paper_id(pdf_path, file_hash)

    candidates = group_and_rank_papers(
        retrieved_chunks, load_paper_titles(), exclude_paper_id=source_paper_id
    )[:TOP_PAPERS]

    print("6. Extracting structured facts for candidate papers...")
    candidates = enrich_candidate_papers_with_facts(candidates)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(pdf_path),
        "sha256": file_hash,
        "source_paper_id": source_paper_id,
        "exact_duplicate": duplicate,
        "title_candidate": title_candidate,
        "abstract": abstract,
        "new_paper_facts": technical_facts,
        "retrieved_chunk_count": len(retrieved_chunks),
        "candidate_papers": candidates,
        "research_profile": None,
        "relevance_assessment": None,
        "comparative_review": None,
        "llm_model": LLM_MODEL
    }

    comparison_markdown = None

    if args.local_only:
        print("7. Local-only mode: generating deterministic research profile...")
        profile = generate_local_research_profile(
            title_candidate=title_candidate,
            abstract=abstract,
            introduction=introduction,
            technical_facts=technical_facts
        )
        report["research_profile"] = profile

        print("8. Local-only mode: screening candidate relevance deterministically...")
        relevance = generate_local_relevance_assessment(
            profile=profile, candidate_papers=candidates
        )
        report["relevance_assessment"] = relevance

        relevance_by_id = {
            item.get("paper_id"): item
            for item in relevance.get("matches", [])
        }

        for paper in candidates:
            paper["relevance_assessment"] = relevance_by_id.get(
                paper["paper_id"], {}
            )

        selected = [
            paper for paper in candidates
            if paper.get("relevance_assessment", {}).get("relevance")
            in ("high", "medium")
        ]
        report["selected_paper_ids"] = [
            paper["paper_id"] for paper in selected
        ]

        print("9. Local-only mode: generating deterministic comparative review...")
        comp_review = generate_local_comparative_review(
            profile=profile,
            technical_facts=technical_facts,
            candidate_papers=selected if selected else candidates,
            relevance_assessment=relevance
        )
        report["comparative_review"] = comp_review
        comparison_markdown = format_local_comparative_review_markdown(comp_review)
        report["comparative_review_markdown"] = comparison_markdown
        print("   Local-only mode: comparative review completed without external API calls.")
    else:
        client = get_client()

        print("7. Extracting research profile with LLM...")
        profile = extract_research_profile(
            client,
            title_candidate,
            abstract,
            introduction,
            technical_facts=technical_facts
        )
        report["research_profile"] = profile

        print("8. Checking candidate relevance with LLM...")
        relevance = select_relevant_papers(
            client, profile, candidates
        )
        report["relevance_assessment"] = relevance

        relevance_by_id = {
            item.get("paper_id"): item
            for item in relevance.get("matches", [])
        }

        for paper in candidates:
            paper["relevance_assessment"] = relevance_by_id.get(
                paper["paper_id"], {}
            )

        selected = [
            paper for paper in candidates
            if paper.get("relevance_assessment", {}).get("relevance")
            in ("high", "medium")
        ]

        # If relevance screening rejects every candidate, do not
        # silently pretend unrelated papers are strong matches.
        if selected:
            print("9. Generating comparative review...")
            comparison_markdown = analyze_comparison(
                client, profile, selected, technical_facts=technical_facts
            )
        else:
            comparison_markdown = (
                "## Insufficient relevant evidence\n\n"
                "The retrieved candidates did not contain enough "
                "evidence to establish a meaningful research-problem "
                "match. Expand the corpus or improve retrieval before "
                "drawing conclusions about novelty or prior methods."
            )

        report["selected_paper_ids"] = [
            paper["paper_id"] for paper in selected
        ]

    json_path, md_path = save_reports(
        report, comparison_markdown
    )

    print("\n" + "=" * 55)
    print("RESEARCH PAPER REVIEW COMPLETE")
    print("=" * 55)
    print(f"Input: {pdf_path}")
    print(
        f"Technical facts: {len(technical_facts['algorithms'])} algorithms, "
        f"{len(technical_facts['techniques_and_architectures'])} techniques, "
        f"{len(technical_facts['metrics_and_results'])} metric results"
    )
    print(f"Retrieved chunks: {len(retrieved_chunks)}")
    print(f"Candidate papers: {len(candidates)}")

    for index, paper in enumerate(candidates, start=1):
        print(f"{index}. {paper['title']} [{paper['paper_id']}]")

    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
