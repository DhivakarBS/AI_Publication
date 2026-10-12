
import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PAPERS_DIR = ROOT / "dataset" / "papers"
VECTOR_DB_DIR = ROOT / "storage" / "vector_db"
METADATA_CSV = ROOT / "dataset" / "metadata" / "papers.csv"
REPORT_DIR = ROOT / "dataset" / "metadata" / "review_reports"

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


# ============================================================
# STEP 1: RESEARCH PROFILE EXTRACTION
# ============================================================

def extract_research_profile(
    client, title_candidate, abstract, introduction
):
    paper_content = {
        "title_candidate": title_candidate,
        "abstract": abstract,
        "introduction": introduction[:8000]
    }

    instructions = """
You are an academic research-paper analysis assistant.

Identify the actual paper title, not the conference name,
journal name, event theme, author affiliation, or publication venue.

Infer the research area from the supplied content. Do not use a
predefined list of domains.

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


def group_and_rank_papers(chunks, title_lookup):
    grouped = {}

    for chunk in chunks:
        grouped.setdefault(chunk["paper_id"], []).append(chunk)

    papers = []

    for paper_id, paper_chunks in grouped.items():
        paper_chunks.sort(key=lambda item: item["distance"])

        selected_chunks = paper_chunks[:MAX_CHUNKS_PER_PAPER]

        title = (
            next(
                (
                    chunk["title"]
                    for chunk in selected_chunks
                    if chunk["title"]
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
                    "chunk_id": chunk["chunk_id"],
                    "section_type": chunk["section_type"],
                    "original_heading": chunk["original_heading"],
                    "text": chunk["text"][:3000]
                }
                for chunk in selected_chunks
            ]
        })

    papers.sort(key=lambda item: item["best_vector_distance"])

    return papers


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

def analyze_comparison(client, profile, selected_papers):
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

    response = client.responses.create(
        model=LLM_MODEL,
        instructions=instructions,
        input=json.dumps(
            {
                "new_paper_profile": profile,
                "related_papers": evidence
            },
            ensure_ascii=False
        )
    )

    return response.output_text


# ============================================================
# REPORT SAVING
# ============================================================

def save_reports(report, comparison_markdown=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = REPORT_DIR / f"review_{timestamp}.json"
    md_path = REPORT_DIR / f"review_{timestamp}.md"

    if comparison_markdown:
        report["comparative_review"] = comparison_markdown

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
        "",
        "## Research profile",
        "",
        "```json",
        json.dumps(report["research_profile"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Retrieved candidate papers",
        ""
    ]

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
        lines.extend([
            "## Comparative research review",
            "",
            comparison_markdown,
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

    print("3. Loading local embedding model and ChromaDB...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    db_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    try:
        collection = db_client.get_collection(COLLECTION_NAME)
    except Exception as error:
        raise RuntimeError(
            f"Cannot open ChromaDB collection '{COLLECTION_NAME}'. "
            f"Check your existing index. Details: {error}"
        )

    print("4. Retrieving candidate papers locally...")
    query_text = (
        f"Title candidate:\n{title_candidate}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Research introduction:\n{introduction[:6000]}"
    )

    retrieved_chunks = retrieve_chunks(
        collection, model, query_text
    )

    candidates = group_and_rank_papers(
        retrieved_chunks, load_paper_titles()
    )[:TOP_PAPERS]

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(pdf_path),
        "sha256": file_hash,
        "exact_duplicate": duplicate,
        "title_candidate": title_candidate,
        "abstract": abstract,
        "retrieved_chunk_count": len(retrieved_chunks),
        "candidate_papers": candidates,
        "research_profile": None,
        "relevance_assessment": None,
        "llm_model": LLM_MODEL
    }

    comparison_markdown = None

    if args.local_only:
        print("5. Local-only mode: skipping all API calls.")
    else:
        client = get_client()

        print("5. Extracting research profile with LLM...")
        profile = extract_research_profile(
            client,
            title_candidate,
            abstract,
            introduction
        )
        report["research_profile"] = profile

        print("6. Checking candidate relevance with LLM...")
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
            print("7. Generating comparative review...")
            comparison_markdown = analyze_comparison(
                client, profile, selected
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
    print(f"Retrieved chunks: {len(retrieved_chunks)}")
    print(f"Candidate papers: {len(candidates)}")

    for index, paper in enumerate(candidates, start=1):
        print(f"{index}. {paper['title']} [{paper['paper_id']}]")

    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
