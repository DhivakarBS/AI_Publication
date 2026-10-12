
import os
import re
import csv
import json
import hashlib
from datetime import datetime
from collections import defaultdict

import fitz  # PyMuPDF

from embedding_service import EmbeddingService
from vector_service import VectorService


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PAPERS_DIR = os.path.join(
    BASE_DIR, "dataset", "papers"
)

METADATA_FILE = os.path.join(
    BASE_DIR, "dataset", "metadata", "papers.csv"
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR, "storage", "vector_db"
)

REPORT_DIR = os.path.join(
    BASE_DIR, "dataset", "metadata", "review_reports"
)

os.makedirs(REPORT_DIR, exist_ok=True)

CHUNK_WORDS = 700
CHUNK_OVERLAP = 100

TOP_K = 5
MAX_EVIDENCE_PER_PAPER = 3
DISPLAY_TOP_PAPERS = 5


# ==================================================
# SHA-256 HASHING
# ==================================================

def calculate_sha256(file_path):
    """Calculate SHA-256 for exact file comparison."""

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


# ==================================================
# LOAD EXISTING PAPER METADATA
# ==================================================

def load_paper_metadata():
    """Map paper IDs to titles and known checksums."""

    papers = {}

    if not os.path.exists(METADATA_FILE):
        print("Warning: papers.csv not found.")
        return papers

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            paper_id = (
                row.get("paper_id")
                or row.get("id")
                or ""
            ).strip()

            if not paper_id:
                continue

            papers[paper_id] = {
                "title": (
                    row.get("title")
                    or row.get("paper_title")
                    or paper_id
                ).strip(),

                "checksum": (
                    row.get("checksum")
                    or row.get("sha256")
                    or row.get("hash")
                    or ""
                ).strip().lower()
            }

    return papers


# ==================================================
# EXACT DUPLICATE CHECK
# ==================================================

def find_exact_duplicate(new_hash, paper_metadata):
    """Check metadata hashes and existing corpus PDFs."""

    # Check hashes already recorded in metadata.
    for paper_id, metadata in paper_metadata.items():

        if metadata["checksum"] == new_hash:
            return {
                "paper_id": paper_id,
                "title": metadata["title"]
            }

    # Calculate file hashes if metadata has no match.
    if not os.path.isdir(PAPERS_DIR):
        return None

    for filename in os.listdir(PAPERS_DIR):

        if not filename.lower().endswith(".pdf"):
            continue

        existing_path = os.path.join(
            PAPERS_DIR, filename
        )

        if calculate_sha256(existing_path) == new_hash:

            paper_id = os.path.splitext(filename)[0]

            metadata = paper_metadata.get(
                paper_id, {}
            )

            return {
                "paper_id": paper_id,
                "title": metadata.get("title", paper_id)
            }

    return None


# ==================================================
# PDF TEXT EXTRACTION
# ==================================================

def extract_pdf_text(pdf_path):
    """Extract text from all pages of a PDF."""

    page_texts = []

    with fitz.open(pdf_path) as document:

        for page in document:
            page_texts.append(
                page.get_text("text")
            )

    return "\n\n".join(page_texts).strip()


# ==================================================
# TEXT CHUNKING
# ==================================================

def split_long_paragraph(text, max_words):
    """Split unusually long paragraphs."""

    words = text.split()

    if len(words) <= max_words:
        return [text]

    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def create_chunks(text):
    """Create paragraph-based chunks with overlap."""

    text = re.sub(r"[ \t]+", " ", text)

    paragraphs = re.split(
        r"\n\s*\n+", text
    )

    cleaned_paragraphs = []

    for paragraph in paragraphs:

        paragraph = re.sub(
            r"\s+", " ", paragraph
        ).strip()

        if len(paragraph) < 50:
            continue

        cleaned_paragraphs.extend(
            split_long_paragraph(
                paragraph, CHUNK_WORDS
            )
        )

    chunks = []
    current_words = []

    for paragraph in cleaned_paragraphs:

        words = paragraph.split()

        if (
            current_words
            and len(current_words) + len(words)
            > CHUNK_WORDS
        ):
            chunks.append(
                " ".join(current_words)
            )

            current_words = current_words[
                -CHUNK_OVERLAP:
            ]

        current_words.extend(words)

    if current_words:
        chunks.append(
            " ".join(current_words)
        )

    return [
        chunk for chunk in chunks
        if chunk.strip()
    ]


# ==================================================
# SAVE JSON REPORT
# ==================================================

def save_report(report):
    """Save detailed results as JSON."""

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = os.path.join(
        REPORT_DIR,
        f"review_{timestamp}.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Also maintain a convenient latest report.
    latest_path = os.path.join(
        REPORT_DIR,
        "latest_review.json"
    )

    with open(
        latest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    return latest_path


# ==================================================
# CONCISE TERMINAL SUMMARY
# ==================================================

def display_summary(
    pdf_path,
    chunks_count,
    ranked_papers,
    report_path
):

    print("\n" + "=" * 58)
    print("          RESEARCH PAPER REVIEW SUMMARY")
    print("=" * 58)

    print(
        f"Input PDF       : "
        f"{os.path.basename(pdf_path)}"
    )

    print(
        "Exact duplicate : Not detected"
    )

    print(
        f"Chunks analysed : {chunks_count}"
    )

    print(
        f"Related papers  : {len(ranked_papers)}"
    )

    print("\nTOP RELATED PAPERS")
    print("-" * 58)

    if not ranked_papers:
        print("No related papers were retrieved.")

    else:
        for rank, paper in enumerate(
            ranked_papers[:DISPLAY_TOP_PAPERS],
            start=1
        ):

            print(
                f"\n{rank}. {paper['title']}"
            )

            print(
                f"   Paper ID   : "
                f"{paper['paper_id']}"
            )

            print(
                f"   Similarity : "
                f"{paper['best_similarity_score']:.2f}%"
            )

    print("\n" + "-" * 58)

    print("\nDetailed report saved to:")

    print(report_path)

    print("=" * 58)


# ==================================================
# MAIN COMPARISON WORKFLOW
# ==================================================

def main():

    print("=" * 58)
    print("       AI RESEARCH PAPER COMPARISON")
    print("=" * 58)

    pdf_path = input(
        "\nEnter the path of the new research PDF: "
    ).strip().strip('"')

    # Validate input.
    if not os.path.isfile(pdf_path):
        print("\nError: PDF file not found.")
        return

    if not pdf_path.lower().endswith(".pdf"):
        print("\nError: Please select a PDF file.")
        return

    if not os.path.isdir(PAPERS_DIR):
        print("\nError: Existing papers folder not found.")
        return

    # Load metadata.
    paper_metadata = load_paper_metadata()

    # Calculate SHA-256.
    print("\n[1/4] Checking exact duplicates...")

    new_hash = calculate_sha256(pdf_path)

    duplicate = find_exact_duplicate(
        new_hash, paper_metadata
    )

    if duplicate:

        print("\nExact PDF duplicate detected.")
        print(
            "Matching paper:",
            duplicate["title"]
        )
        print(
            "Paper ID:",
            duplicate["paper_id"]
        )

        report = {
            "input_file": os.path.basename(pdf_path),
            "sha256": new_hash,
            "status": "exact_duplicate",
            "matching_paper": duplicate,
            "note": (
                "SHA-256 detects identical file contents. "
                "It does not detect rewritten or reformatted "
                "versions of a paper."
            )
        }

        report_path = save_report(report)

        print(
            "\nDetailed report saved to:",
            report_path
        )

        return

    print("No exact duplicate detected.")

    # Extract text.
    print("\n[2/4] Extracting PDF text...")

    try:
        text = extract_pdf_text(pdf_path)

    except Exception as error:
        print("Error reading PDF:", error)
        return

    if not text:
        print(
            "No text extracted. The PDF may be scanned "
            "and require OCR."
        )
        return

    chunks = create_chunks(text)

    if not chunks:
        print("No usable text chunks were created.")
        return

    print("Chunks created:", len(chunks))

    # Initialize existing services.
    print("\n[3/4] Loading embedding and vector services...")

    try:
        embedding_service = EmbeddingService()

        vector_service = VectorService(
            VECTOR_DB_DIR
        )

    except Exception as error:
        print("Could not initialize services:", error)
        return

    database_count = vector_service.count()

    if database_count == 0:
        print(
            "ChromaDB is empty. Run "
            "build_vector_index.py first."
        )
        return

    print("Existing vectors:", database_count)

    # Retrieve related literature.
    print("\n[4/4] Searching existing literature...")

    evidence_by_paper = defaultdict(list)

    for chunk_index, chunk_text in enumerate(
        chunks, start=1
    ):

        try:
            query_embedding = (
                embedding_service.encode(chunk_text)
            )

            results = vector_service.search(
                query_embedding,
                top_k=min(TOP_K, database_count)
            )

        except Exception as error:
            print(
                f"Search failed for chunk {chunk_index}:",
                error
            )
            return

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents, metadatas, distances
        ):

            paper_id = metadata.get(
                "paper_id", "Unknown"
            )

            # Valid because the collection uses cosine distance.
            similarity = (
                1.0 - float(distance)
            ) * 100.0

            title = paper_metadata.get(
                paper_id, {}
            ).get("title", paper_id)

            evidence_by_paper[paper_id].append({
                "paper_id": paper_id,
                "title": title,
                "similarity_score": round(
                    similarity, 2
                ),
                "source_chunk_id": metadata.get(
                    "chunk_id"
                ),
                "new_paper_chunk": chunk_index,
                "existing_passage": document
            })

        if (
            chunk_index % 10 == 0
            or chunk_index == len(chunks)
        ):
            print(
                f"Processed {chunk_index}/"
                f"{len(chunks)} chunks"
            )

    # Aggregate results by unique paper.
    ranked_papers = []

    for paper_id, evidence in evidence_by_paper.items():

        evidence.sort(
            key=lambda item: item["similarity_score"],
            reverse=True
        )

        selected_evidence = evidence[
            :MAX_EVIDENCE_PER_PAPER
        ]

        ranked_papers.append({
            "paper_id": paper_id,
            "title": selected_evidence[0]["title"],
            "best_similarity_score": (
                selected_evidence[0]["similarity_score"]
            ),
            "evidence": selected_evidence
        })

    ranked_papers.sort(
        key=lambda item: item["best_similarity_score"],
        reverse=True
    )

    # Construct detailed JSON evidence.
    report = {
        "input_file": os.path.basename(pdf_path),
        "sha256": new_hash,
        "status": "comparison_evidence_generated",
        "processed_chunks": len(chunks),
        "retrieval_method": "cosine embedding similarity",
        "existing_papers_retrieved": len(ranked_papers),
        "results": ranked_papers,
        "limitations": [
            (
                "Similarity scores are embedding-based retrieval "
                "signals, not plagiarism or originality percentages."
            ),
            (
                "Only the existing indexed literature is searched."
            ),
            (
                "Retrieved passages require deeper analysis to "
                "establish methodological or contribution-level "
                "similarities."
            )
        ],
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        )
    }

    report_path = save_report(report)

    # Display the concise summary.
    display_summary(
        pdf_path=pdf_path,
        chunks_count=len(chunks),
        ranked_papers=ranked_papers,
        report_path=report_path
    )


if __name__ == "__main__":
    main()
