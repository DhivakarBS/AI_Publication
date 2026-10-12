import os
import csv
import time
import requests
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\mainproject\gpt_explore"

PDF_DIR = os.path.join(BASE_DIR, "dataset", "papers")
METADATA_DIR = os.path.join(BASE_DIR, "dataset", "metadata")

TARGET_PAPERS = 100

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# OpenAlex API
API_URL = "https://api.openalex.org/works"

# AI/ML search terms
SEARCH_TERMS = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "computer vision",
    "natural language processing",
    "reinforcement learning",
    "generative artificial intelligence",
    "large language models"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_pdf_url(work):
    """
    Try to obtain an open-access PDF URL from an OpenAlex work.
    """

    # Primary location
    primary_location = work.get("primary_location")

    if primary_location:
        pdf_info = primary_location.get("pdf_url")

        if pdf_info:
            return pdf_info

        source = primary_location.get("source")

    # Other locations
    locations = work.get("locations", [])

    for location in locations:

        pdf_url = location.get("pdf_url")

        if pdf_url:
            return pdf_url

    return None


def clean_filename(name):
    """
    Make a safe filename for Windows.
    """

    invalid_chars = '<>:"/\\|?*'

    for char in invalid_chars:
        name = name.replace(char, "_")

    return name[:150]


def download_pdf(url, filepath):

    try:

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "AI-ML-Paper-Review-Project/1.0"
            }
        )

        if response.status_code != 200:
            return False

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        # Basic validation
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return False

        with open(filepath, "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:

        print("Download error:", e)

        return False


# ============================================================
# MAIN
# ============================================================

papers = []
seen_ids = set()

print("=" * 60)
print("AI/ML RESEARCH PAPER DATASET COLLECTION")
print("=" * 60)

for search_term in SEARCH_TERMS:

    if len(papers) >= TARGET_PAPERS:
        break

    print("\nSearching:", search_term)

    params = {
        "search": search_term,
        "filter": "type:article,from_publication_date:2018-01-01",
        "per-page": 50
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except Exception as e:

        print("API error:", e)
        continue

    for work in data.get("results", []):

        if len(papers) >= TARGET_PAPERS:
            break

        work_id = work.get("id")

        if not work_id:
            continue

        if work_id in seen_ids:
            continue

        seen_ids.add(work_id)

        # ----------------------------------------------------
        # Get PDF
        # ----------------------------------------------------

        pdf_url = get_pdf_url(work)

        if not pdf_url:
            continue

        title = work.get("title") or "Untitled"

        paper_number = len(papers) + 1

        safe_title = clean_filename(title)

        filename = (
            f"PAPER_{paper_number:03d}_{safe_title}.pdf"
        )

        filepath = os.path.join(
            PDF_DIR,
            filename
        )

        print(
            f"\n[{paper_number}/{TARGET_PAPERS}] "
            f"{title}"
        )

        success = download_pdf(
            pdf_url,
            filepath
        )

        if not success:

            print("   PDF download failed")
            continue

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        authors = []

        for authorship in work.get("authorships", []):

            author = authorship.get("author")

            if author:

                author_name = author.get("display_name")

                if author_name:
                    authors.append(author_name)

        abstract = ""

        inverted_index = work.get(
            "abstract_inverted_index"
        )

        if inverted_index:

            words = []

            for word, positions in inverted_index.items():

                for position in positions:

                    words.append(
                        (position, word)
                    )

            words.sort()

            abstract = " ".join(
                word for _, word in words
            )

        paper_info = {

            "paper_id":
                f"PAPER_{paper_number:03d}",

            "title":
                title,

            "authors":
                "; ".join(authors),

            "year":
                work.get("publication_year"),

            "doi":
                work.get("doi"),

            "openalex_id":
                work_id,

            "pdf_url":
                pdf_url,

            "pdf_filename":
                filename,

            "abstract":
                abstract,

            "search_term":
                search_term
        }

        papers.append(paper_info)

        print("   Downloaded successfully")

        time.sleep(0.2)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_file = os.path.join(
    METADATA_DIR,
    "papers.csv"
)

fieldnames = [
    "paper_id",
    "title",
    "authors",
    "year",
    "doi",
    "openalex_id",
    "pdf_url",
    "pdf_filename",
    "abstract",
    "search_term"
]

with open(
    metadata_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(papers)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print(
    f"DATASET COLLECTION COMPLETE"
)

print(
    f"Papers downloaded: {len(papers)}"
)

print(
    f"PDF folder: {PDF_DIR}"
)

print(
    f"Metadata file: {metadata_file}"
)

print("=" * 60)