import os
import re
import json
import csv


BASE_DIR = r"D:\mainproject\gpt_explore"

TEXT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "extracted_text"
)

CHUNK_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "chunks"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "metadata",
    "papers.csv"
)

os.makedirs(
    CHUNK_DIR,
    exist_ok=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Remove excessive whitespace
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    # Fix spaces before punctuation
    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text
    )

    return text.strip()


# ============================================================
# PARAGRAPH EXTRACTION
# ============================================================

def get_paragraphs(text):

    text = clean_text(text)

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    paragraphs = [
        p.strip()
        for p in paragraphs
        if len(p.strip()) > 50
    ]

    return paragraphs


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(
    paragraphs,
    target_words=700,
    overlap_words=100
):

    chunks = []

    current_words = []

    for paragraph in paragraphs:

        words = paragraph.split()

        # ----------------------------------------------------
        # If paragraph itself is too large
        # ----------------------------------------------------

        if len(words) > target_words:

            # Save current chunk first
            if current_words:

                chunks.append(
                    " ".join(current_words)
                )

                current_words = []

            # Split large paragraph
            start = 0

            while start < len(words):

                end = start + target_words

                chunk = words[start:end]

                chunks.append(
                    " ".join(chunk)
                )

                start = end - overlap_words

            continue

        # ----------------------------------------------------
        # Add paragraph to current chunk
        # ----------------------------------------------------

        if (
            len(current_words)
            + len(words)
            <= target_words
        ):

            current_words.extend(
                words
            )

        else:

            # Save current chunk
            if current_words:

                chunks.append(
                    " ".join(current_words)
                )

            # Start next chunk with overlap
            if overlap_words > 0:

                overlap = (
                    current_words[
                        -overlap_words:
                    ]
                    if current_words
                    else []
                )

                current_words = (
                    overlap
                    + words
                )

            else:

                current_words = words

    # --------------------------------------------------------
    # Final chunk
    # --------------------------------------------------------

    if current_words:

        chunks.append(
            " ".join(current_words)
        )

    return chunks


# ============================================================
# LOAD PAPER METADATA
# ============================================================

metadata = {}

if os.path.exists(METADATA_FILE):

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            metadata[
                row["paper_id"]
            ] = row


# ============================================================
# PROCESS PAPERS
# ============================================================

text_files = [
    file
    for file in os.listdir(TEXT_DIR)
    if file.lower().endswith(".txt")
]


print("=" * 60)
print("AI/ML PAPER CHUNKING")
print("=" * 60)

print(
    f"Text files found: {len(text_files)}"
)

total_chunks = 0


for index, filename in enumerate(
    text_files,
    start=1
):

    text_path = os.path.join(
        TEXT_DIR,
        filename
    )

    # --------------------------------------------------------
    # Determine Paper ID
    # --------------------------------------------------------

    match = re.search(
        r"(PAPER_\d+)",
        filename,
        re.IGNORECASE
    )

    if match:

        paper_id = (
            match.group(1)
            .upper()
        )

    else:

        paper_id = os.path.splitext(
            filename
        )[0]

    print(
        f"[{index}/{len(text_files)}] "
        f"{paper_id}"
    )

    # --------------------------------------------------------
    # Read text
    # --------------------------------------------------------

    with open(
        text_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    if len(text.strip()) < 100:

        print(
            "   ⚠ Not enough text"
        )

        continue

    # --------------------------------------------------------
    # Paragraphs
    # --------------------------------------------------------

    paragraphs = get_paragraphs(
        text
    )

    # --------------------------------------------------------
    # Chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        paragraphs,
        target_words=700,
        overlap_words=100
    )

    # --------------------------------------------------------
    # Save chunks
    # --------------------------------------------------------

    output = []

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        output.append({

            "paper_id":
                paper_id,

            "chunk_id":
                f"{paper_id}_CHUNK_{chunk_number:03d}",

            "text":
                chunk,

            "word_count":
                len(chunk.split())
        })

    output_file = os.path.join(
        CHUNK_DIR,
        f"{paper_id}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    total_chunks += len(chunks)

    print(
        f"   ✓ {len(chunks)} chunks"
    )


print()
print("=" * 60)
print("CHUNKING COMPLETE")
print("=" * 60)

print(
    f"Papers processed: {len(text_files)}"
)

print(
    f"Total chunks: {total_chunks}"
)

print(
    f"Output: {CHUNK_DIR}"
)