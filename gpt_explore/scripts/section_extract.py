import os
import re
import json

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\mainproject\gpt_explore"

TEXT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "extracted_text"
)

SECTION_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "sections"
)

os.makedirs(SECTION_DIR, exist_ok=True)


# ============================================================
# SECTION HEADINGS
# ============================================================

SECTION_PATTERNS = {

    "abstract": [
        r"^\s*abstract\s*$"
    ],

    "introduction": [
        r"^\s*1[\.\s]+introduction\s*$",
        r"^\s*introduction\s*$"
    ],

    "related_work": [
        r"^\s*\d*[\.\s]*(related work|related works)\s*$",
        r"^\s*\d*[\.\s]*(literature review|literature survey)\s*$",
        r"^\s*\d*[\.\s]*background\s*$"
    ],

    "methodology": [
        r"^\s*\d*[\.\s]*(methodology|method|methods)\s*$",
        r"^\s*\d*[\.\s]*(proposed method|proposed approach)\s*$",
        r"^\s*\d*[\.\s]*(materials and methods)\s*$"
    ],

    "dataset": [
        r"^\s*\d*[\.\s]*(dataset|datasets)\s*$",
        r"^\s*\d*[\.\s]*(data collection|data description)\s*$"
    ],

    "experiments": [
        r"^\s*\d*[\.\s]*(experiments|experimental setup)\s*$",
        r"^\s*\d*[\.\s]*(implementation)\s*$"
    ],

    "results": [
        r"^\s*\d*[\.\s]*(results|experimental results)\s*$",
        r"^\s*\d*[\.\s]*(results and discussion)\s*$"
    ],

    "discussion": [
        r"^\s*\d*[\.\s]*discussion\s*$"
    ],

    "limitations": [
        r"^\s*\d*[\.\s]*(limitations|limitations and future work)\s*$"
    ],

    "conclusion": [
        r"^\s*\d*[\.\s]*(conclusion|conclusions)\s*$",
        r"^\s*\d*[\.\s]*(conclusion and future work)\s*$"
    ],

    "references": [
        r"^\s*(references|bibliography)\s*$",
        r"^\s*\d*[\.\s]*(references|bibliography)\s*$"
    ]
}


# ============================================================
# FIND SECTION
# ============================================================

def identify_section(line):

    cleaned = line.strip()

    if not cleaned:
        return None

    # Avoid treating very long lines as headings
    if len(cleaned) > 100:
        return None

    for section_name, patterns in SECTION_PATTERNS.items():

        for pattern in patterns:

            if re.match(
                pattern,
                cleaned,
                re.IGNORECASE
            ):
                return section_name

    return None


# ============================================================
# EXTRACT SECTIONS
# ============================================================

def extract_sections(text):

    sections = {

        "title": "",
        "abstract": "",
        "introduction": "",
        "related_work": "",
        "methodology": "",
        "dataset": "",
        "experiments": "",
        "results": "",
        "discussion": "",
        "limitations": "",
        "conclusion": "",
        "references": ""
    }

    lines = text.splitlines()

    current_section = None

    title_found = False

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        # ----------------------------------------------------
        # Try to identify section heading
        # ----------------------------------------------------

        detected_section = identify_section(
            stripped
        )

        if detected_section:

            current_section = detected_section

            continue

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        if (
            not title_found
            and current_section is None
            and len(stripped) > 10
        ):

            # Avoid obvious metadata lines
            lower = stripped.lower()

            if not any(
                word in lower
                for word in [
                    "doi:",
                    "arxiv",
                    "www.",
                    "http"
                ]
            ):

                sections["title"] = stripped

                title_found = True

                continue

        # ----------------------------------------------------
        # Add content to current section
        # ----------------------------------------------------

        if current_section:

            sections[current_section] += (
                stripped + "\n"
            )

    return sections


# ============================================================
# PROCESS FILES
# ============================================================

text_files = [
    file
    for file in os.listdir(TEXT_DIR)
    if file.lower().endswith(".txt")
]

print("=" * 60)
print("AI/ML PAPER SECTION EXTRACTION")
print("=" * 60)

print(f"Text files found: {len(text_files)}")
print()

successful = 0


for index, filename in enumerate(
    text_files,
    start=1
):

    text_path = os.path.join(
        TEXT_DIR,
        filename
    )

    print(
        f"[{index}/{len(text_files)}] "
        f"{filename}"
    )

    try:

        with open(
            text_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        sections = extract_sections(
            text
        )

        # ----------------------------------------------------
        # Determine Paper ID
        # ----------------------------------------------------

        base_name = os.path.splitext(
            filename
        )[0]

        # Example:
        # PAPER_001_Title
        # -> PAPER_001

        match = re.search(
            r"(PAPER_\d+)",
            base_name,
            re.IGNORECASE
        )

        if match:

            paper_id = match.group(1).upper()

        else:

            paper_id = base_name

        sections["paper_id"] = paper_id

        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

        output_file = os.path.join(
            SECTION_DIR,
            paper_id + ".json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                sections,
                file,
                indent=4,
                ensure_ascii=False
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        detected = []

        for key, value in sections.items():

            if (
                key != "paper_id"
                and value.strip()
            ):
                detected.append(key)

        print(
            f"   ✓ Sections detected: "
            f"{len(detected)}"
        )

        successful += 1

    except Exception as e:

        print(
            f"   ❌ Error: {e}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SECTION EXTRACTION COMPLETE")
print("=" * 60)

print(f"Processed: {successful}")

print()
print("Output location:")
print(SECTION_DIR)

print("=" * 60)