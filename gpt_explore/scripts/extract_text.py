import os
import fitz  # PyMuPDF

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\mainproject\gpt_explore"

PDF_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "papers"
)

TEXT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "extracted_text"
)

# Create output directory if it doesn't exist
os.makedirs(TEXT_DIR, exist_ok=True)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path):

    text = ""

    try:

        document = fitz.open(pdf_path)

        for page in document:

            page_text = page.get_text()

            if page_text:
                text += page_text
                text += "\n"

        document.close()

    except Exception as e:

        print(f"Error reading {pdf_path}")
        print(e)
        return None

    return text


# ============================================================
# PROCESS ALL PDFs
# ============================================================

pdf_files = [
    file
    for file in os.listdir(PDF_DIR)
    if file.lower().endswith(".pdf")
]

print("=" * 60)
print("AI/ML RESEARCH PAPER TEXT EXTRACTION")
print("=" * 60)

print(f"PDFs found: {len(pdf_files)}")
print()


successful = 0
failed = 0


for index, filename in enumerate(pdf_files, start=1):

    pdf_path = os.path.join(
        PDF_DIR,
        filename
    )

    # Remove .pdf extension
    base_name = os.path.splitext(filename)[0]

    text_filename = base_name + ".txt"

    text_path = os.path.join(
        TEXT_DIR,
        text_filename
    )

    print(
        f"[{index}/{len(pdf_files)}] "
        f"{filename}"
    )

    extracted_text = extract_text_from_pdf(
        pdf_path
    )

    if extracted_text is None:

        failed += 1

        print("   ❌ Failed")

        continue

    # Check whether meaningful text was extracted
    if len(extracted_text.strip()) < 100:

        failed += 1

        print(
            "   ⚠ Very little text extracted"
        )

        continue

    # Save extracted text
    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(extracted_text)

    successful += 1

    print(
        f"   ✓ Extracted "
        f"({len(extracted_text):,} characters)"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("TEXT EXTRACTION COMPLETE")
print("=" * 60)

print(f"Total PDFs : {len(pdf_files)}")
print(f"Successful : {successful}")
print(f"Failed     : {failed}")

print()
print("Extracted text location:")
print(TEXT_DIR)

print("=" * 60)