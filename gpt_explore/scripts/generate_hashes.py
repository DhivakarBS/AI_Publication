import os
import csv
import hashlib

BASE_DIR = r"D:\mainproject\gpt_explore"

PDF_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "papers"
)

METADATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "metadata"
)

METADATA_FILE = os.path.join(
    METADATA_DIR,
    "papers.csv"
)


def calculate_sha256(file_path):
    """
    Calculate SHA-256 hash without loading
    the complete file into memory.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def main():

    os.makedirs(METADATA_DIR, exist_ok=True)

    pdf_files = sorted(
        [
            file
            for file in os.listdir(PDF_DIR)
            if file.lower().endswith(".pdf")
        ]
    )

    if not pdf_files:

        print("No PDF files found.")
        return

    # --------------------------------------------------------
    # Read existing metadata
    # --------------------------------------------------------

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

                filename = row.get(
                    "pdf_filename",
                    ""
                )

                if filename:

                    metadata[filename] = row

    # --------------------------------------------------------
    # Generate hashes
    # --------------------------------------------------------

    seen_hashes = {}

    print("=" * 60)
    print("SHA-256 PAPER HASH GENERATION")
    print("=" * 60)

    for index, filename in enumerate(
        pdf_files,
        start=1
    ):

        filepath = os.path.join(
            PDF_DIR,
            filename
        )

        print(
            f"[{index}/{len(pdf_files)}] "
            f"{filename}"
        )

        sha256 = calculate_sha256(
            filepath
        )

        print(
            f"   SHA-256: {sha256}"
        )

        # ----------------------------------------------------
        # Detect exact duplicate
        # ----------------------------------------------------

        if sha256 in seen_hashes:

            print(
                "   ⚠ EXACT DUPLICATE"
            )

            print(
                f"   Same as: "
                f"{seen_hashes[sha256]}"
            )

        else:

            seen_hashes[sha256] = filename

        # ----------------------------------------------------
        # Existing metadata?
        # ----------------------------------------------------

        if filename in metadata:

            metadata[filename]["sha256"] = sha256

        else:

            # Generate new Paper ID
            paper_number = index

            metadata[filename] = {

                "paper_id":
                    f"PAPER_{paper_number:03d}",

                "title":
                    "",

                "authors":
                    "",

                "year":
                    "",

                "doi":
                    "",

                "openalex_id":
                    "",

                "pdf_url":
                    "",

                "pdf_filename":
                    filename,

                "abstract":
                    "",

                "search_term":
                    "",

                "sha256":
                    sha256,

                "pdf_path":
                    os.path.relpath(
                        filepath,
                        BASE_DIR
                    ),

                "extracted_text_path":
                    ""
            }

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

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
        "search_term",
        "sha256",
        "pdf_path",
        "extracted_text_path"
    ]

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for filename in sorted(metadata):

            row = metadata[filename]

            writer.writerow({
                field: row.get(
                    field,
                    ""
                )
                for field in fieldnames
            })

    print()
    print("=" * 60)
    print("HASH GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Papers processed: {len(pdf_files)}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()