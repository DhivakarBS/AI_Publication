import os
import json

from embedding_service import EmbeddingService
from vector_service import VectorService


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "chunks"
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "storage",
    "vector_db"
)


# --------------------------------------------------
# Create services
# --------------------------------------------------

embedding_service = EmbeddingService()

vector_service = VectorService(
    VECTOR_DB_DIR
)


# --------------------------------------------------
# Process chunk files
# --------------------------------------------------

json_files = [
    file
    for file in os.listdir(CHUNKS_DIR)
    if file.endswith(".json")
]

json_files.sort()

print()
print("Chunk files found:", len(json_files))
print()


total_chunks = 0


for file_name in json_files:

    file_path = os.path.join(
        CHUNKS_DIR,
        file_name
    )

    print("=" * 60)
    print("Processing:", file_name)

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:
        print("No chunks found.")
        continue

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Generating embeddings for",
        len(texts),
        "chunks..."
    )

    embeddings = embedding_service.encode_batch(
        texts
    )

    ids = []
    documents = []
    metadatas = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        chunk_id = chunk["chunk_id"]
        paper_id = chunk["paper_id"]

        vector_id = f"{paper_id}_{chunk_id}"

        ids.append(vector_id)

        documents.append(
            chunk["text"]
        )

        metadatas.append(
            {
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "word_count": chunk["word_count"],
                "source_file": file_name
            }
        )

    vector_service.add_documents(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    total_chunks += len(chunks)

    print(
        "Stored:",
        len(chunks),
        "vectors"
    )


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print("=" * 60)
print("VECTOR INDEX BUILD COMPLETE")
print("=" * 60)

print(
    "Total chunks processed:",
    total_chunks
)

print(
    "Vectors in ChromaDB:",
    vector_service.count()
)

print(
    "Database location:",
    VECTOR_DB_DIR
)