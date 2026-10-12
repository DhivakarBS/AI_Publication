import os
import csv

from embedding_service import EmbeddingService
from vector_service import VectorService


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "storage",
    "vector_db"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "metadata",
    "papers.csv"
)


# Load paper titles
paper_titles = {}

with open(METADATA_FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:
        paper_titles[row["paper_id"]] = row["title"]


# Initialize
embedding_service = EmbeddingService()
vector_service = VectorService(VECTOR_DB_DIR)


# Query
query = input("\nEnter research topic: ")

query_embedding = embedding_service.encode(query)

results = vector_service.search(
    query_embedding,
    top_k=20
)


# Display short results


documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

# Keep the highest similarity score for each paper
unique_papers = {}

for metadata, distance in zip(metadatas, distances):
    paper_id = metadata["paper_id"]
    similarity = (1 - float(distance)) * 100

    if (
        paper_id not in unique_papers
        or similarity > unique_papers[paper_id]["similarity"]
    ):
        unique_papers[paper_id] = {
            "similarity": similarity
        }

# Display unique papers
print("\nSimilar Papers")
print("-" * 50)

ranked_papers = sorted(
    unique_papers.items(),
    key=lambda item: item[1]["similarity"],
    reverse=True
)

for rank, (paper_id, data) in enumerate(
    ranked_papers[:5], start=1
):
    title = paper_titles.get(paper_id, paper_id)

    print(f"{rank}. Paper ID: {paper_id}")
    print(f"   {title}")
    print(f"   Similarity Score: {data['similarity']:.2f}%")
    print()

