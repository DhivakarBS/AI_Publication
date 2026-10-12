
import chromadb


class VectorService:

    def __init__(self, database_path):
        self.client = chromadb.PersistentClient(
            path=database_path
        )

        self.collection = self.client.get_or_create_collection(
            name="ai_ml_papers",
            metadata={"hnsw:space": "cosine"}
        )

        print("ChromaDB collection ready.")

    def add_documents(self, ids, embeddings, documents, metadatas):
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k=5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def count(self):
        return self.collection.count()
