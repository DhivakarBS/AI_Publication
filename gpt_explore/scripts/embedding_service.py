from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:

    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Embedding model loaded successfully.")

    def encode(self, text):
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def encode_batch(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.tolist()


if __name__ == "__main__":

    service = EmbeddingService()

    text = "Deep learning is used for image classification."

    embedding = service.encode(text)

    print("Embedding generated.")
    print("Embedding dimension:", len(embedding))