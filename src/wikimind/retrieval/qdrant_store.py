import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding

# Local persistent storage for Qdrant
QDRANT_PATH = os.path.join(os.getcwd(), "local_qdrant")
COLLECTION_NAME = "wikimind_hybrid"

class QdrantVectorStore:
    def __init__(self):
        # Initialize local Qdrant Client
        self.client = QdrantClient(path=QDRANT_PATH)
        
        # Initialize FastEmbed for dense vector generation
        # BGE-small is extremely fast and effective for testing
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if it doesn't exist."""
        collections = [col.name for col in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            print(f"Creating collection '{COLLECTION_NAME}'...")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384, # Size for BAAI/bge-small-en-v1.5
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")

    def add_chunks(self, chunks: list):
        """
        Embeds and upserts LangChain Document chunks into Qdrant.
        """
        if not chunks:
            return

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate dense embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings_generator = self.embedding_model.embed(texts)
        embeddings = list(embeddings_generator)

        # In Qdrant, we just upsert with an ID, vector, and payload (metadata + original text)
        points = []
        for i, (vector, text, metadata) in enumerate(zip(embeddings, texts, metadatas)):
            # Combine metadata and page_content for the payload
            payload = metadata.copy()
            payload["page_content"] = text
            
            # Using a simple integer ID for now, offset by current count
            points.append({
                "id": self.client.count(COLLECTION_NAME).count + i + 1,
                "vector": vector.tolist(),
                "payload": payload
            })
            
        print(f"Upserting {len(points)} points to Qdrant...")
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print("Upsert complete.")

if __name__ == "__main__":
    print("Testing Qdrant Local Store Initialization...")
    store = QdrantVectorStore()
    print(f"Total documents in collection: {store.client.count(COLLECTION_NAME).count}")
