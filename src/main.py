import pandas as pd
from embed_index import ComplaintEmbedder  # Ensure this class is defined as we did before

def main():
    # Step 1: Load cleaned complaints
    df = pd.read_csv("data/processed/filtered_complaints.csv")

    # Step 2: Initialize embedder
    embedder = ComplaintEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="vector_store/chroma_index"
    )

    # Step 3: Chunk complaint narratives
    print("🔍 Chunking complaints...")
    documents = embedder.chunk_text(df)

    # Step 4: Generate embeddings & persist in ChromaDB
    print("📦 Embedding and storing in ChromaDB...")
    embedder.build_chroma_index(documents)

    print("✅ Embedding and indexing completed successfully!")

if __name__ == "__main__":
    main()
