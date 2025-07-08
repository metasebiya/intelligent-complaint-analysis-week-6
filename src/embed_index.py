
import os
from typing import List
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class ComplaintEmbedder:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "../vector_store/chroma_index"):
        """
        Initialize the embedder with a chosen model and a directory to persist the vector store.
        """
        self.model_name = model_name
        self.persist_directory = persist_directory

        print(f"📦 Loading embedding model: {self.model_name}")
        self.embedder = HuggingFaceEmbeddings(model_name=self.model_name)

    def chunk_text(self, df: pd.DataFrame,
                   text_col: str = "cleaned_narrative",
                   chunk_size: int = 300,
                   chunk_overlap: int = 50) -> List[Document]:
        """
        Chunk long complaint narratives into manageable pieces for embedding.

        Returns a list of LangChain Document objects with metadata.
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        df = df.dropna(subset=[text_col])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        documents = []
        for _, row in df.iterrows():
            chunks = text_splitter.split_text(row[text_col])
            for chunk in chunks:
                metadata = {
                    "complaint_id": row.get("complaint_id"),
                    "product": row.get("product")
                }
                documents.append(Document(page_content=chunk, metadata=metadata))

        print(f"📄 Total chunks created: {len(documents)}")
        return documents

    def build_chroma_index(self, documents: List[Document]) -> Chroma:
        """
        Embed and index chunks using ChromaDB.
        """
        print(f"🚀 Building ChromaDB index at {self.persist_directory} ...")

        os.makedirs(self.persist_directory, exist_ok=True)
        print("✅ Entered here")
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            persist_directory=self.persist_directory
        )
        print("✅ Finished the db")
        db.persist()
        print("✅ Vector store built and persisted.")
        return db

    def load_index(self) -> Chroma:
        """
        Load an existing Chroma index from disk.
        """
        print(f"📂 Loading Chroma index from {self.persist_directory} ...")
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedder
        )

    def query_index(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform a semantic search on the vector store.
        """
        db = self.load_index()
        results = db.similarity_search(query, k=k)
        return results
