from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        Initializes the text chunker with the given chunk size and overlap.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_documents(self, texts):
        """
        Splits each narrative into smaller overlapping chunks.

        Parameters:
        - texts: List of cleaned narrative strings

        Returns:
        - List of dictionaries with chunk_text, doc_id, and chunk_id
        """
        chunked_data = []
        for idx, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                chunks = self.splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    chunked_data.append({
                        "chunk_text": chunk,
                        "doc_id": idx,
                        "chunk_id": i
                    })
        return chunked_data
