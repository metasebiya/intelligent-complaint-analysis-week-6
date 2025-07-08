import os
from typing import List, Dict

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain, StuffDocumentsChain, RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline


class ChromaRetriever:
    def __init__(self, persist_directory: str, embedding_model: str = "all-MiniLM-L6-v2", top_k: int = 5):
        print("📦 Loading embeddings...")
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)

        print("📂 Loading Chroma vector DB...")
        self.db = Chroma(
            embedding_function=self.embedder,
            persist_directory=persist_directory
        )
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


class ChromaGenerator:
    def __init__(self, model_name="google/flan-t5-small", device=0):
        print(f"🤖 Loading fast LLM: {model_name} on device {device}")
        hf_pipeline = pipeline(
            "text2text-generation",  # encoder-decoder format
            model=model_name,
            device=device,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

    @staticmethod
    def format_prompt(context: str, question: str) -> str:
        return (
            "You are a financial analyst assistant for CrediTrust.\n"
            "Use only the provided complaint excerpts to answer the question below.\n"
            "If the answer isn't in the context, say you don't have enough information.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )


class RAGChromaPipeline:
    def __init__(self, persist_directory: str, top_k: int = 5, device: int = 0):
        print("🚀 Initializing Retriever...")
        self.retriever = ChromaRetriever(persist_directory, top_k=top_k)

        print("🚀 Initializing Generator...")
        self.generator = ChromaGenerator(device=device)

        print("🧠 Building Custom Prompt...")
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=ChromaGenerator.format_prompt("{context}", "{question}")
        )

        print("🔗 Wrapping LLM in chain...")
        llm_chain = LLMChain(llm=self.generator.llm, prompt=prompt_template)

        print("📚 Creating StuffDocumentsChain...")
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )

        print("✅ Creating RetrievalQA pipeline...")
        self.chain = RetrievalQA(
            retriever=self.retriever.retriever,
            combine_documents_chain=combine_documents_chain,
            return_source_documents=True
        )

    def ask(self, question: str) -> Dict:
        result = self.chain({"query": question})
        return {
            "question": question,
            "answer": result["result"],
            "sources": result["source_documents"]
        }

    def run(self, question: str) -> Dict:
        """
        Alias for ask(), used for compatibility with streamlit and other interfaces.
        """
        return self.ask(question)

    def print_response(self, resp: Dict):
        print(f"\n🔍 Q: {resp['question']}")
        print(f"\n💡 A: {resp['answer']}\n")
        print("📂 Sources:")
        for i, doc in enumerate(resp["sources"], 1):
            print(f"\n[{i}] {doc.metadata} | {doc.page_content[:200]}...")


if __name__ == "__main__":
    print("⚙️ Starting RAGChromaPipeline...")
    pipeline = RAGChromaPipeline(persist_directory="./chroma_db", top_k=5, device=0)

    question = "Why are customers complaining about BNPL charges?"
    print(f"\n🧾 Asking: {question}")

    resp = pipeline.ask(question)
    pipeline.print_response(resp)
