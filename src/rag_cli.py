import fire
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from openai_api import OpenAIAPI
from advanced_rag import AdvancedRAG
from rag_chain import RAGChain


class RAGCLI:
    """Command-line interface for interacting with the RAG system."""

    def __init__(self):
        """Initialize without a document. Users must load a document first."""
        self.document_text = None
        self.api_instance = None
        self.retriever = None
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=200,
            max_retries=2
        )
        self.prompt_template = PromptTemplate.from_template("""
        Use the provided context to answer the user's question with a short and precise response.
        If you do not know the answer, respond with: "No answer based on the provided context".

        context: {context}

        question: {query}

        answer: """)

        self.advanced_rag = None
        self.compression_retriever = None
        self.rag_chain = None

    def load_document_and_ask(self, document_path: str, query: str):
        """Loads a document and immediately asks a question."""
        # Load the document first
        if not os.path.exists(document_path):
            return f"Error: File '{document_path}' does not exist."

        with open(document_path, "r", encoding="utf-8") as file:
            self.document_text = file.read().strip()  # Strip empty spaces

        if not self.document_text:
            return "Error: Document is empty. Please provide a valid file."

        # Initialize OpenAIAPI for Vector Storage & Retrieval
        self.api_instance = OpenAIAPI(documents=self.document_text)
        self.retriever = self.api_instance.get_retriever()

        # Advanced RAG with Query Rewriting and Reranking
        self.advanced_rag = AdvancedRAG(self.llm, self.retriever)
        self.compression_retriever = self.advanced_rag.document_reranker()

        # Create RAG Chain
        self.rag_chain = RAGChain(
            self.llm,
            self.compression_retriever,
            self.prompt_template
        )

        # Ask the question
        return self.ask(query)

    def ask(self, query: str):
        """Ask a question based on the loaded document."""
        if not self.document_text:
            return "Error: No document loaded. Please load a document first."

        rewritten_query = self.advanced_rag.query_rewrite(query)
        docs = self.compression_retriever.invoke(rewritten_query)
        context = self.rag_chain.format_docs(docs)
        response = self.rag_chain.invoke(rewritten_query)

        # Show the result with references
        return {
            "\nQuery": query,
            "\nRewritten_query": rewritten_query,
            "\nAnswer": response.content,
            "\nReference_chunks": context.split(". ")
        }


if __name__ == "__main__":
    fire.Fire(RAGCLI)
