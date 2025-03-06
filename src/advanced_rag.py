from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank


class AdvancedRAG:
    """Handles query rewriting and document reranking for improved retrieval."""

    def __init__(self, llm: ChatOpenAI, retriever):
        self.llm = llm
        self.retriever = retriever

    def query_rewrite(self, query: str) -> str:
        """Rewrites the user query for better retrieval."""
        query_rewrite_prompt = f"""
        You are a helpful assistant that rewrites user queries to improve search relevance.
        Given a user query, generate one alternative query that broadens the topic while maintaining the user's intent.
        Just return the rewritten query—no explanations or formatting.

        Original query: {query}

        Rewritten query: """
        
        response = self.llm.invoke(query_rewrite_prompt)
        return response.content if hasattr(response, "content") else response

    def document_reranker(self):
        """Applies Flashrank Reranking on the retrieved documents."""
        compressor = FlashrankRerank()
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
        return self.retriever  # Store and return
