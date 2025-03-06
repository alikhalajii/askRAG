from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class RAGChain:
    """Handles retrieval and response generation."""

    def __init__(self, llm: ChatOpenAI, retriever, prompt: PromptTemplate):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt

    def format_docs(self, docs) -> str:
        """Formats retrieved documents for LLM processing."""
        return ", ".join([doc.page_content for doc in docs])

    def invoke(self, query: str):
        """Retrieves relevant documents and generates a response."""
        docs = self.retriever.get_relevant_documents(query)
        context = self.format_docs(docs)

        # Format Prompt with Context and Query
        final_prompt = self.prompt.format(context=context, query=query)
        return self.llm.invoke(final_prompt)
