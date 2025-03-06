import os
# import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter


class OpenAIAPI:
    """Handles embeddings, vector storage, and retrieval."""

    load_dotenv()

    def __init__(self, documents: str, embedding_model: str = "text-embedding-3-large", persist_dir=".chroma"):

        # self._logger = logging.getLogger(__name__)
        # self._logger.info("Initializing OpenAIAPI")
        self.embedding_model = embedding_model
        self.documents = documents
        self.persist_dir = persist_dir

        # Check for API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OpenAI API Key not set in environment")

        # Initialize Embeddings
        embeddings = OpenAIEmbeddings(model=self.embedding_model)

        # Initialize Chroma Vector Store
        self.vector_store = Chroma(
            collection_name="text_collection",
            embedding_function=embeddings,
            persist_directory=self.persist_dir
        )

        # Initialize Text Splitter
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separator=".",
            length_function=len
        )

        # Split Documents into Chunks
        if isinstance(self.documents, str):
            self.documents = [self.documents]

        # Create Document Objects
        text_chunks = text_splitter.create_documents(self.documents)

        # Add Documents to Chroma
        self.vector_store.add_documents(text_chunks)
         
        # Define Retriever
        self.retriever = self.vector_store.as_retriever()

    def get_retriever(self):
        """Returns the vector database retriever."""
        return self.retriever
