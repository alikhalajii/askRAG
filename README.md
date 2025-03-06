# askRAG: A Retrieval-Augmented Generation (RAG) CLI Tool

`askRAG` is a **command-line tool** for **context-aware question-answering** using **Retrieval-Augmented Generation (RAG)** with OpenAI models. It allows users to load text-based documents, retrieve relevant information, and generate answers based on the content of the documents.

With `askRAG`, you can use advanced techniques like **Query Rewriting** and **Reranking** to improve the quality of the answers retrieved from the documents. This tool leverages OpenAI’s **GPT-4o-mini** model for answer generation and vector storage for document retrieval.

## 📦 Installation


### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/askRAG.git
cd askRAG
```


### 2. Prepare Your Documents
Place your document files inside the my_docs/ directory.
Optionally, you can use the files that are already present in the directory.
 

### 3. Set up your OpenAI API key:
Create a .env file inside the src/ directory and add your OpenAI API key:
 
```bash
echo "OPENAI_API_KEY='your_openai_api_key_here'" > src/.env
```

### 4. Build and start the Docker Container:

 ```bash
docker build -t ask_rag_image .
docker run -it --name ask_rag -v $(pwd)/my_docs:/app/docs ask_rag_image bash


 ```

 ### 4. Run askRAG app
Once the container is running, you can interact with askRAG by executing:

Make sure to replace YOUR_FILE with the actual file name inside the "my_docs" folder.

```bash
python rag_cli.py load_document_and_ask \
--document_path "/app/docs/YOUR_FILE" \
--query "What is this document about?"
```
-----------------------------------------
### Example:

```bash
python rag_cli.py load_document_and_ask \
--document_path "../docs/computers.txt" \
--query "What is the purpose of the MEDLARS system developed for the National Library of Medicine?"
```
### Output

**Query:** What is the purpose of the MEDLARS system developed for the National Library of Medicine?

**Rewritten_query:**  What are the functions and benefits of the MEDLARS system used by the National Library of Medicine?

**Answer:** The MEDLARS system, contracted by the U.S. Public Health Service, functions as a computerized medical literature analysis and retrieval system. Its benefits include processing several hundred thousand pieces of medical information each year, facilitating access to medical literature for researchers and improving the efficiency of medical information retrieval.

**Reference_chunks:** ["Information on 2-1/2 million patients from thirty-four states\nwill be processed by a Honeywell 400 computer to evaluate diagnostic and\nhospital care and to compare the performance of the various\ninstitutions.\n\nIn the first phase of a computerized medical literature analysis and\nretrieval system for the National Library of Medicine, the U.S", "Public\nHealth Service contracted with General Electric for a system called\nMEDLARS, MEDical Literature Analysis and Retrieval System",............

## 🛠 Features

- **Load Documents**: Load text-based documents to be used as context for answering questions.
- **Retrieval-Augmented Generation (RAG)**: Uses OpenAI embeddings and vector databases for efficient document retrieval.
- **Query Rewriting & Reranking**: Improves search relevance with AI-driven query optimization.
- **LLM Integration**: Built on OpenAI’s GPT-4o-mini model for high-quality answers.

