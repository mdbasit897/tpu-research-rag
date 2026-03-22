import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class ResearchRAG:
    def __init__(self, tpu_engine):
        self.llm = tpu_engine
        self.db_dir = "./data/chroma_db"

        # Force the embedding model to run purely on the CPU
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

        print("Loading Embedding model onto CPU...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Embedding model loaded on CPU!")

        # Initialize Vector Store
        self.vectorstore = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)

    def process_and_store_document(self, file_path):
        print(f"Processing document: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Break large academic papers into smaller semantic chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Save to ChromaDB
        self.vectorstore.add_documents(chunks)
        print(f"Stored {len(chunks)} chunks in vector database.")

    def query(self, user_question):
        # 1. Retrieve the most relevant chunks from the database
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(user_question)

        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Build the final prompt for the TPU
        augmented_prompt = f"""
        Context information is below:
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the following question:
        {user_question}
        """

        # 3. Generate answer using the TPU
        return self.llm.generate_response(augmented_prompt)