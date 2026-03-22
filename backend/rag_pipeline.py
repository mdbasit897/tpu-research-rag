import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


class ResearchRAG:
    def __init__(self, tpu_engine):
        self.llm = tpu_engine
        self.db_dir = "./data/chroma_db"

        print("Loading ONNX FastEmbed model (Bypassing PyTorch entirely)...")
        # FastEmbed natively runs on CPU/ONNX, completely isolated from TPU
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("Embedding model loaded successfully!")

        # Initialize Vector Store
        self.vectorstore = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)

    def process_and_store_document(self, file_path):
        print(f"Processing document: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        self.vectorstore.add_documents(chunks)
        print(f"Stored {len(chunks)} chunks in vector database.")

    def query(self, user_question):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(user_question)

        context = "\n\n".join([doc.page_content for doc in docs])

        augmented_prompt = f"""
        Context information is below:
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the following question:
        {user_question}
        """

        return self.llm.generate_response(augmented_prompt)
