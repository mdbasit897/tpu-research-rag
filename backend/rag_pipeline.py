import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


class ResearchRAG:
    def __init__(self, tpu_engine):
        self.llm = tpu_engine
        self.db_dir = "./data/chroma_db"
        self.processed_log = "./data/processed_files.txt"
        os.makedirs("./data", exist_ok=True)

        print("Loading ONNX FastEmbed model (Bypassing PyTorch entirely)...")
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("Embedding model loaded successfully!")

        self.vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings,
        )

    def _file_hash(self, file_path):
        """MD5 hash of file to detect duplicates regardless of filename."""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def _already_processed(self, file_hash):
        if not os.path.exists(self.processed_log):
            return False
        with open(self.processed_log) as f:
            return file_hash in f.read()

    def _mark_processed(self, file_hash, filename):
        with open(self.processed_log, "a") as f:
            f.write(f"{file_hash}  {filename}\n")

    def process_and_store_document(self, file_path):
        file_hash = self._file_hash(file_path)

        # Guard against duplicate ingestion (re-uploads, Streamlit reruns)
        if self._already_processed(file_hash):
            print(f"Skipping {file_path} — already in knowledge base.")
            return

        print(f"Processing document: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Larger chunks for academic papers — preserves paragraph-level context
        # that smaller chunks destroy, giving the model enough to answer from
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        self._mark_processed(file_hash, os.path.basename(file_path))
        print(f"Stored {len(chunks)} chunks in vector database.")

    def query(self, user_question):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        docs = retriever.invoke(user_question)

        if not docs:
            return "No relevant documents found in the knowledge base. Please upload a PDF first."

        # Cap context at 2000 chars — roughly 500 tokens, safe within 768 budget
        context_parts = []
        total_chars = 0
        for doc in docs:
            chunk = doc.page_content.strip()
            if total_chars + len(chunk) > 2000:
                remaining = 2000 - total_chars
                if remaining > 150:
                    context_parts.append(chunk[:remaining])
                break
            context_parts.append(chunk)
            total_chars += len(chunk)

        context = "\n\n---\n\n".join(context_parts)

        augmented_prompt = (
            f"Context from uploaded documents:\n"
            f"===\n{context}\n===\n\n"
            f"Question: {user_question}\n"
            f"Answer (based only on the context above):"
        )

        return self.llm.generate_response(augmented_prompt)