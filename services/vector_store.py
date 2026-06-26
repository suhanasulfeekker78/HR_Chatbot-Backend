import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

class VectorStoreService:
    def __init__(self):
        # Setup local persistent storage client
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        
        # Use sentence-transformers natively matching your specified schema
        self.embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key="not-needed-local",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        
        # Exact requirements: 350 chunk size, 50 overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Parses byte streams instantly via rapid PyMuPDF layout maps."""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def add_document(self, filename: str, file_bytes: bytes):
        """Extracts, chunks, and inserts document footprints with metadata filtering parameters."""
        # First, purge any older versions of this file if they exist to prevent duplicates
        self.delete_document(filename)
        
        raw_text = self.extract_text_from_pdf(file_bytes)
        chunks = self.text_splitter.split_text(raw_text)
        
        if not chunks:
            return
            
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

    def delete_document(self, filename: str):
        """Purges target chunks instantly out of the matrix index using source metadata criteria."""
        self.collection.delete(where={"source": filename})

    def search_policies(self, query: str, limit: int = 4) -> list:
        """Finds top context vectors to construct dynamic RAG responses."""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        # Flatten documents structure returning string segments lists
        return results["documents"][0] if results["documents"] else []

vector_service = VectorStoreService()