from typing import Any
import fitz  # PyMuPDF
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Load a SentenceTransformer model for generating embeddings with GPU failover."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model

def embed_texts(model: Any, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of text strings, converting numpy arrays to list float arrays."""
    embeddings = []
    for chunk in texts:
        embedding = model.encode(chunk).tolist()
        embeddings.append(embedding)
    return embeddings

def embed_query(model: Any, query: str) -> list[float]:
    """Embed a single query string for retrieval matrix execution."""
    embedding = model.encode(query).tolist()
    return embedding

class VectorStoreService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        
        self.embedding_model = load_embedding_model()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            length_function=len,
        )

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def add_document(self, filename: str, file_bytes: bytes):
        self.delete_document(filename)
        raw_text = self.extract_text_from_pdf(file_bytes)
        chunks = self.text_splitter.split_text(raw_text)
        
        if not chunks:
            return
            
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in range(len(chunks))]
        
        calculated_embeddings = embed_texts(self.embedding_model, chunks)
        
        self.collection.add(
            documents=chunks, 
            embeddings=calculated_embeddings, 
            ids=ids, 
            metadatas=metadatas
        )

    def delete_document(self, filename: str):
        self.collection.delete(where={"source": filename})

    def search_policies(self, query: str, limit: int = 4) -> list:
        # Generate target vector query float array manually
        query_embedding = embed_query(self.embedding_model, query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=limit
        )
        return results["documents"][0] if results["documents"] else []

    def get_all_document_names(self) -> list[str]:
        """Scans the collection metadata to extract unique filenames for the dashboard view."""
        existing_data = self.collection.get(include=["metadatas"])
        if not existing_data or not existing_data.get("metadatas"):
            return []
            
        unique_filenames = {meta["source"] for meta in existing_data["metadatas"] if "source" in meta}
        return sorted(list(unique_filenames))

hr_vector_service = VectorStoreService()