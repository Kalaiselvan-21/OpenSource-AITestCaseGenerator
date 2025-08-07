"""
FAISS Vector Store for AI Test Case Generator
Hybrid Approach: HuggingFace Embeddings + Ollama LLM
"""

import os
import logging
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# Set up logging
logger = logging.getLogger(__name__)

class HuggingFaceEmbeddingsWrapper(Embeddings):
    """
    Wrapper to make SentenceTransformer compatible with LangChain
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize HuggingFace embeddings wrapper
        
        Args:
            model_name: Name of the SentenceTransformer model
        """
        self.model_name = model_name
        print(f"[EMBEDDINGS] 🤗 Loading {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"[EMBEDDINGS] ✅ Ready")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()

class TestCaseVectorStore:
    """
    FAISS-based vector store for AI Test Case Generator
    Uses HuggingFace embeddings for high-quality semantic search
    """
    
    def __init__(self, 
                 knowledge_base_path: str = "./knowledge_base",
                 embeddings_model: str = "all-MiniLM-L6-v2",
                 vector_store_path: str = "./vector_store",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the hybrid vector store
        
        Args:
            knowledge_base_path: Path to knowledge base documents
            embeddings_model: HuggingFace model for embeddings
            vector_store_path: Path to save/load vector store
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.knowledge_base_path = knowledge_base_path
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print(f"[VECTOR_STORE] 🚀 AI-ONLY Mode Ready")
        print(f"[VECTOR_STORE] 🤗 HuggingFace + 🦙 Ollama + ⚡ FAISS")
        
        # Initialize HuggingFace embeddings
        try:
            self.embeddings = HuggingFaceEmbeddingsWrapper(embeddings_model)
            logger.info(f"Initialized HuggingFace embeddings with model {embeddings_model}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {str(e)}")
            raise RuntimeError(f"HuggingFace embeddings initialization failed: {str(e)}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Create directories
        os.makedirs(knowledge_base_path, exist_ok=True)
        os.makedirs(vector_store_path, exist_ok=True)
        
        logger.info("Hybrid TestCaseVectorStore initialized successfully")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with vector store statistics
        """
        stats = {
            "embeddings_model": self.embeddings.model_name,
            "embeddings_type": "HuggingFace",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "knowledge_base_path": self.knowledge_base_path,
            "vector_store_path": self.vector_store_path,
            "vector_store_loaded": self.vector_store is not None
        }
        
        if self.vector_store:
            try:
                # Get document count from FAISS
                stats["document_count"] = self.vector_store.index.ntotal
                stats["embedding_dimension"] = self.vector_store.index.d
            except Exception as e:
                logger.warning(f"Could not get vector store stats: {str(e)}")
                stats["document_count"] = "unknown"
                stats["embedding_dimension"] = "unknown"
        else:
            stats["document_count"] = 0
            stats["embedding_dimension"] = 384  # Default for all-MiniLM-L6-v2
            
        return stats

    def load_vector_store(self) -> bool:
        """
        Load existing vector store from disk
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    index_name="index",
                    allow_dangerous_deserialization=True
                )
                
                stats = self.get_stats()
                print(f"[VECTOR_STORE] ✅ Loaded {stats['document_count']} documents")
                logger.info(f"Loaded vector store with {stats['document_count']} documents")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False

    def create_vector_store(self) -> bool:
        """
        Create vector store from knowledge base documents
        
        Returns:
            True if created successfully, False otherwise
        """
        try:
            print(f"[VECTOR_STORE] 🔨 Creating new vector store...")
            
            # Load documents from knowledge base
            documents = self._load_documents()
            
            if not documents:
                print(f"[VECTOR_STORE] ⚠️ No documents found in {self.knowledge_base_path}")
                return False
            
            print(f"[VECTOR_STORE] 📚 Found {len(documents)} documents")
            
            # Split documents into chunks
            print(f"[VECTOR_STORE] ✂️ Splitting documents into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            print(f"[VECTOR_STORE] 📄 Created {len(chunks)} chunks")
            
            # Create vector store from documents
            print(f"[VECTOR_STORE] 🧮 Generating embeddings...")
            self.vector_store = FAISS.from_documents(
                chunks,
                self.embeddings
            )
            
            # Save vector store
            print(f"[VECTOR_STORE] 💾 Saving vector store...")
            self.vector_store.save_local(self.vector_store_path, index_name="index")
            
            stats = self.get_stats()
            print(f"[VECTOR_STORE] ✅ Created vector store with {stats['document_count']} embeddings")
            logger.info(f"Created vector store with {len(chunks)} chunks from {len(documents)} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            print(f"[VECTOR_STORE] ❌ Failed to create vector store: {str(e)}")
            return False

    def _load_documents(self) -> List[Document]:
        """
        Load documents from knowledge base directory
        
        Returns:
            List of Document objects
        """
        documents = []
        
        if not os.path.exists(self.knowledge_base_path):
            logger.warning(f"Knowledge base path does not exist: {self.knowledge_base_path}")
            return documents
        
        # Load text files
        for filename in os.listdir(self.knowledge_base_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.knowledge_base_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={
                                "filename": filename,
                                "source": file_path,
                                "type": "knowledge_base"
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
        
        return documents

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search using HuggingFace embeddings
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            return []

    def get_relevant_context(self, query: str, max_tokens: int = 1000) -> str:
        """
        Get relevant context for a query using HuggingFace semantic search
        
        Args:
            query: Search query
            max_tokens: Maximum tokens to return
            
        Returns:
            Relevant context string
        """
        if not self.vector_store:
            return "No relevant context found."
        
        try:
            # Get similar documents
            docs = self.similarity_search(query, k=3)
            
            if not docs:
                return "No relevant context found."
            
            # Combine context from similar documents
            context_parts = []
            current_tokens = 0
            
            for doc in docs:
                content = doc.page_content
                # Rough token estimation (1 token ≈ 4 characters)
                content_tokens = len(content) // 4
                
                if current_tokens + content_tokens <= max_tokens:
                    context_parts.append(f"From {doc.metadata.get('filename', 'knowledge base')}:\n{content}")
                    current_tokens += content_tokens
                else:
                    # Add partial content if it fits
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > 50:  # Only add if meaningful
                        partial_content = content[:remaining_tokens * 4]
                        context_parts.append(f"From {doc.metadata.get('filename', 'knowledge base')}:\n{partial_content}...")
                    break
            
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return "Error retrieving context."

# Global vector store instance
_vector_store_instance = None

def get_vector_store() -> TestCaseVectorStore:
    """
    Get or create the global vector store instance
    
    Returns:
        TestCaseVectorStore instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = TestCaseVectorStore()
    
    return _vector_store_instance

def initialize_vector_store() -> bool:
    """
    Initialize the vector store (load existing or create new)
    
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        vs = get_vector_store()
        
        # Try to load existing vector store first
        if vs.load_vector_store():
            return True
        
        # Create new vector store if loading failed
        if vs.create_vector_store():
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"Vector store initialization failed: {str(e)}")
        return False
