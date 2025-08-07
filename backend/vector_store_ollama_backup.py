import os
import json
import pickle
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCaseVectorStore:
    """
    FAISS-based vector store for AI Test Case Generator
    Handles document ingestion, embedding, and similarity search
    """
    
    def __init__(self, 
                 knowledge_base_path: str = "./knowledge_base",
                 embeddings_model: str = "llama2",
                 vector_store_path: str = "./vector_store",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the vector store
        
        Args:
            knowledge_base_path: Path to knowledge base documents
            embeddings_model: Ollama model for embeddings
            vector_store_path: Path to save/load vector store
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.knowledge_base_path = knowledge_base_path
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings with Ollama
        self.embeddings = OllamaEmbeddings(
            model=embeddings_model,
            base_url="http://localhost:11434"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vector_store = None
        self.metadata_store = {}
        
        # Create vector store directory if it doesn't exist
        os.makedirs(vector_store_path, exist_ok=True)
        
    def load_documents(self) -> List[Document]:
        """
        Load documents from knowledge base directory
        
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            # Load text files from knowledge base
            loader = DirectoryLoader(
                self.knowledge_base_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            docs = loader.load()
            
            # Add metadata for better context
            for doc in docs:
                # Extract filename for metadata
                filename = os.path.basename(doc.metadata.get('source', ''))
                doc.metadata['filename'] = filename
                doc.metadata['doc_type'] = self._classify_document_type(filename)
                
            documents.extend(docs)
            
            logger.info(f"Loaded {len(documents)} documents from knowledge base")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            
        return documents
    
    def _classify_document_type(self, filename: str) -> str:
        """
        Classify document type based on filename
        
        Args:
            filename: Name of the file
            
        Returns:
            Document type classification
        """
        filename_lower = filename.lower()
        
        if 'test_case' in filename_lower or 'example' in filename_lower:
            return 'test_case_example'
        elif 'best_practice' in filename_lower:
            return 'best_practice'
        elif 'functionality' in filename_lower:
            return 'functionality_spec'
        else:
            return 'general'
    
    def create_vector_store(self, force_recreate: bool = False) -> bool:
        """
        Create or load FAISS vector store
        
        Args:
            force_recreate: Force recreation of vector store
            
        Returns:
            True if successful, False otherwise
        """
        vector_store_file = os.path.join(self.vector_store_path, "faiss_index")
        metadata_file = os.path.join(self.vector_store_path, "metadata.pkl")
        
        # Try to load existing vector store
        if not force_recreate and os.path.exists(vector_store_file) and os.path.exists(metadata_file):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    index_name="faiss_index"
                )
                
                with open(metadata_file, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                    
                logger.info("Loaded existing vector store")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {str(e)}")
        
        # Create new vector store
        try:
            # Load and process documents
            documents = self.load_documents()
            
            if not documents:
                logger.error("No documents found to create vector store")
                return False
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            
            # Create vector store from documents
            self.vector_store = FAISS.from_documents(
                chunks,
                self.embeddings
            )
            
            # Save vector store
            self.vector_store.save_local(
                self.vector_store_path,
                index_name="faiss_index"
            )
            
            # Save metadata
            self.metadata_store = {
                'total_chunks': len(chunks),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'created_at': str(np.datetime64('now')),
                'documents_processed': len(documents)
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logger.info(f"Created and saved vector store with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Perform similarity search in vector store
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 5) -> List[tuple]:
        """
        Perform similarity search with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} results with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to existing vector store
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Split new documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Save updated vector store
            self.vector_store.save_local(
                self.vector_store_path,
                index_name="faiss_index"
            )
            
            logger.info(f"Added {len(chunks)} new chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def get_relevant_context(self, 
                           query: str, 
                           max_tokens: int = 2000) -> str:
        """
        Get relevant context for test case generation
        
        Args:
            query: Query describing the test case requirements
            max_tokens: Maximum tokens in returned context
            
        Returns:
            Formatted context string
        """
        # Search for relevant documents
        results = self.similarity_search_with_score(query, k=10)
        
        if not results:
            return "No relevant context found."
        
        # Format context with relevance scores
        context_parts = []
        total_length = 0
        
        for doc, score in results:
            # Only include highly relevant results (adjust threshold as needed)
            if score < 0.8:  # Lower score = more similar in FAISS
                doc_text = doc.page_content.strip()
                doc_type = doc.metadata.get('doc_type', 'general')
                filename = doc.metadata.get('filename', 'unknown')
                
                context_part = f"[{doc_type.upper()} - {filename}]\n{doc_text}\n"
                
                # Check token limit (rough estimation: 1 token ≈ 4 characters)
                if total_length + len(context_part) > max_tokens * 4:
                    break
                
                context_parts.append(context_part)
                total_length += len(context_part)
        
        if not context_parts:
            return "No highly relevant context found."
        
        return "\n---\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'vector_store_initialized': self.vector_store is not None,
            'metadata': self.metadata_store.copy() if self.metadata_store else {}
        }
        
        if self.vector_store:
            try:
                stats['index_size'] = self.vector_store.index.ntotal
            except:
                stats['index_size'] = 'unknown'
        
        return stats

# Global vector store instance
vector_store_instance = None

def get_vector_store() -> TestCaseVectorStore:
    """
    Get or create global vector store instance
    
    Returns:
        TestCaseVectorStore instance
    """
    global vector_store_instance
    
    if vector_store_instance is None:
        vector_store_instance = TestCaseVectorStore()
        
        # Initialize vector store
        if not vector_store_instance.create_vector_store():
            logger.error("Failed to initialize vector store")
    
    return vector_store_instance

def initialize_vector_store(force_recreate: bool = False) -> bool:
    """
    Initialize the global vector store
    
    Args:
        force_recreate: Force recreation of vector store
        
    Returns:
        True if successful, False otherwise
    """
    global vector_store_instance
    
    vector_store_instance = TestCaseVectorStore()
    return vector_store_instance.create_vector_store(force_recreate=force_recreate)
