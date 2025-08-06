"""
Knowledge Base

This module provides functionality to store and retrieve domain knowledge using vector embeddings.
"""

import os
import json
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Configure logging
logger = logging.getLogger('ai-test-generator')

class KnowledgeBase:
    """
    Class for storing and retrieving domain knowledge using vector embeddings
    """
    
    def __init__(self, storage_dir: str = None, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the knowledge base
        
        Args:
            storage_dir (str): Directory to store knowledge base files
            embedding_model (str): The OpenAI embedding model to use
        """
        # Use backend/knowledge_base as default storage directory
        self.storage_dir = storage_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'knowledge_base')
        self.embedding_model = embedding_model
        
        # Debug information
        print(f"[DEBUG] Initializing KnowledgeBase with storage_dir: {self.storage_dir}")
        print(f"[DEBUG] Embedding model: {self.embedding_model}")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize embeddings
        try:
            # Get API key from AWS Secrets Manager
            try:
                # Add project root to Python path to ensure imports work correctly
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if project_root not in sys.path:
                    sys.path.append(project_root)
                
                from src.utils.secrets_manager import SecretsManager
                secrets_manager = SecretsManager()
                secrets = secrets_manager.get_secret('ai-test-generator/api-keys')
                api_key = secrets.get('OPENAI_API_KEY')
                logger.info("Retrieved OpenAI API key from AWS Secrets Manager")
                self.embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
            except Exception as secrets_error:
                logger.error(f"Failed to retrieve API key from AWS Secrets Manager: {str(secrets_error)}")
                self.embeddings = None
                
            logger.info(f"Initialized embeddings with model {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            self.embeddings = None
        
        # Check embeddings initialization
        if self.embeddings is None:
            print("[DEBUG] Embeddings not initialized. Vector store will not be available.")
        
        # Initialize vector store
        self.vector_store = None
        
        # Debug information
        print(f"[DEBUG] Vector store initialized: {self.vector_store is not None}")
        
        # Initialize knowledge items
        self.knowledge_items = []
        
        # Load existing knowledge base if available
        self.load()
    
    def add_knowledge(self, content: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add knowledge to the knowledge base
        
        Args:
            content (str): Knowledge content
            source (str): Source of the knowledge
            metadata (Dict, optional): Additional metadata
            
        Returns:
            str: ID of the added knowledge item
        """
        # Generate a unique ID
        knowledge_id = f"k{len(self.knowledge_items) + 1:04d}"
        
        # Create knowledge item
        knowledge_item = {
            'id': knowledge_id,
            'content': content,
            'source': source,
            'added_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Add to knowledge items
        self.knowledge_items.append(knowledge_item)
        
        # Add to vector store if embeddings are available
        if self.embeddings:
            try:
                # Create document for vector store
                document = Document(
                    page_content=content,
                    metadata={
                        'id': knowledge_id,
                        'source': source,
                        'added_at': knowledge_item['added_at'],
                        **(metadata or {})
                    }
                )
                
                # Add to vector store
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents([document], self.embeddings)
                    logger.info("Created new vector store")
                else:
                    self.vector_store.add_documents([document])
                    logger.info("Added document to existing vector store")
            except Exception as e:
                logger.error(f"Failed to add document to vector store: {str(e)}")
        
        # Save knowledge base
        self.save()
        
        logger.info(f"Added knowledge item {knowledge_id} from source {source}")
        return knowledge_id
    
    def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a knowledge item by ID
        
        Args:
            knowledge_id (str): ID of the knowledge item
            
        Returns:
            Dict or None: Knowledge item if found, None otherwise
        """
        for item in self.knowledge_items:
            if item['id'] == knowledge_id:
                return item
        
        return None
    
    def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge items
        
        Returns:
            List[Dict]: List of all knowledge items
        """
        return self.knowledge_items
    
    def search_knowledge(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search knowledge items by semantic similarity
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of knowledge items sorted by relevance
        """
        if not self.vector_store:
            logger.warning("Vector store is not available, returning all knowledge items")
            return self.knowledge_items[:k]
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                # Find the original knowledge item by ID
                knowledge_id = doc.metadata.get('id')
                knowledge_item = self.get_knowledge(knowledge_id)
                
                if knowledge_item:
                    formatted_results.append({
                        **knowledge_item,
                        'similarity_score': float(score)
                    })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching knowledge: {str(e)}")
            return self.knowledge_items[:k]
    
    def save(self) -> None:
        """Save the knowledge base to disk"""
        try:
            # Save knowledge items
            kb_file = os.path.join(self.storage_dir, 'knowledge_base.json')
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_items, f, indent=2)
            
            # Save vector store if it exists
            if self.vector_store:
                vector_store_path = os.path.join(self.storage_dir, 'vector_store')
                self.vector_store.save_local(vector_store_path)
                logger.info(f"Saved vector store to {vector_store_path}")
            
            logger.info(f"Saved knowledge base with {len(self.knowledge_items)} items")
        
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {str(e)}")
            raise
    
    def load(self) -> None:
        """Load the knowledge base from disk"""
        try:
            # Load knowledge items
            kb_file = os.path.join(self.storage_dir, 'knowledge_base.json')
            if os.path.exists(kb_file):
                with open(kb_file, 'r', encoding='utf-8') as f:
                    self.knowledge_items = json.load(f)
                
                logger.info(f"Loaded knowledge base with {len(self.knowledge_items)} items")
            else:
                logger.info("No existing knowledge base found, starting with empty knowledge base")
            
            # Load vector store if embeddings are available
            if self.embeddings:
                vector_store_path = os.path.join(self.storage_dir, 'vector_store')
                if os.path.exists(vector_store_path):
                    try:
                        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
                        logger.info("Loaded vector store from disk")
                    except Exception as e:
                        logger.error(f"Failed to load vector store, will recreate: {str(e)}")
                        self._recreate_vector_store()
                else:
                    logger.info("No existing vector store found, will create if needed")
                    self._recreate_vector_store()
        
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            # Start with empty knowledge base on error
            self.knowledge_items = []
            self.vector_store = None
    
    def _recreate_vector_store(self) -> None:
        """Recreate the vector store from knowledge items"""
        if not self.embeddings:
            logger.warning("Embeddings not available, cannot recreate vector store")
            return
            
        try:
            if not self.knowledge_items:
                logger.info("No knowledge items to recreate vector store")
                return
            
            # Create documents from knowledge items
            documents = []
            for item in self.knowledge_items:
                document = Document(
                    page_content=item['content'],
                    metadata={
                        'id': item['id'],
                        'source': item['source'],
                        'added_at': item['added_at'],
                        **(item.get('metadata', {}))
                    }
                )
                documents.append(document)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Recreated vector store with {len(documents)} documents")
        
        except Exception as e:
            logger.error(f"Failed to recreate vector store: {str(e)}")
            self.vector_store = None
