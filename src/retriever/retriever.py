"""
Retriever Module

This module provides functionality to create and query vector stores
for semantic search and retrieval of test cases.
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Import environment variable loader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.env_loader import load_env_variables, check_required_env_vars

# Configure logging
logger = logging.getLogger('ai-test-generator')

class Retriever:
    """
    Class for creating, querying, and managing test cases in a vector store
    """
    
    def __init__(self, vector_store_path: Optional[str] = None, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the retriever
        
        Args:
            vector_store_path (str, optional): Path to an existing vector store
            embedding_model (str): The OpenAI embedding model to use
        """
        # Load environment variables
        load_env_variables()
        
        # Check if OpenAI API key is set
        success, missing_vars = check_required_env_vars(["OPENAI_API_KEY"])
        if not success:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        
        # Load existing vector store if provided
        if vector_store_path and os.path.exists(vector_store_path):
            self.load(vector_store_path)
            logger.info(f"Loaded vector store from {vector_store_path}")
    
    def create_from_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Create a vector store from a list of texts
        
        Args:
            texts (List[str]): List of text documents
            metadatas (List[Dict], optional): Metadata for each document
        """
        try:
            documents = [
                Document(page_content=text, metadata=meta if meta else {})
                for text, meta in zip(texts, metadatas or [{}] * len(texts))
            ]
            
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created vector store with {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def create_from_documents(self, documents: List[Document]) -> None:
        """
        Create a vector store from a list of Document objects
        
        Args:
            documents (List[Document]): List of Document objects
        """
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created vector store with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def add_test_cases(self, test_cases: List[Dict[str, Any]]) -> None:
        """
        Add test cases to the vector store
        
        Args:
            test_cases (List[Dict]): List of test case dictionaries
        """
        texts = []
        metadatas = []
        
        for tc in test_cases:
            # Create a text representation of the test case
            text = f"Title: {tc.get('title', '')}\n"
            
            if 'description' in tc:
                text += f"Description: {tc['description']}\n"
                
            if 'steps' in tc:
                text += "Steps:\n"
                for i, step in enumerate(tc['steps']):
                    text += f"- Step {i+1}: {step}\n"
                    
            if 'expected_result' in tc:
                text += f"Expected Result: {tc['expected_result']}\n"
                
            # Extract metadata
            metadata = {k: v for k, v in tc.items() if k not in ['title', 'description', 'steps', 'expected_result']}
            
            texts.append(text)
            metadatas.append(metadata)
        
        # Create or update the vector store
        if self.vector_store is None:
            self.create_from_texts(texts, metadatas)
        else:
            # Add documents to existing vector store
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas)
            ]
            self.vector_store.add_documents(documents)
            
        logger.info(f"Added {len(test_cases)} test cases to the vector store")
    
    def save(self, path: str) -> None:
        """
        Save the vector store to disk
        
        Args:
            path (str): Path to save the vector store
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store has not been created yet")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            self.vector_store.save_local(path)
            logger.info(f"Saved vector store to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load a vector store from disk
        
        Args:
            path (str): Path to load the vector store from
        """
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings)
            logger.info(f"Loaded vector store from {path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform a similarity search
        
        Args:
            query (str): The query text
            k (int): Number of results to return
            
        Returns:
            List[Document]: List of similar documents
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store has not been created yet")
                
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search with scores
        
        Args:
            query (str): The query text
            k (int): Number of results to return
            
        Returns:
            List[Tuple[Document, float]]: List of documents and their similarity scores
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store has not been created yet")
                
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search with score: {str(e)}")
            raise
    
    def find_similar_test_cases(self, description: str, acceptance_criteria: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar test cases based on description and acceptance criteria
        
        Args:
            description (str): The description or user story
            acceptance_criteria (str): The acceptance criteria
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of similar test cases with their metadata
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store has not been created yet")
            
            # Combine description and acceptance criteria for better search
            query = f"{description}\n{acceptance_criteria}"
            
            # Get similar documents with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            similar_test_cases = []
            for doc, score in results:
                similar_test_cases.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return similar_test_cases
        except Exception as e:
            logger.error(f"Error finding similar test cases: {str(e)}")
            raise
    
    def find_similar_by_user_story(self, description: str, acceptance_criteria: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar test cases based on user story and acceptance criteria
        
        Args:
            description (str): The user story or description
            acceptance_criteria (str): The acceptance criteria
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of similar test cases with their metadata
        """
        return self.find_similar_test_cases(description, acceptance_criteria, k=k)
    
    def export_test_cases_to_json(self, output_path: str) -> None:
        """
        Export all test cases to a JSON file
        
        Args:
            output_path (str): Path to save the JSON file
        """
        if not self.vector_store:
            raise ValueError("Vector store has not been created yet")
            
        # Get all documents from the vector store
        documents = self.vector_store.docstore._dict.values()
        
        # Convert documents to test cases
        test_cases = []
        for doc in documents:
            # Parse the document content
            content = doc.page_content
            metadata = doc.metadata
            
            # Extract title
            title = ""
            title_match = content.split("Title:", 1)
            if len(title_match) > 1:
                title = title_match[1].split("\n", 1)[0].strip()
            
            # Extract description
            description = ""
            desc_match = content.split("Description:", 1)
            if len(desc_match) > 1:
                description = desc_match[1].split("\n", 1)[0].strip()
            
            # Extract steps
            steps = []
            steps_match = content.split("Steps:", 1)
            if len(steps_match) > 1:
                steps_section = steps_match[1].split("Expected Result:", 1)[0]
                step_lines = [line.strip()[2:].strip() for line in steps_section.split("\n") if line.strip().startswith("- Step")]
                steps = step_lines
            
            # Extract expected result
            expected_result = ""
            er_match = content.split("Expected Result:", 1)
            if len(er_match) > 1:
                expected_result = er_match[1].strip()
            
            # Create test case
            test_case = {
                "title": title,
                "description": description,
                "steps": steps,
                "expected_result": expected_result,
                **metadata
            }
            
            test_cases.append(test_case)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(test_cases, f, indent=2)
            
        logger.info(f"Exported {len(test_cases)} test cases to {output_path}")
    
    def import_test_cases_from_json(self, input_path: str) -> None:
        """
        Import test cases from a JSON file
        
        Args:
            input_path (str): Path to the JSON file
        """
        try:
            with open(input_path, 'r') as f:
                test_cases = json.load(f)
                
            self.add_test_cases(test_cases)
            logger.info(f"Imported {len(test_cases)} test cases from {input_path}")
        except Exception as e:
            logger.error(f"Error importing test cases from {input_path}: {str(e)}")
            raise
    
    def get_test_case_by_id(self, test_case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a test case by its ID
        
        Args:
            test_case_id (str): The test case ID
            
        Returns:
            Dict or None: The test case if found, None otherwise
        """
        if not self.vector_store:
            raise ValueError("Vector store has not been created yet")
            
        # Get all documents from the vector store
        documents = self.vector_store.docstore._dict.values()
        
        # Find the document with the matching ID
        for doc in documents:
            metadata = doc.metadata
            if metadata.get('id') == test_case_id:
                content = doc.page_content
                
                # Extract title
                title = ""
                title_match = content.split("Title:", 1)
                if len(title_match) > 1:
                    title = title_match[1].split("\n", 1)[0].strip()
                
                # Extract description
                description = ""
                desc_match = content.split("Description:", 1)
                if len(desc_match) > 1:
                    description = desc_match[1].split("\n", 1)[0].strip()
                
                # Extract steps
                steps = []
                steps_match = content.split("Steps:", 1)
                if len(steps_match) > 1:
                    steps_section = steps_match[1].split("Expected Result:", 1)[0]
                    step_lines = [line.strip()[2:].strip() for line in steps_section.split("\n") if line.strip().startswith("- Step")]
                    steps = step_lines
                
                # Extract expected result
                expected_result = ""
                er_match = content.split("Expected Result:", 1)
                if len(er_match) > 1:
                    expected_result = er_match[1].strip()
                
                # Create test case
                test_case = {
                    "title": title,
                    "description": description,
                    "steps": steps,
                    "expected_result": expected_result,
                    **metadata
                }
                
                return test_case
                
        return None


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a retriever
    retriever = Retriever()
    
    # Example test cases
    test_cases = [
        {
            "title": "User Login with Valid Credentials",
            "description": "Verify that users can log in with valid credentials",
            "steps": [
                "Navigate to the login page",
                "Enter valid username and password",
                "Click the login button"
            ],
            "expected_result": "User should be logged in and redirected to the dashboard",
            "category": "authentication",
            "priority": "high",
            "id": "TC001"
        },
        {
            "title": "Password Reset via Email",
            "description": "Verify that users can reset their password via email",
            "steps": [
                "Navigate to the login page",
                "Click on 'Forgot Password'",
                "Enter registered email address",
                "Click on 'Send Reset Link'",
                "Open email and click on reset link",
                "Enter new password and confirm"
            ],
            "expected_result": "Password should be updated and user should be able to login with new password",
            "category": "authentication",
            "priority": "medium",
            "id": "TC002"
        }
    ]
    
    # Add test cases
    retriever.add_test_cases(test_cases)
    
    # Save vector store
    retriever.save("./test_case_store")
    
    # Find similar test cases
    description = "As a user, I want to log in to the system"
    acceptance_criteria = "The system should validate my credentials and redirect me to the dashboard"
    
    similar_test_cases = retriever.find_similar_by_user_story(description, acceptance_criteria)
    
    print("Similar test cases:")
    for i, test_case in enumerate(similar_test_cases):
        print(f"{i+1}. {test_case['content']}")
        print(f"   Metadata: {test_case['metadata']}")
        print(f"   Similarity Score: {test_case['similarity_score']}")
        print()
    
    # Export test cases to JSON
    retriever.export_test_cases_to_json("./test_cases.json")
