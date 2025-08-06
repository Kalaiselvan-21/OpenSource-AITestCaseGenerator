"""
Knowledge Ingestion

This module provides functionality to ingest knowledge from text files.
"""

import os
import logging
from typing import List, Dict, Any, Union

from .text_connector import TextConnector

# Configure logging
logger = logging.getLogger('ai-test-generator')

class KnowledgeIngestion:
    """
    Class for ingesting knowledge from text files
    """
    
    def __init__(self):
        """
        Initialize the knowledge ingestion pipeline
        """
        self.text_connector = TextConnector()
    
    def ingest_from_source(self, source: str) -> str:
        """
        Ingest knowledge from a source (file path)
        
        Args:
            source (str): Source file path
            
        Returns:
            str: Extracted content
        """
        try:
            # Handle file path
            return self.text_connector.get_file_content(source)
        
        except Exception as e:
            logger.error(f"Failed to ingest from source {source}: {str(e)}")
            raise
    
    def ingest_from_multiple_sources(self, sources: List[str]) -> Dict[str, str]:
        """
        Ingest knowledge from multiple sources
        
        Args:
            sources (List[str]): List of source paths
            
        Returns:
            Dict[str, str]: Dictionary mapping sources to their content
        """
        results = {}
        
        for source in sources:
            try:
                content = self.ingest_from_source(source)
                results[source] = content
            except Exception as e:
                logger.error(f"Failed to ingest from source {source}: {str(e)}")
                results[source] = f"ERROR: {str(e)}"
        
        return results
