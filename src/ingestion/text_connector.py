"""
Text Connector

This module provides functionality to read and process text files.
"""

import os
import logging

# Configure logging
logger = logging.getLogger('ai-test-generator')

class TextConnector:
    """
    Class for reading and processing text files
    """
    
    def __init__(self):
        """Initialize the text connector"""
        pass
    
    def get_file_content(self, file_path):
        """
        Get content from a text file
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: File content
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            logger.info(f"Successfully read content from file: {file_path}")
            return content
        
        except Exception as e:
            logger.error(f"Failed to read file content: {str(e)}")
            raise
