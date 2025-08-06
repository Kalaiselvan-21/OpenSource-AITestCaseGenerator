"""
Environment Variable Loader

This module provides functionality to load environment variables from .env files
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logger = logging.getLogger('ai-test-generator')

def load_env_variables(env_file=None):
    """
    Load environment variables from .env file
    
    Args:
        env_file (str, optional): Path to .env file. If None, looks in default locations.
        
    Returns:
        bool: True if environment variables were loaded successfully, False otherwise
    """
    try:
        # If env_file is provided, use it directly
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
            return True
            
        # Otherwise, look in default locations
        # 1. Current directory
        if os.path.exists(".env"):
            load_dotenv(".env")
            logger.info("Loaded environment variables from ./.env")
            return True
            
        # 2. Project root directory
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
            return True
            
        logger.warning("No .env file found")
        return False
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")
        return False

def check_required_env_vars(required_vars):
    """
    Check if required environment variables are set
    
    Args:
        required_vars (list): List of required environment variable names
        
    Returns:
        tuple: (bool, list) - Success status and list of missing variables
    """
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
            
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False, missing_vars
    
    return True, []

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load environment variables
    load_env_variables()
    
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    success, missing_vars = check_required_env_vars(required_vars)
    
    if success:
        print("All required environment variables are set")
    else:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
