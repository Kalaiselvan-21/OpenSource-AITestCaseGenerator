import os
import yaml
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai-test-generator')

def load_config(config_path):
    """
    Load configuration from a YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration as a dictionary
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
        
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def save_output(content, output_dir, filename=None):
    """
    Save content to a file in the specified directory
    
    Args:
        content (str): Content to save
        output_dir (str): Directory to save the file
        filename (str, optional): Filename to use. If None, generates a timestamp-based name
        
    Returns:
        str: Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.txt"
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    # Save content
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        logger.info(f"Content saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving content: {str(e)}")
        return None
