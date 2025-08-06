"""
Secrets Manager Utility

This module provides functionality to securely retrieve secrets from AWS Secrets Manager.
"""

import json
import logging
import os
import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger('ai-test-generator')

class SecretsManager:
    """
    Class for retrieving secrets from AWS Secrets Manager
    """
    
    def __init__(self, region_name=None):
        """
        Initialize the secrets manager
        
        Args:
            region_name (str, optional): AWS region name. If not provided, will use AWS_REGION from environment or default to us-east-1.
        """
        self.region_name = region_name or os.environ.get('AWS_REGION', 'us-east-1')
        self.client = boto3.client('secretsmanager', region_name=self.region_name)
        logger.info(f"Initialized SecretsManager with region {self.region_name}")
    
    def get_secret(self, secret_name=None):
        """
        Get a secret from AWS Secrets Manager
        
        Args:
            secret_name (str, optional): Name of the secret. If not provided, will use AWS_SECRET_NAME from environment.
            
        Returns:
            dict: Secret key-value pairs
        """
        # Get secret name from environment if not provided
        secret_name = secret_name or os.environ.get('AWS_SECRET_NAME')
        
        if not secret_name:
            logger.error("Secret name not provided and AWS_SECRET_NAME not set in environment")
            raise ValueError("Secret name must be provided either as an argument or in the AWS_SECRET_NAME environment variable")
        
        try:
            logger.info(f"Retrieving secret {secret_name}")
            response = self.client.get_secret_value(SecretId=secret_name)
            
            if 'SecretString' in response:
                secret_data = json.loads(response['SecretString'])
                logger.info(f"Successfully retrieved secret with {len(secret_data)} key(s)")
                return secret_data
            else:
                logger.warning("Secret is binary and not supported by this utility")
                return {}
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Failed to retrieve secret: {error_code} - {error_message}")
            raise
    
    @staticmethod
    def get_secret_value(secret_name=None, key=None, region_name=None):
        """
        Static method to get a specific secret value
        
        Args:
            secret_name (str, optional): Name of the secret
            key (str, optional): Key within the secret to retrieve
            region_name (str, optional): AWS region name
            
        Returns:
            str: Secret value
        """
        # Get secret name and key from environment if not provided
        secret_name = secret_name or os.environ.get('AWS_SECRET_NAME')
        key = key or os.environ.get('AWS_SECRET_KEY')
        
        if not secret_name:
            raise ValueError("Secret name must be provided either as an argument or in the AWS_SECRET_NAME environment variable")
        
        if not key:
            raise ValueError("Secret key must be provided either as an argument or in the AWS_SECRET_KEY environment variable")
        
        # Create secrets manager and get secret
        secrets_manager = SecretsManager(region_name=region_name)
        secret_data = secrets_manager.get_secret(secret_name)
        
        # Get specific key
        if key in secret_data:
            return secret_data[key]
        else:
            logger.error(f"Key '{key}' not found in secret '{secret_name}'")
            raise KeyError(f"Key '{key}' not found in secret '{secret_name}'")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Get all secrets
    try:
        secrets_manager = SecretsManager()
        secrets = secrets_manager.get_secret('ai-test-generator/api-keys')
        print(f"Retrieved {len(secrets)} secrets")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 2: Get a specific secret value
    try:
        api_key = SecretsManager.get_secret_value(
            secret_name='ai-test-generator/api-keys',
            key='OPENAI_API_KEY'
        )
        print(f"Retrieved API key: {api_key[:5]}...")
    except Exception as e:
        print(f"Error: {str(e)}")
