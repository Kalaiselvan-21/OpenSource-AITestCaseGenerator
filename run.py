#!/usr/bin/env python3
"""
AI Test Case Generator - Main Entry Point

This script provides a command-line interface to start different components
of the AI Test Case Generator system.
"""

import argparse
import os
import subprocess
import sys

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI Test Case Generator')
    
    # Main command options
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start the API server')
    api_parser.add_argument('--port', type=int, default=5002, help='Port to run the API server on')
    api_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    api_parser.add_argument('--env', choices=['dev', 'prod'], default='dev', help='Environment to run in')
    
    # CLI command
    cli_parser = subparsers.add_parser('cli', help='Run the CLI tool')
    cli_parser.add_argument('--user-story', type=str, help='User story text')
    cli_parser.add_argument('--acceptance-criteria', type=str, help='Acceptance criteria text')
    cli_parser.add_argument('--output-file', type=str, help='Output file path')
    
    # Forge app command
    forge_parser = subparsers.add_parser('forge', help='Manage the Forge app')
    forge_parser.add_argument('--action', choices=['deploy', 'install', 'tunnel'], 
                             default='deploy', help='Action to perform with the Forge app')
    forge_parser.add_argument('--env', choices=['dev', 'prod'], default='dev', help='Environment to deploy to')
    
    # Knowledge command
    knowledge_parser = subparsers.add_parser('knowledge', help='Manage domain knowledge')
    knowledge_subparsers = knowledge_parser.add_subparsers(dest='knowledge_command', help='Knowledge command')
    
    # Add command to ingest knowledge
    ingest_parser = knowledge_subparsers.add_parser('ingest', help='Ingest knowledge from a text file')
    ingest_parser.add_argument('--source', type=str, required=True, help='Source file path')
    ingest_parser.add_argument('--name', type=str, help='Name for the knowledge source')
    
    # Add command to list knowledge
    list_parser = knowledge_subparsers.add_parser('list', help='List all knowledge sources')
    
    return parser.parse_args()

def start_api_server(port, debug, env):
    """Start the API server"""
    print(f"Starting API server on port {port} in {env} environment...")
    
    # Set the Python path to include our src directory
    python_path = os.path.join(os.path.dirname(__file__), 'src')
    os.environ['PYTHONPATH'] = os.path.dirname(__file__)
    os.environ['APP_ENV'] = env
    
    # Get API key from AWS Secrets Manager
    try:
        from src.utils.secrets_manager import SecretsManager
        secrets_manager = SecretsManager()
        secrets = secrets_manager.get_secret('ai-test-generator/api-keys')
        if 'OPENAI_API_KEY' in secrets:
            print("Retrieved OpenAI API key from AWS Secrets Manager")
    except Exception as e:
        print(f"Error: Could not retrieve secrets from AWS Secrets Manager: {str(e)}")
        print("API functionality requiring OpenAI embeddings will not work properly")
    
    # Create a simple Flask app directly here to avoid import issues
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from src.generators.test_case_generator import TestCaseGenerator
    from src.utils.common_utils import load_config, save_output
    from src.ingestion.knowledge_base import KnowledgeBase
    
    app = Flask(__name__)
    CORS(app)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    config = load_config(config_path)
    
    # Initialize knowledge base
    knowledge_base = KnowledgeBase()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy", 
            "service": "ai-test-case-generator-api", 
            "environment": env,
            "knowledge_items": len(knowledge_base.get_all_knowledge())
        })
    
    @app.route('/generate-test-cases', methods=['POST'])
    @app.route('/generate', methods=['POST'])  # Add this route to handle the ngrok URL path
    def generate_test_cases():
        """Generate test cases from description and acceptance criteria"""
        try:
            # Get JSON data from request
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            # Extract required fields
            description = data.get('description')
            acceptance_criteria = data.get('acceptance_criteria')
            
            if not description or not acceptance_criteria:
                return jsonify({
                    "error": "Missing required fields",
                    "required_fields": ["description", "acceptance_criteria"]
                }), 400
            
            # Get optional parameters
            model_name = data.get('model') or config.get('llm', {}).get('model', 'mistral')
            use_knowledge = data.get('use_knowledge', True)
            
            # Generate test cases
            generator = TestCaseGenerator(model_name=model_name, knowledge_base=knowledge_base)
            test_cases = generator.generate_test_cases(
                description, 
                acceptance_criteria,
                use_knowledge=use_knowledge
            )
            
            # Save output to file (optional)
            output_dir = config.get('output', {}).get('default_directory', './output')
            output_path = save_output(test_cases, output_dir)
            
            # Return response
            return jsonify({
                "success": True,
                "test_cases": test_cases,
                "output_file": output_path,
                "used_knowledge": use_knowledge and len(knowledge_base.get_all_knowledge()) > 0
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/knowledge', methods=['GET'])
    def list_knowledge():
        """List all knowledge sources"""
        try:
            knowledge_items = knowledge_base.get_all_knowledge()
            
            # Format the response
            formatted_items = []
            for item in knowledge_items:
                formatted_items.append({
                    "id": item["id"],
                    "source": item["source"],
                    "added_at": item["added_at"],
                    "metadata": item["metadata"],
                    "content_preview": item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"]
                })
            
            return jsonify({
                "success": True,
                "count": len(knowledge_items),
                "items": formatted_items
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=debug)

def run_cli_tool(user_story, acceptance_criteria, output_file):
    """Run the CLI tool"""
    print("Running CLI tool...")
    
    # Import the test case generator
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from src.generators.test_case_generator import TestCaseGenerator
    from src.utils.common_utils import save_output
    from src.ingestion.knowledge_base import KnowledgeBase
    
    # Initialize knowledge base
    knowledge_base = KnowledgeBase()
    
    # Generate test cases
    generator = TestCaseGenerator(knowledge_base=knowledge_base)
    test_cases = generator.generate_test_cases(user_story, acceptance_criteria)
    
    # Save output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(test_cases)
        print(f"Test cases saved to {output_file}")
    else:
        print("\nGenerated Test Cases:\n")
        print(test_cases)

def manage_forge_app(action, env):
    """Manage the Forge app"""
    print(f"Managing Forge app: {action} in {env} environment...")
    
    # Navigate to the Forge app directory
    forge_dir = os.path.join(os.path.dirname(__file__), 'frontend', 'forge-app')
    os.chdir(forge_dir)
    
    # Run the appropriate Forge command
    if action == 'deploy':
        if env == 'prod':
            subprocess.run(['forge', 'deploy', '--environment', 'production'])
        else:
            subprocess.run(['forge', 'deploy'])
    elif action == 'install':
        subprocess.run(['forge', 'install', '--product', 'jira'])
    elif action == 'tunnel':
        subprocess.run(['forge', 'tunnel'])

def manage_knowledge(args):
    """Manage domain knowledge"""
    from src.ingestion.knowledge_ingestion import KnowledgeIngestion
    from src.ingestion.knowledge_base import KnowledgeBase
    
    kb = KnowledgeBase()
    
    if args.knowledge_command == 'ingest':
        # Create knowledge ingestion pipeline
        ingestion = KnowledgeIngestion()
        
        # Ingest knowledge
        try:
            print(f"Ingesting knowledge from {args.source}...")
            content = ingestion.ingest_from_source(args.source)
            
            # Add to knowledge base
            metadata = {
                'name': args.name or args.source,
                'type': 'file'
            }
            
            knowledge_id = kb.add_knowledge(content, args.source, metadata)
            
            print(f"Successfully ingested knowledge from {args.source} with ID {knowledge_id}")
            print(f"Content length: {len(content)} characters")
            print(f"First 200 characters: {content[:200]}...")
        
        except Exception as e:
            print(f"Error ingesting knowledge: {str(e)}")
    
    elif args.knowledge_command == 'list':
        # List all knowledge sources
        knowledge_items = kb.get_all_knowledge()
        
        if not knowledge_items:
            print("No knowledge sources found")
        else:
            print(f"Found {len(knowledge_items)} knowledge sources:")
            for item in knowledge_items:
                print(f"ID: {item['id']}")
                print(f"Source: {item['source']}")
                print(f"Added at: {item['added_at']}")
                if 'metadata' in item and 'name' in item['metadata']:
                    print(f"Name: {item['metadata']['name']}")
                print(f"Content length: {len(item['content'])} characters")
                print(f"Preview: {item['content'][:100]}...")
                print()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.command == 'api':
        start_api_server(args.port, args.debug, args.env)
    elif args.command == 'cli':
        run_cli_tool(args.user_story, args.acceptance_criteria, args.output_file)
    elif args.command == 'forge':
        manage_forge_app(args.action, args.env)
    elif args.command == 'knowledge':
        manage_knowledge(args)
    else:
        print("Please specify a command. Use --help for more information.")

if __name__ == '__main__':
    main()
