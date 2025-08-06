from flask import Blueprint, request, jsonify
from generators.test_case_generator import TestCaseGenerator
from utils.common_utils import load_config, save_output
import os
import json

api_bp = Blueprint('api', __name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
config = load_config(config_path)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "ai-test-case-generator-api"})

@api_bp.route('/generate-test-cases', methods=['POST'])
@api_bp.route('/generate', methods=['POST'])  # Add this route to handle the ngrok URL path
def generate_test_cases():
    """
    Generate test cases from description and acceptance criteria
    
    Expected JSON payload:
    {
        "description": "User story or description text",
        "acceptance_criteria": "Acceptance criteria text",
        "model": "mistral"  # Optional
    }
    """
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
        
        # Get optional model parameter
        model_name = data.get('model') or config.get('llm', {}).get('model', 'mistral')
        
        # Generate test cases
        generator = TestCaseGenerator(model_name=model_name)
        test_cases = generator.generate_test_cases(description, acceptance_criteria)
        
        # Save output to file (optional)
        output_dir = config.get('output', {}).get('default_directory', '../../output')
        output_path = save_output(test_cases, output_dir)
        
        # Return response with indentation for better readability
        response_data = {
            "success": True,
            "test_cases": test_cases,
            "output_file": output_path
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Legacy endpoint for backward compatibility
@api_bp.route('/legacy/generate', methods=['POST'])
def generate_jira_test_cases_legacy():
    """
    Legacy endpoint for Jira integration
    
    Expected JSON payload:
    {
        "summary": "Issue summary",
        "description": "Issue description"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        summary = data.get('summary', '')
        description = data.get('description', '')
        
        # Use the test case generator
        generator = TestCaseGenerator()
        test_cases = generator.generate_test_cases(summary, description)
        
        return jsonify({"testCases": test_cases})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
