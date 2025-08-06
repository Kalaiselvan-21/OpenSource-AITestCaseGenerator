from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.generators.test_case_generator import TestCaseGenerator

app = Flask(__name__)
CORS(app)

# Initialize test case generator
generator = TestCaseGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        user_story = data.get('description', '')
        acceptance_criteria = data.get('acceptance_criteria', '')
        use_knowledge = data.get('use_knowledge', True)
        use_retrieval = data.get('use_retrieval', True)
        
        if not user_story or not acceptance_criteria:
            return jsonify({"error": "User story and acceptance criteria are required"}), 400
        
        # Generate test cases
        test_cases = generator.generate_test_cases(
            user_story=user_story,
            acceptance_criteria=acceptance_criteria,
            use_knowledge=use_knowledge,
            use_retrieval=use_retrieval
        )
        
        return jsonify({
            "success": True,
            "test_cases": test_cases
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for API integration testing"""
    try:
        # Check if the test case generator is initialized
        if generator:
            return jsonify({
                "status": "healthy",
                "message": "API is running and test case generator is initialized",
                "version": "1.0.0"
            })
        else:
            return jsonify({
                "status": "degraded",
                "message": "API is running but test case generator is not initialized",
                "version": "1.0.0"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "message": f"API health check failed: {str(e)}",
            "version": "1.0.0"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
