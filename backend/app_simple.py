from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Simple backend is healthy'
    })

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    """Simple endpoint to generate test cases without dependencies"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Print received data for debugging
        print(f"Received data: {data}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract fields
        description = data.get('description', '')
        acceptance_criteria = data.get('acceptance_criteria', '')
        
        # Print extracted fields
        print(f"Description: {description[:50]}...")
        print(f"Acceptance Criteria: {acceptance_criteria[:50]}...")
        
        if not description or not acceptance_criteria:
            return jsonify({
                "error": f"Missing required fields. Received: description={bool(description)}, acceptance_criteria={bool(acceptance_criteria)}"
            }), 400
        
        # Extract key points from acceptance criteria
        key_points = []
        ac_lines = acceptance_criteria.split('\n')
        for line in ac_lines:
            line = line.strip()
            if line and len(line) > 10:
                key_points.append(line)
        
        # Generate test cases for each key point
        test_cases = []
        for i, point in enumerate(key_points):
            if len(point) > 50:
                title = point[:50] + "..."
            else:
                title = point
                
            test_case = {
                "id": f"TC-{i+1:03d}",
                "title": f"Verify {title}",
                "description": f"Ensure the system correctly implements: {point}",
                "steps": [
                    "Set up test environment",
                    "Execute test actions",
                    "Verify results"
                ],
                "expected": f"System behavior matches the requirement: {point}"
            }
            test_cases.append(test_case)
        
        # Format test cases as markdown
        markdown = "# Test Cases\n\n"
        
        for tc in test_cases:
            markdown += f"## {tc['id']}: {tc['title']}\n"
            markdown += f"**Description**: {tc['description']}\n"
            markdown += "**Preconditions**: System is properly configured\n"
            markdown += "**Test Steps**:\n"
            
            for i, step in enumerate(tc['steps']):
                markdown += f"{i+1}. {step}\n"
                
            markdown += f"\n**Expected Results**: {tc['expected']}\n"
            markdown += "**Priority**: High\n\n"
        
        # Add standard test cases
        markdown += """## TC-EDGE: Test Edge Cases
**Description**: Verify behavior with edge cases
**Preconditions**: System is in a stable state
**Test Steps**:
1. Test with minimum allowed values
2. Test with maximum allowed values
3. Test with special characters

**Expected Results**: System handles edge cases gracefully
**Priority**: Medium

## TC-ERROR: Verify Error Handling
**Description**: Ensure proper error handling
**Preconditions**: System is accessible
**Test Steps**:
1. Simulate expected error conditions
2. Verify error messages

**Expected Results**: User-friendly error messages are displayed
**Priority**: High
"""
        
        return jsonify({
            "success": True,
            "test_cases": markdown
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
