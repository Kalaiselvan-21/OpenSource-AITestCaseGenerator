from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
import traceback
from token_counter import TokenCounter

app = Flask(__name__)
CORS(app)

# Initialize token counter
token_counter = TokenCounter()

@app.route('/health', methods=['GET'])
def health_check():
    # Get token usage statistics
    token_stats = token_counter.get_usage_stats()
    
    return jsonify({
        'status': 'healthy',
        'message': 'AI Test Case Generator API is running',
        'token_usage': token_stats
    })

@app.route('/token-usage', methods=['GET'])
def token_usage():
    """Endpoint to get token usage statistics"""
    try:
        token_stats = token_counter.get_usage_stats()
        return jsonify({
            'success': True,
            'token_usage': token_stats
        })
    except Exception as e:
        print(f"Error getting token usage stats: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    start_time = time.time()
    
    try:
        data = request.json
        
        # Validate required fields
        if not data.get('description'):
            return jsonify({'error': 'Description is required'}), 400
        
        if not data.get('acceptance_criteria'):
            return jsonify({'error': 'Acceptance criteria is required'}), 400
        
        # Get parameters
        description = data.get('description')
        acceptance_criteria = data.get('acceptance_criteria')
        use_knowledge = data.get('use_knowledge', True)
        use_retrieval = data.get('use_retrieval', True)
        quick_mode = data.get('quick_mode', False)
        structure_only = data.get('structure_only', False)
        test_case_structure = data.get('test_case_structure', '')
        track_tokens = data.get('track_tokens', True)  # Default to tracking tokens
        
        # Force token tracking for debugging
        track_tokens = True
        
        # Log request details with token tracking flag
        print(f"Generating test cases for: {description[:50]}...")
        print(f"Parameters: use_knowledge={use_knowledge}, use_retrieval={use_retrieval}, quick_mode={quick_mode}, structure_only={structure_only}, track_tokens={track_tokens}")
        
        # Add a direct token tracking entry for the request
        try:
            token_counter.log_request(
                request_type="test_case_request",
                prompt_text=f"Description: {description}\nAcceptance Criteria: {acceptance_criteria}",
                completion_text=None,  # No completion yet
                metadata={
                    "structure_only": structure_only,
                    "quick_mode": quick_mode,
                    "use_knowledge": use_knowledge,
                    "use_retrieval": use_retrieval,
                    "track_tokens": track_tokens,
                    "source": "generate_test_cases_endpoint",
                    "phase": "request"
                }
            )
            print("Logged initial request to token counter")
        except Exception as e:
            print(f"Error logging initial request to token counter: {str(e)}")
            traceback.print_exc()
        
        # Since we can't use the LLM model, generate test cases based on the input
        # This is a more sophisticated version than the simple hardcoded test cases
        
        # Extract key information from the description and acceptance criteria
        keywords = extract_keywords(description + " " + acceptance_criteria)
        feature = keywords[0] if keywords else "feature"
        action = keywords[1] if len(keywords) > 1 else "functionality"
        
        # Generate test cases based on the input
        test_cases = generate_smart_test_cases(description, acceptance_criteria, feature, action)
        
        # Log token usage for successful completion
        if track_tokens:
            try:
                # Don't log the prompt again, just the completion
                result_text = json.dumps(test_cases)
                
                token_counter.log_request(
                    request_type="test_case_completion",
                    prompt_text="",  # Empty because we already logged the prompt
                    completion_text=result_text,
                    metadata={
                        "structure_only": structure_only,
                        "quick_mode": quick_mode,
                        "use_knowledge": use_knowledge,
                        "use_retrieval": use_retrieval,
                        "generation_time": time.time() - start_time,
                        "source": "generate_test_cases_completion",
                        "phase": "completion"
                    }
                )
                print("Successfully logged completion to token counter")
            except Exception as e:
                print(f"Error logging completion to token counter: {str(e)}")
                traceback.print_exc()
        
        # Format the test cases as a readable string
        formatted_test_cases = format_test_cases(test_cases)
        
        # Return the formatted test cases as a string
        return jsonify({
            'success': True,
            'test_cases': formatted_test_cases
        })
        
    except Exception as e:
        print(f"Error generating test cases: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def format_test_cases(test_cases):
    """Format test cases into a readable string"""
    formatted = ""
    
    for i, tc in enumerate(test_cases):
        formatted += f"Test Case {i+1}: {tc['title']}\n"
        formatted += f"Description: {tc['description']}\n\n"
        
        formatted += "Steps:\n"
        for j, step in enumerate(tc['steps']):
            formatted += f"{j+1}. {step}\n"
        formatted += "\n"
        
        formatted += "Expected Results:\n"
        for result in tc['expected_results']:
            formatted += f"- {result}\n"
        
        if i < len(test_cases) - 1:
            formatted += "\n---\n\n"
    
    return formatted

def extract_keywords(text):
    """Extract keywords from text"""
    # Remove common words and punctuation
    import re
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove common words
    common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                   'be', 'been', 'being', 'to', 'of', 'for', 'with', 'by', 'about', 
                   'against', 'between', 'into', 'through', 'during', 'before', 'after',
                   'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                   'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                   'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                   'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                   'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                   'just', 'don', 'should', 'now', 'that', 'this', 'these', 'those'}
    
    keywords = [word for word in words if word not in common_words and len(word) > 3]
    
    # Return unique keywords
    return list(set(keywords))

def generate_smart_test_cases(description, acceptance_criteria, feature, action):
    """Generate test cases based on the input"""
    # Parse acceptance criteria into separate items
    ac_items = parse_acceptance_criteria(acceptance_criteria)
    
    test_cases = []
    
    # Add a basic test case
    test_cases.append({
        "title": f"Verify {feature} Basic Functionality",
        "description": f"This test case verifies the basic functionality of the {feature} as described in the user story.",
        "steps": [
            f"Log in to the system as a user with appropriate permissions",
            f"Navigate to the {feature} section",
            f"Verify that the {action} functionality is available",
            f"Attempt to use the {feature} with valid inputs",
            f"Verify the results match the expected behavior"
        ],
        "expected_results": [
            f"The {feature} is accessible to authorized users",
            f"The {action} functionality works as described in the user story",
            f"The system displays appropriate feedback after the action is completed"
        ]
    })
    
    # Add test cases based on acceptance criteria
    for i, ac in enumerate(ac_items):
        if not ac.strip():
            continue
            
        # Extract keywords from this acceptance criterion
        ac_keywords = extract_keywords(ac)
        ac_feature = ac_keywords[0] if ac_keywords else feature
        ac_action = ac_keywords[1] if len(ac_keywords) > 1 else action
        
        test_cases.append({
            "title": f"Verify {ac_feature} {ac_action} Functionality",
            "description": f"This test case verifies the acceptance criterion: \"{ac}\"",
            "steps": [
                f"Log in to the system as a user with appropriate permissions",
                f"Navigate to the {ac_feature} section",
                f"Set up the test conditions for testing \"{ac}\"",
                f"Perform the actions necessary to test this specific acceptance criterion",
                f"Verify that the {ac_feature} behaves as expected"
            ],
            "expected_results": [
                f"The acceptance criterion \"{ac}\" is fully satisfied",
                f"The {ac_feature} behaves correctly under the test conditions",
                f"The system provides appropriate feedback or results"
            ]
        })
    
    # Add an edge case test
    test_cases.append({
        "title": f"Verify {feature} Error Handling",
        "description": f"This test case verifies that the {feature} handles error conditions appropriately.",
        "steps": [
            f"Log in to the system as a user with appropriate permissions",
            f"Navigate to the {feature} section",
            f"Attempt to use the {feature} with invalid inputs",
            f"Verify that appropriate error messages are displayed",
            f"Attempt to use the {feature} with boundary values",
            f"Verify that the system handles edge cases correctly"
        ],
        "expected_results": [
            f"The system displays clear error messages for invalid inputs",
            f"The {feature} handles boundary values correctly",
            f"The system prevents invalid operations from being completed"
        ]
    })
    
    return test_cases

def parse_acceptance_criteria(acceptance_criteria):
    """Parse acceptance criteria into separate items"""
    # Split by common separators
    items = []
    
    # Try to split by numbered items (1., 2., etc.)
    import re
    numbered = re.split(r'\d+\.', acceptance_criteria)
    if len(numbered) > 1:
        # Remove the first empty item if it exists
        if not numbered[0].strip():
            numbered = numbered[1:]
        items = [item.strip() for item in numbered]
    else:
        # Try to split by bullet points
        bullet_split = re.split(r'[\*\-•]', acceptance_criteria)
        if len(bullet_split) > 1:
            # Remove the first empty item if it exists
            if not bullet_split[0].strip():
                bullet_split = bullet_split[1:]
            items = [item.strip() for item in bullet_split]
        else:
            # Split by newlines as a last resort
            items = [line.strip() for line in acceptance_criteria.split('\n') if line.strip()]
    
    return items

@app.route('/add-test-tokens', methods=['GET'])
def add_test_tokens():
    """Endpoint to add test tokens for demonstration"""
    try:
        # Add a test entry with a significant number of tokens
        result = token_counter.log_request(
            request_type="manual_test",
            prompt_text="This is a test prompt " * 100,  # About 500 tokens
            completion_text="This is a test completion " * 50,  # About 250 tokens
            metadata={"manual": True, "source": "add-test-tokens endpoint"}
        )
        
        return jsonify({
            'success': True,
            'message': 'Test tokens added successfully',
            'tokens_added': result
        })
    except Exception as e:
        print(f"Error adding test tokens: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
