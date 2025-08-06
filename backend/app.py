from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import traceback
import subprocess
import platform
import sys
from datetime import datetime
from token_counter import TokenCounter

app = Flask(__name__)
CORS(app)

# Initialize token counter
token_counter = TokenCounter()

@app.route('/health', methods=['GET'])
def health_check():
    """
    Enhanced health check endpoint that verifies all prerequisites for the AI test agent
    and returns comprehensive system status information.
    """
    status = "ok"
    message = "All systems operational"
    warnings = []
    errors = []
    
    # Get token usage statistics
    try:
        token_stats = token_counter.get_usage_stats()
    except Exception as e:
        token_stats = {"error": str(e)}
        warnings.append(f"Token counter error: {str(e)}")
    
    # Check LLM model availability
    model_status = {"available": False, "name": "unknown"}
    try:
        # Try to check if Ollama is installed and running
        ollama_check = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if ollama_check.returncode == 0:
            # Ollama is installed, check if it's running
            model_status["ollama_installed"] = True
            
            # Try to list models
            try:
                models_check = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if models_check.returncode == 0:
                    model_status["available"] = True
                    model_status["models"] = models_check.stdout.strip().split("\n")
                    model_status["name"] = "mistral" if "mistral" in models_check.stdout else "unknown"
                    model_status["status"] = "running"
                else:
                    model_status["status"] = "installed but not running"
                    model_status["error"] = models_check.stderr.strip()
            except subprocess.TimeoutExpired:
                model_status["status"] = "timeout checking models"
                warnings.append("Ollama command timed out - service may be slow or not running")
            except Exception as e:
                model_status["status"] = "error checking models"
                model_status["error"] = str(e)
        else:
            model_status["ollama_installed"] = False
            model_status["status"] = "not installed"
            warnings.append("Ollama not found in PATH")
            
        # Also try to import the test case generator to check model availability
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.generators.test_case_generator import TestCaseGenerator
            model_status["generator_module_found"] = True
        except ImportError:
            model_status["generator_module_found"] = False
            warnings.append("TestCaseGenerator module not found")
    except Exception as e:
        model_status["error"] = str(e)
        warnings.append(f"Error checking model: {str(e)}")
    
    # Check knowledge base
    kb_status = {"available": False}
    try:
        # Check if knowledge directory exists
        knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")
        kb_status["directory_exists"] = os.path.isdir(knowledge_dir)
        
        if kb_status["directory_exists"]:
            # Count files in knowledge directory
            knowledge_files = [f for f in os.listdir(knowledge_dir) if os.path.isfile(os.path.join(knowledge_dir, f))]
            kb_status["file_count"] = len(knowledge_files)
            kb_status["available"] = len(knowledge_files) > 0
            kb_status["status"] = "populated" if len(knowledge_files) > 0 else "empty"
            
            if not knowledge_files:
                warnings.append("Knowledge directory exists but contains no files")
        else:
            warnings.append("Knowledge directory not found")
            
        # Also try to import the knowledge base module
        try:
            from src.ingestion.knowledge_base import KnowledgeBase
            kb = KnowledgeBase()
            kb_status["module_found"] = True
            
            # Try to get knowledge items
            try:
                knowledge_items = kb.get_all_knowledge()
                kb_status["items_count"] = len(knowledge_items)
            except:
                kb_status["items_count"] = "unknown"
        except ImportError:
            kb_status["module_found"] = False
            warnings.append("KnowledgeBase module not found")
    except Exception as e:
        warnings.append(f"Knowledge base error: {str(e)}")
        kb_status["error"] = str(e)
    
    # Check for tokenizer
    tokenizer_status = {"available": False}
    try:
        # Simple check if we can count tokens
        sample_text = "This is a sample text to check tokenizer functionality."
        token_count = token_counter.count_tokens(sample_text)
        tokenizer_status = {
            "available": True,
            "sample_count": token_count,
            "implementation": "Approximate word-based tokenizer",
            "method": "word-based approximation (1.67 tokens per word)"
        }
    except Exception as e:
        warnings.append(f"Tokenizer error: {str(e)}")
        tokenizer_status["error"] = str(e)
    
    # Basic system info without resource utilization
    system_status = {}
    try:
        system_status = {
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }
    except Exception as e:
        system_status["error"] = str(e)
        warnings.append(f"System info error: {str(e)}")
    
    # Check environment variables
    env_vars_status = {}
    required_env_vars = ["OPENAI_API_KEY", "CONFLUENCE_API_TOKEN"]
    for var in required_env_vars:
        env_vars_status[var] = var in os.environ
        if not env_vars_status[var]:
            warnings.append(f"Missing environment variable: {var}")
    
    # Check Python dependencies
    dependencies_status = {"checked": []}
    required_packages = ["flask", "flask_cors", "langchain", "langchain_ollama"]
    for package in required_packages:
        try:
            __import__(package)
            dependencies_status["checked"].append({"name": package, "installed": True})
        except ImportError:
            dependencies_status["checked"].append({"name": package, "installed": False})
            warnings.append(f"Missing required package: {package}")
    
    # Update overall status based on warnings and errors
    if errors:
        status = "error"
        message = "Critical issues detected"
    elif warnings:
        status = "warning"
        message = "System operational with warnings"
    
    # Compile the complete health status
    health_status = {
        'status': status,
        'message': message,
        'warnings': warnings if warnings else None,
        'errors': errors if errors else None,
        'components': {
            'model': model_status,
            'knowledge_base': kb_status,
            'tokenizer': tokenizer_status,
            'system': system_status,
            'environment': env_vars_status,
            'dependencies': dependencies_status
        },
        'token_usage': token_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(health_status)
    
    return jsonify(health_status)

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    """
    Enhanced endpoint to generate test cases using LLM models with knowledge base integration,
    vector database retrieval, and best practices incorporation.
    """
    try:
        data = request.json
        # Log the full incoming request for debugging
        print("[DEBUG] Incoming request data:", data)
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract only required fields, making summary optional
        description = data.get('description', '')
        acceptance_criteria = data.get('acceptance_criteria', '')
        use_knowledge = data.get('use_knowledge', True)
        use_retrieval = data.get('use_retrieval', True)
        print(f"[DEBUG] Extracted description: {description}")
        print(f"[DEBUG] Extracted acceptance_criteria: {acceptance_criteria}")
        
        # Extract summary from description if not provided
        summary = data.get('summary', '')
        if not summary and description:
            # Try to extract a summary from the first line or sentence of the description
            summary_lines = description.split('\n')
            summary = summary_lines[0][:50] + ('...' if len(summary_lines[0]) > 50 else '')
        
        # Simplified validation - only check if both fields exist
        if not description or not acceptance_criteria:
            return jsonify({"error": f"Missing required fields: description and acceptance_criteria. Received: description={bool(description)}, acceptance_criteria={bool(acceptance_criteria)}"}), 400
        
        # Log token usage for the request
        prompt_text = f"Description: {description}\n\nAcceptance Criteria: {acceptance_criteria}"
        
        # Try to use the TestCaseGenerator from src if available
        try:
            # Add project root to path
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Import the test case generator
            from src.generators.test_case_generator import TestCaseGenerator
            from src.ingestion.knowledge_base import KnowledgeBase
            
            # Initialize knowledge base
            kb = KnowledgeBase()
            
            # Initialize test case generator
            generator = TestCaseGenerator(knowledge_base=kb)
            
            # Generate test cases using the LLM
            test_cases = generator.generate_test_cases(
                description, 
                acceptance_criteria,
                use_knowledge=use_knowledge
            )
            
            # Import and apply post-processing to remove unwanted fields
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.generators.post_processor import post_process_test_cases
            
            # Apply post-processing
            processed_test_cases = post_process_test_cases(test_cases)
            
            # Save output to file
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}.txt"
            output_path = os.path.join(output_dir, output_filename)
            
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Write processed test cases to file
            with open(output_path, "w") as f:
                f.write(processed_test_cases)
            
            # Log success
            print(f"Successfully generated test cases using TestCaseGenerator")
            print(f"Output saved to: {output_path}")
            
            # Log the token usage
            token_counter.log_request(
                request_type="test_case_generation",
                prompt_text=prompt_text,
                completion_text=test_cases,
                metadata={
                    "source": "TestCaseGenerator",
                    "use_knowledge": use_knowledge,
                    "use_retrieval": use_retrieval
                }
            )
            
            return jsonify({
                "success": True,
                "test_cases": processed_test_cases,
                "source": "llm_generator",
                "output_file": output_path
            })
            
        except ImportError as e:
            print(f"Could not import TestCaseGenerator: {str(e)}")
            print("Falling back to enhanced template-based generation")
        except Exception as e:
            print(f"Error using TestCaseGenerator: {str(e)}")
            print(f"Exception details: {traceback.format_exc()}")
            print("Falling back to enhanced template-based generation")
        
        # If we reach here, we need to use our enhanced template approach
        # This is a fallback if the TestCaseGenerator is not available
        
        # Extract key information from the description and acceptance criteria
        key_points = []
        try:
            # Improved extraction of key points from acceptance criteria
            # First, split by paragraphs (double newlines)
            paragraphs = acceptance_criteria.split('\n\n')
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                    
                # Process each paragraph
                lines = paragraph.split('\n')
                current_point = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # If line starts with a list marker, it's likely a new point
                    if line.startswith('-') or line.startswith('*') or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        # Save previous point if exists
                        if current_point:
                            key_points.append(current_point)
                        current_point = line
                    else:
                        # Continue accumulating the current point
                        if current_point:
                            current_point += " " + line
                        else:
                            current_point = line
                
                # Add the last point from this paragraph
                if current_point:
                    key_points.append(current_point)
                # If no points were extracted from this paragraph, add the whole paragraph
                elif paragraph.strip():
                    key_points.append(paragraph.strip())
                    
            # If we still have no points, use the entire acceptance criteria
            if not key_points:
                key_points.append(acceptance_criteria)
                        
            # Debug output
            print(f"Extracted {len(key_points)} key points from acceptance criteria")
            for i, point in enumerate(key_points):
                print(f"Point {i+1}: {point[:100]}...")
                
        except Exception as e:
            print(f"Error extracting key points: {str(e)}")
            print(traceback.format_exc())
            # Fallback: use the whole acceptance criteria
            key_points = [acceptance_criteria]
        
        # Generate more specific test cases based on the key points
        specific_test_cases = []
        
        # Generate test cases for each key point from acceptance criteria
        for i, point in enumerate(key_points):
            # Clean up the point text
            point_text = point.strip()
            if point_text.startswith('-') or point_text.startswith('*'):
                point_text = point_text[1:].strip()
            elif len(point_text) > 2 and point_text[0].isdigit() and point_text[1] == '.':
                point_text = point_text[2:].strip()
                
            # Create a title (limit to 80 chars for display)
            title = point_text[:80] + ('...' if len(point_text) > 80 else '')
            
            # Generate more specific test steps based on the point content
            steps = []
            
            # Check for specific keywords to generate more targeted steps
            if "button" in point_text.lower() or "cta" in point_text.lower():
                steps.append(f"Navigate to the page containing the button/CTA")
                steps.append(f"Verify the button/CTA is visible and correctly labeled")
                steps.append(f"Click the button/CTA")
                steps.append(f"Verify the expected action occurs")
            elif "field" in point_text.lower() or "input" in point_text.lower() or "enter" in point_text.lower():
                steps.append(f"Navigate to the page containing the input field")
                steps.append(f"Verify the field is visible and correctly labeled")
                steps.append(f"Enter test data into the field")
                steps.append(f"Verify the field accepts/validates the input correctly")
            elif "message" in point_text.lower() or "display" in point_text.lower() or "show" in point_text.lower():
                steps.append(f"Set up the conditions to trigger the message/display")
                steps.append(f"Perform the action that should trigger the message/display")
                steps.append(f"Verify the message/display appears correctly")
                steps.append(f"Verify the content of the message/display is correct")
            elif "redirect" in point_text.lower() or "navigate" in point_text.lower():
                steps.append(f"Set up the conditions for the redirection/navigation")
                steps.append(f"Perform the action that triggers the redirection/navigation")
                steps.append(f"Verify the user is redirected/navigated to the correct page")
            elif "header" in point_text.lower() or "text" in point_text.lower():
                steps.append(f"Navigate to the page containing the header/text")
                steps.append(f"Verify the header/text is displayed correctly")
                steps.append(f"Verify the content matches the specified text")
            elif "search" in point_text.lower() or "find" in point_text.lower():
                steps.append(f"Navigate to the search functionality")
                steps.append(f"Enter search criteria")
                steps.append(f"Initiate the search")
                steps.append(f"Verify search results are displayed correctly")
            elif "layout" in point_text.lower() or "placement" in point_text.lower():
                steps.append(f"Navigate to the specified page")
                steps.append(f"Verify the layout and element placements match the requirements")
                steps.append(f"Check responsiveness on different screen sizes if applicable")
            else:
                # Generic steps if no specific pattern is matched
                steps.append(f"Navigate to the relevant page/section")
                steps.append(f"Set up the test conditions for this requirement")
                steps.append(f"Perform the actions needed to test this requirement")
                steps.append(f"Verify the system behavior matches the expected outcome")
            
            # Create expected result based on the point content
            expected_result = f"The system correctly implements: {point_text}"
            
            specific_test_cases.append({
                "title": f"Verify {title}",
                "description": f"Ensure the system correctly implements: {point_text}",
                "steps": steps,
                "expected": expected_result
            })
        
        # Format the test cases in markdown
        test_cases_md = f"# Test Cases for Add Business Page\n\n"
        
        for i, tc in enumerate(specific_test_cases):
            tc_num = str(i + 1).zfill(3)
            test_cases_md += f"## TC-{tc_num}: {tc['title']}\n"
            test_cases_md += f"**Description**: {tc['description']}\n"
            test_cases_md += "**Preconditions**: System is properly configured and accessible\n"
            test_cases_md += "**Test Steps**:\n"
            
            for j, step in enumerate(tc['steps']):
                test_cases_md += f"{j+1}. {step}\n"
            
            test_cases_md += f"\n**Expected Results**: {tc['expected']}\n"
            test_cases_md += "**Test Data**: Appropriate test data for this scenario\n"
            test_cases_md += "**Priority**: High\n\n"
        
        # Only add standard test cases if we have fewer than 5 specific test cases
        # This ensures we don't dilute the specific test cases with generic ones
        if len(specific_test_cases) < 5:
            # Add standard test cases for edge cases and error handling
            test_cases_md += """## TC-EDGE: Test Edge Cases
**Description**: Verify behavior with edge cases
**Preconditions**: System is in a stable state
**Test Steps**:
1. Test with minimum allowed values (e.g., minimum length EIN)
2. Test with maximum allowed values (e.g., maximum length business name)
3. Test with special characters in input fields
4. Test with unexpected user behavior (e.g., rapid clicking)

**Expected Results**: System handles edge cases gracefully
**Test Data**: Min/max values, special characters
**Priority**: Medium

## TC-ERROR: Verify Error Handling
**Description**: Ensure proper error handling
**Preconditions**: System is accessible
**Test Steps**:
1. Enter invalid EIN format
2. Try to search with empty EIN field
3. Verify error messages are displayed
4. Verify recovery options are provided

**Expected Results**: User-friendly error messages are displayed with recovery options
**Test Data**: Invalid EIN formats, empty fields
**Priority**: High
"""
        
        # Save output to file
        from src.generators.post_processor import post_process_test_cases
        processed_test_cases_md = post_process_test_cases(test_cases_md)
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.txt"
        output_path = os.path.join(output_dir, output_filename)
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Write test cases to file
        with open(output_path, "w") as f:
            f.write(processed_test_cases_md)
        print(f"Output saved to: {output_path}")
        # Log the token usage
        try:
            token_counter.log_request(
                request_type="test_case_generation",
                prompt_text=prompt_text,
                completion_text=processed_test_cases_md,
                metadata={
                    "source": "enhanced_template",
                    "use_knowledge": use_knowledge,
                    "use_retrieval": use_retrieval
                }
            )
        except Exception as e:
            print(f"Error logging tokens: {str(e)}")
        return jsonify({
            "success": True,
            "test_cases": processed_test_cases_md,
            "source": "enhanced_template",
            "output_file": output_path
        })
        
    except Exception as e:
        print(f"Error generating test cases: {str(e)}")
        print(f"Exception details: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
