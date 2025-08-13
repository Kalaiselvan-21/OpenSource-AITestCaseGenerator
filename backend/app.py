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

# Initialize default FAISS vector store on startup (project-agnostic)
print("[STARTUP] 🚀 AI Test Case Generator - AI-ONLY Mode")
try:
    from vector_store import initialize_vector_store
    vector_store_initialized = initialize_vector_store(project_name=None)
    if vector_store_initialized:
        print("[STARTUP] ✅ FAISS vector store ready")
    else:
        print("[STARTUP] ❌ FAISS vector store failed")
except Exception as e:
    print(f"[STARTUP] ❌ Vector store error: {str(e)}")
    print("[STARTUP] 🔄 System will use intelligent fallback when needed")
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
    
    # Check FAISS vector store status
    vector_store_status = {"available": False, "stats": {}}
    try:
        from vector_store import get_vector_store
        vs = get_vector_store()
        vector_store_status = {
            "available": True,
            "stats": vs.get_stats(),
            "status": "operational"
        }
        
        # Test vector store with a simple query
        test_results = vs.similarity_search("test", k=1)
        vector_store_status["test_query_successful"] = True
        
    except Exception as e:
        vector_store_status = {
            "available": False,
            "error": str(e),
            "status": "failed"
        }
        warnings.append(f"FAISS vector store error: {str(e)}")

    # Check enhanced test generator status
    enhanced_generator_status = {"available": False}
    try:
        # Add project root to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.generators.test_case_generator import TestCaseGenerator
        
        test_generator = TestCaseGenerator()
        enhanced_generator_status = {
            "available": True,
            "ai_mode": getattr(test_generator, 'ai_mode', 'unknown'),
            "fallback_reason": getattr(test_generator, 'fallback_reason', None),
            "status": "operational"
        }
        
        if test_generator.ai_mode == "fallback":
            warnings.append(f"Enhanced generator in fallback mode: {test_generator.fallback_reason}")
            
    except Exception as e:
        enhanced_generator_status = {
            "available": False,
            "error": str(e),
            "status": "failed"
        }
        errors.append(f"Enhanced test generator error: {str(e)}")

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
            'vector_store': vector_store_status,
            'enhanced_generator': enhanced_generator_status,
            'tokenizer': tokenizer_status,
            'system': system_status,
            'environment': env_vars_status,
            'dependencies': dependencies_status
        },
        'token_usage': token_stats,
        'timestamp': datetime.now().isoformat(),
        'ai_capabilities': {
            'faiss_enabled': vector_store_status.get('available', False),
            'ollama_enabled': model_status.get('available', False),
            'enhanced_mode': enhanced_generator_status.get('ai_mode', 'unknown')
        }
    }
    
    return jsonify(health_status)

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    """
    Enhanced endpoint to generate test cases using LLM models with knowledge base integration,
    and support optional project_name for per-project vector store isolation.
    """
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "No data provided"}), 400

        description = data.get('description') or ''
        acceptance_criteria = data.get('acceptance_criteria') or ''
        use_knowledge = bool(data.get('use_knowledge', True))
        use_retrieval = bool(data.get('use_retrieval', True))
        project_name = (data.get('project_name') or '').strip() or None

        # Optional summary derivation
        summary = data.get('summary') or ''
        if not summary and description:
            first_line = description.split('\n', 1)[0]
            summary = first_line[:50] + ('...' if len(first_line) > 50 else '')

        if not description or not acceptance_criteria:
            return jsonify({
                "error": (
                    "Missing required fields: description and acceptance_criteria. "
                    f"Received: description={bool(description)}, acceptance_criteria={bool(acceptance_criteria)}"
                )
            }), 400

        # Validate project_name format if provided
        if project_name:
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
                return jsonify({"error": "Invalid project_name. Use only alphanumeric characters, underscores, and hyphens."}), 400

        prompt_text = f"Description: {description}\n\nAcceptance Criteria: {acceptance_criteria}"

        # Ensure project root is on sys.path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.generators.test_case_generator import TestCaseGenerator
        from src.generators.post_processor import post_process_test_cases

        # Initialize per-project vector store (best-effort)
        try:
            from vector_store import initialize_vector_store
            initialize_vector_store(project_name=project_name)
        except Exception:
            pass

        # Create generator bound to project_name
        generator = TestCaseGenerator(project_name=project_name)

        # Generate
        if hasattr(generator, 'generate_test_cases_with_metadata'):
            result = generator.generate_test_cases_with_metadata(
                description,
                acceptance_criteria,
                use_knowledge=use_knowledge
            )
            if result.get('success'):
                test_cases = result['test_cases']
                generation_metadata = result['metadata']
                print(f"[SYSTEM] ✅ Generated using {generation_metadata.get('model_used', 'AI')} in {generator.ai_mode.upper()} mode")
            else:
                raise Exception(f"Enhanced generation failed: {result.get('error', 'Unknown error')}")
        else:
            test_cases = generator.generate_test_cases(
                description,
                acceptance_criteria,
                use_knowledge=use_knowledge
            )
            generation_metadata = {
                "generated_at": datetime.now().isoformat(),
                "model_used": "llama2",
                "ai_mode": getattr(generator, 'ai_mode', 'unknown'),
            }

        # Post-process
        processed_test_cases = post_process_test_cases(test_cases)

        # Save output
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"output_{timestamp}.txt")
        with open(output_path, 'w') as f:
            f.write(processed_test_cases)

        ai_mode = getattr(generator, 'ai_mode', 'unknown')
        print(f"[SYSTEM] ✅ Test cases generated using {ai_mode.upper()} mode")
        print(f"[SYSTEM] 📁 Output saved to: {output_path}")

        # Log usage
        token_counter.log_request(
            request_type="test_case_generation",
            prompt_text=prompt_text,
            completion_text=test_cases,
            metadata={
                "source": "EnhancedTestCaseGenerator",
                "ai_mode": ai_mode,
                "use_knowledge": use_knowledge,
                "use_retrieval": use_retrieval,
                "project_name": project_name,
                "faiss_enabled": hasattr(generator, 'vector_store') and generator.vector_store is not None,
                **generation_metadata,
            },
        )

        return jsonify({
            "success": True,
            "test_cases": processed_test_cases,
            "source": "enhanced_llm_generator",
            "ai_mode": ai_mode,
            "metadata": {
                **generation_metadata,
                "project_name": project_name,
                "faiss_enabled": hasattr(generator, 'vector_store') and generator.vector_store is not None,
                "fallback_reason": getattr(generator, 'fallback_reason', None),
            },
            "output_file": output_path,
        })

    except ImportError as e:
        print(f"❌ Could not import Enhanced TestCaseGenerator: {str(e)}")
        print("🔄 This indicates a critical system configuration issue")
        return jsonify({
            "success": False,
            "error": "Enhanced test case generator not available",
            "details": str(e),
            "suggestion": "Please ensure all dependencies are installed and project structure is correct",
        }), 500
    except Exception as e:
        print(f"❌ Error generating test cases: {str(e)}")
        print(f"📋 Exception details: {traceback.format_exc()}")
        if "ollama" in str(e).lower() or "faiss" in str(e).lower():
            return jsonify({
                "success": False,
                "error": "AI components unavailable",
                "details": str(e),
                "suggestion": "Please ensure Ollama is running and FAISS vector store is initialized",
            }), 503
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "message": "Test case generation failed. Please check system status via /health endpoint.",
        }), 500

if __name__ == '__main__':
    print("🚀 AI Test Case Generator - AI-ONLY Mode")
    print("🤗 HuggingFace Embeddings + 🦙 Ollama LLM + ⚡ FAISS")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5002)
        
