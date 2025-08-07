import os
import re
import logging
import platform
import importlib
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

class TestCaseGenerator:
    """
    Enhanced test case generator with FAISS vector store integration
    Generate test cases from user stories and acceptance criteria using AI-first approach
    """
    
    def __init__(self, retriever=None, llm=None, ai_only=True):
        """
        Initialize the enhanced test case generator with AI-ONLY approach
        
        Args:
            retriever: Retriever for similar test cases (deprecated)
            llm: Language model for test case generation
            ai_only: If True (default), only use AI components, fail if not available
        """
        self.retriever = retriever
        self.ai_mode = "unknown"
        self.ai_only = ai_only
        self.initialization_error = None
        
        # AI-ONLY APPROACH: Initialize full AI stack or fail
        self._initialize_ai_components(llm)
        
        # Only initialize fallback if explicitly requested (ai_only=False)
        if not self.ai_only and self.ai_mode != "ai":
            self._initialize_fallback_components()
        elif self.ai_mode != "ai":
            # AI-only mode but AI failed - raise error
            error_msg = f"AI-only mode requested but AI components failed: {self.initialization_error}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Initialize prompt template
        self._initialize_prompt_template()
        
        logger.info(f"🚀 Enhanced Test Case Generator initialized in {self.ai_mode.upper()} mode (AI-only: {self.ai_only})")

    def _initialize_ai_components(self, provided_llm=None):
        """
        Initialize AI components (Ollama LLM + FAISS Vector Store)
        Sets ai_mode to 'ai' if successful, stores error if failed
        """
        try:
            # Step 1: Initialize and verify FAISS vector store
            logger.info("🔍 Initializing FAISS vector store...")
            
            # Import vector store from backend directory
            import sys
            backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend')
            if backend_path not in sys.path:
                sys.path.append(backend_path)
            
            from vector_store import get_vector_store
            self.vector_store = get_vector_store()
            
            # Try to load existing vector store
            if not self.vector_store.load_vector_store():
                logger.warning("No existing vector store found, will use empty vector store")
            
            # Check if vector store is properly initialized (even if empty)
            if not hasattr(self.vector_store, 'embeddings') or self.vector_store.embeddings is None:
                raise RuntimeError("FAISS vector store embeddings not properly initialized")
            
            # Test vector store with a simple query
            test_results = self.vector_store.similarity_search("test", k=1)
            logger.info("✅ FAISS vector store verified and operational")
            
            # Step 2: Initialize and verify Ollama LLM
            logger.info("🤖 Initializing Ollama LLM...")
            if provided_llm is None:
                from langchain_ollama import OllamaLLM
                import ollama
                
                # Verify Ollama server is running
                try:
                    models_response = ollama.list()
                    # Handle new Ollama API format
                    if hasattr(models_response, 'models'):
                        available_models = [model.model for model in models_response.models]
                    else:
                        # Fallback for older API format
                        available_models = [model.get('name', model.get('model', '')) for model in models_response.get('models', [])]
                    
                    if not any('llama2' in model for model in available_models):
                        raise RuntimeError("llama2 model not found. Please run: ollama pull llama2")
                    
                    self.llm = OllamaLLM(model="llama2", base_url="http://localhost:11434")
                    
                    # Test LLM with a simple query
                    test_response = self.llm.invoke("Hello")
                    if not test_response or len(test_response.strip()) == 0:
                        raise RuntimeError("LLM test query returned empty response")
                        
                    logger.info("✅ Ollama LLM verified and operational")
                    
                except Exception as ollama_error:
                    raise RuntimeError(f"Ollama verification failed: {str(ollama_error)}")
                    
            else:
                self.llm = provided_llm
                logger.info(f"✅ Using provided LLM: {type(provided_llm).__name__}")
            
            # If we reach here, all AI components are operational
            self.ai_mode = "ai"
            self.initialization_error = None
            print("[SYSTEM] 🚀 AI-ONLY MODE Ready")
            
        except Exception as e:
            # AI initialization failed - store error for AI-only mode
            self.ai_mode = "failed"
            self.initialization_error = str(e)
            self.vector_store = None
            self.llm = None
            
            logger.error(f"❌ AI components initialization failed: {str(e)}")
            print(f"[SYSTEM] ❌ AI INITIALIZATION FAILED: {str(e)}")
            
            if self.ai_only:
                logger.error("🚫 AI-only mode requested - will not fallback to inferior methods")
                print("[SYSTEM] 🚫 AI-ONLY MODE: Refusing to use fallback methods")

    # Removed _initialize_fallback_components method - AI-ONLY mode

    def _initialize_prompt_template(self):
        """
        Initialize the AI-only prompt template with HuggingFace context support
        """
        from langchain_core.prompts import PromptTemplate
        
        if self.ai_mode == "ai":
            # AI-only prompt with HuggingFace embeddings context
            self.prompt = PromptTemplate(
                input_variables=["user_story", "acceptance_criteria", "domain_knowledge", "similar_examples"],
                template="""
You are an expert test engineer with deep domain knowledge. Given the following user story, acceptance criteria, and relevant context from HuggingFace semantic search, generate detailed, comprehensive test cases that ensure 100% coverage.

CONTEXT FROM KNOWLEDGE BASE (via HuggingFace embeddings):
{domain_knowledge}

SIMILAR TEST CASE EXAMPLES (via semantic search):
{similar_examples}

USER STORY:
{user_story}

ACCEPTANCE CRITERIA:
{acceptance_criteria}

INSTRUCTIONS:
- For EACH acceptance criterion listed above, generate at least one specific test case that directly addresses it.
- The test cases MUST be generated in the SAME ORDER as the acceptance criteria. The first test case should correspond to the first acceptance criterion, the second to the second, and so on.
- Number the test cases to match the criteria.
- Each test case title must start with 'Verify' or 'Validate'.
- Use the domain knowledge and examples above to ensure context-appropriate test cases.
- Include both positive and negative test scenarios.
- For each test case, provide:
  * Clear, descriptive title
  * Numbered test steps
  * Expected results
  * Any relevant preconditions

IMPORTANT:
- Ensure EVERY acceptance criterion has corresponding test cases (no omissions).
- Use specific details from the domain knowledge when relevant.
- Follow patterns from similar examples but adapt to current requirements.
- Include edge cases and error scenarios.
- Be specific, not generic.

Generate the test cases now:
"""
            )
            
            # Create runnable chain for AI mode
            if self.llm:
                self.chain = self.prompt | self.llm
                logger.info("✅ AI-only prompt template initialized with HuggingFace context support")
            else:
                logger.error("❌ Cannot create chain: LLM not available")
                
        else:
            # AI mode failed - no prompt template in AI-only mode
            logger.error("❌ AI-only mode: Cannot initialize prompt template without AI components")
            self.prompt = None
            self.chain = None

    def generate_test_cases(self, description: str, acceptance_criteria: str, use_knowledge: bool = True) -> str:
        """
        Generate test cases using AI-ONLY approach with HuggingFace embeddings + Ollama LLM
        
        Args:
            description (str): The user story or feature description (mandatory)
            acceptance_criteria (str): The acceptance criteria for the feature (mandatory)
            use_knowledge (bool, optional): Whether to use domain knowledge (default: True)
        
        Returns:
            str: The generated test cases as a string
        
        Raises:
            RuntimeError: If AI components are not available and ai_only=True
        """
        # Ensure AI-only mode
        if self.ai_mode != "ai":
            error_msg = f"AI components not available. AI-only mode requires HuggingFace embeddings + Ollama LLM. Error: {self.initialization_error}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        if not self.llm:
            raise RuntimeError("Ollama LLM is not initialized. Cannot generate test cases.")

        # Get enhanced context using FAISS vector store with HuggingFace embeddings
        domain_knowledge = ""
        similar_examples = ""
        
        if use_knowledge and self.vector_store:
            try:
                # Use FAISS vector store for semantic search with HuggingFace embeddings
                query = f"{description}\n{acceptance_criteria}"
                
                # Get relevant domain knowledge
                domain_context = self.vector_store.get_relevant_context(
                    query=query,
                    max_tokens=1000
                )
                
                # Get similar test case examples
                similar_docs = self.vector_store.similarity_search(
                    query=f"test cases examples for {description}",
                    k=3
                )
                
                domain_knowledge = domain_context if domain_context != "No relevant context found." else ""
                
                if similar_docs:
                    similar_examples = "\n---\n".join([
                        f"Example from {doc.metadata.get('filename', 'knowledge base')}:\n{doc.page_content[:500]}..."
                        for doc in similar_docs
                    ])
                
                print(f"[AI] 🤗 Using semantic search")
                
            except Exception as e:
                logger.error(f"❌ FAISS vector store failed: {str(e)}")
                # In AI-only mode, we don't fallback - we fail
                raise RuntimeError(f"AI-only mode: FAISS vector store failed: {str(e)}")

        # Create the prompt input for AI mode
        prompt_input = {
            "user_story": description,
            "acceptance_criteria": acceptance_criteria,
            "domain_knowledge": domain_knowledge,
            "similar_examples": similar_examples
        }
        
        print(f"[AI] 🚀 Generating test cases")
        
        # Generate test cases using AI-only approach
        try:
            result = self.chain.invoke(prompt_input)
            if isinstance(result, dict) and 'content' in result:
                return result['content']
            return str(result)
            
        except Exception as e:
            logger.error(f"❌ AI test case generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"AI-only mode: Test case generation failed: {str(e)}")

    def generate_test_cases_with_metadata(self, description: str, acceptance_criteria: str, use_knowledge: bool = True) -> Dict[str, Any]:
        """
        Generate test cases and return with metadata about the AI-only generation process
        
        Args:
            description: User story description
            acceptance_criteria: Acceptance criteria
            use_knowledge: Whether to use knowledge base
            
        Returns:
            Dictionary with test cases and metadata
        """
        start_time = datetime.now()
        
        try:
            test_cases = self.generate_test_cases(description, acceptance_criteria, use_knowledge)
            
            # Get vector store statistics if available
            vector_stats = {}
            if hasattr(self, 'vector_store') and self.vector_store:
                vector_stats = self.vector_store.get_stats()
            
            return {
                "success": True,
                "test_cases": test_cases,
                "metadata": {
                    "generated_at": start_time.isoformat(),
                    "generation_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "model_used": "llama2",
                    "ai_mode": self.ai_mode,
                    "ai_only": self.ai_only,
                    "embeddings_type": "HuggingFace",
                    "embeddings_model": "all-MiniLM-L6-v2",
                    "vector_store_used": True,
                    "faiss_enabled": True,
                    "vector_store_stats": vector_stats,
                    "knowledge_used": use_knowledge,
                    "system_type": "hybrid_ai_only"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "generated_at": start_time.isoformat(),
                    "generation_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "ai_mode": self.ai_mode,
                    "ai_only": self.ai_only,
                    "error_occurred": True,
                    "initialization_error": self.initialization_error,
                    "system_type": "hybrid_ai_only"
                }
            }

    def search_similar_test_cases(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar test cases in the vector store (AI mode only)
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar test cases with metadata
        """
        if self.ai_mode != "ai" or not self.vector_store:
            logger.warning("⚠️ Vector search not available in fallback mode")
            return []
            
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            similar_cases = []
            for doc, score in results:
                similar_cases.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "source": doc.metadata.get('filename', 'unknown')
                })
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"Error searching similar test cases: {str(e)}")
            return []

    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the test case generation system
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "llm_model": "llama2" if self.llm else "not_initialized",
            "ai_mode": self.ai_mode,
            "ai_only": self.ai_only,
            "embeddings_type": "HuggingFace",
            "embeddings_model": "all-MiniLM-L6-v2",
            "system_status": "operational" if self.llm else "degraded"
        }
        
        if hasattr(self, 'vector_store') and self.vector_store:
            stats["vector_store_stats"] = self.vector_store.get_stats()
        
        return stats
