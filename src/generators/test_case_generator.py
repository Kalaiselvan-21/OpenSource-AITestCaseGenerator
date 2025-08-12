import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Constants for prompt size management
MAX_PROMPT_TOKENS = 2048  # Example, adjust to model's actual max
SAFE_PROMPT_TOKENS = int(MAX_PROMPT_TOKENS * 0.75)

class TestCaseGenerator:
    """
    Enhanced test case generator with FAISS vector store integration
    Generate test cases from user stories and acceptance criteria using AI-first approach
    """
    
    def __init__(self, retriever=None, llm=None, ai_only=True, model_name=None):
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
        self.model_name = model_name
        
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
            
            from backend.vector_store import get_vector_store
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
                    self.llm = OllamaLLM(model="llama2", base_url="http://localhost:11434", temperature=0.1)
                    
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
            logger.info("[SYSTEM] 🚀 AI-ONLY MODE Ready")
            
        except Exception as e:
            # AI initialization failed - store error for AI-only mode
            self.ai_mode = "failed"
            self.initialization_error = str(e)
            self.vector_store = None
            self.llm = None
            
            logger.error(f"❌ AI components initialization failed: {str(e)}")
            logger.error(f"[SYSTEM] ❌ AI INITIALIZATION FAILED: {str(e)}")
            
            if self.ai_only:
                logger.error("🚫 AI-only mode requested - will not fallback to inferior methods")
                logger.error("[SYSTEM] 🚫 AI-ONLY MODE: Refusing to use fallback methods")

    # Removed _initialize_fallback_components method - AI-ONLY mode

    def _initialize_prompt_template(self):
        """
        Initialize the AI-only prompt template with HuggingFace context support
        """
        from langchain_core.prompts import PromptTemplate
        if self.ai_mode == "ai":
            # Updated prompt to enforce requested output format and ordering for readability
            self.prompt = PromptTemplate(
                input_variables=[
                    "user_story",
                    "acceptance_criteria",
                    "domain_knowledge",
                    "similar_examples",
                    "criteria_list",
                    "criteria_count",
                    "previous_criteria",
                ],
                template="""
You are a senior test engineer. Based on the user story and acceptance criteria, produce clear and readable test cases with the EXACT fields and order below:

Fields per test case (no extra fields):
- Acceptance Criteria
- Test case Title
- Steps
- Expected Result

Important formatting and ordering rules:
- Output must be in Markdown.
- First, provide ALL Positive test cases for ALL acceptance criteria.
- Then, after finishing positives, provide ALL Negative test cases for ALL acceptance criteria.
- Group by Acceptance Criterion inside each section and ensure complete coverage.
- Use concise, numbered Steps for clarity.
- Keep Expected Result singular and specific.

Acceptance Criteria to cover ({criteria_count} items):

{criteria_list}

Context from previous criteria (for cross-criterion awareness):
{previous_criteria}

Additional domain knowledge and example references (use for context only, do not copy verbatim):
{domain_knowledge}

{similar_examples}

Now generate the test cases in this structure:

## Positive Test Cases

### AC <number>
- Acceptance Criteria: <paste the AC text>
- Test case Title: <concise, outcome-focused title>
- Steps:
    1. <step>
    2. <step>
- Expected Result: <singular expected outcome>

### AC <number>
- Acceptance Criteria: <paste the AC text>
- Test case Title: <concise, outcome-focused title>
- Steps:
    1. <step>
    2. <step>
- Expected Result: <singular expected outcome>

...repeat until all ACs have at least one Positive test case…

## Negative Test Cases

### AC <number>
- Acceptance Criteria: <paste the AC text>
- Test case Title: <concise, failure-mode-focused title>
- Steps:
    1. <step>
    2. <step>
- Expected Result: <singular expected outcome>

### AC <number>
- Acceptance Criteria: <paste the AC text>
- Test case Title: <concise, failure-mode-focused title>
- Steps:
    1. <step>
    2. <step>
- Expected Result: <singular expected outcome>

 Do not include any other fields. Keep wording grounded in the given acceptance criteria.
"""
            )
            # Create runnable chain for AI mode
            if self.llm:
                self.chain = self.prompt | self.llm
                logger.info("✅ Prompt template initialized with requested format and ordering")
            else:
                logger.error("❌ Cannot create chain: LLM not available")
        else:
            # AI mode failed - no prompt template in AI-only mode
            logger.error("❌ AI-only mode: Cannot initialize prompt template without AI components")
            self.prompt = None
            self.chain = None

    def _summarize_text(self, text: str) -> str:
        """
        Simple rule-based summarization: condense long text to key sentences.
        For production, replace with a proper summarizer.
        """
        # Take first 5 sentences as summary
        sentences = re.split(r'(?<=[.!?]) +', text)
        summary = ' '.join(sentences[:5])
        return summary if summary else text[:500]

    def _chunk_criteria(self, criteria: List[str], chunk_size: int = 5) -> List[List[str]]:
        """
        Split criteria into chunks for batch processing.
        """
        return [criteria[i:i+chunk_size] for i in range(0, len(criteria), chunk_size)]

    def _prompt_length(self, prompt: str) -> int:
        """
        Estimate prompt length in tokens (simple word count for now).
        """
        return len(prompt.split())

    def enumerate_criteria(self, ac_text, manual_override: list = None, use_nlp: bool = True):
        """
        Parse acceptance criteria into atomic requirements.
        If manual_override is provided, use it directly.
        If use_nlp is True, use spaCy for sentence segmentation and conjunction detection.
        Otherwise, fallback to regex-based splitting.
        """
        if manual_override:
            return manual_override
        if use_nlp:
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(ac_text)
                items = []
                for sent in doc.sents:
                    # Only split on conjunctions if sentence is compound and both sides are complete clauses
                    if any(tok.dep_ == "cc" for tok in sent):
                        # Try to split only if both sides have a verb
                        clauses = [clause.text.strip() for clause in sent._.clauses] if hasattr(sent, "_.clauses") else [sent.text.strip()]
                        items.extend([cl for cl in clauses if cl])
                    else:
                        items.append(sent.text.strip())
                # Remove empty items
                return [item for item in items if item]
            except Exception as e:
                logger.warning(f"spaCy NLP parsing failed, falling back to regex: {str(e)}")
        # Fallback: regex-based splitting
        import re
        items = []
        numbered = re.split(r'\d+\.', ac_text)
        if len(numbered) > 1:
            if not numbered[0].strip():
                numbered = numbered[1:]
            items = [item.strip() for item in numbered if item.strip()]
        else:
            bullet_split = re.split(r'[\*\-•]', ac_text)
            if len(bullet_split) > 1:
                if not bullet_split[0].strip():
                    bullet_split = bullet_split[1:]
                items = [item.strip() for item in bullet_split if item.strip()]
            else:
                lines = [line.strip() for line in ac_text.split('\n') if line.strip()]
                for line in lines:
                    atomic = re.split(r'\band\b|\bor\b|;|\.', line)
                    items.extend([a.strip() for a in atomic if a.strip()])
        return items

    def extract_test_cases(self, output):
        """
        Extract test cases from output using flexible regex.
        Returns a list of test case blocks.
        """
        import re
        # Flexible pattern: looks for 'Test Case', 'Title:', and 'Steps:' in proximity
        pattern = re.compile(r'(Test Case \d+.*?Title:.*?Steps:.*?Expected Results:.*?)(?=Test Case \d+|$)', re.DOTALL | re.IGNORECASE)
        return pattern.findall(output)

    def _enforce_output_structure(self, raw_text: str, ac_items: List[str]) -> str:
        """
        Enforce the requested output structure:
        - Positive test cases for all ACs first, then Negative test cases
        - Fields: Acceptance Criteria, Test case Title, Steps, Expected Result
        Returns formatted Markdown if parsing succeeds; otherwise returns raw_text.
        """
        import re

        def parse(text):
            cases = {"positive": [], "negative": []}
            section = None
            current = None
            lines = text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                low = line.lower()
                if re.match(r"^##\s*positive\s*test\s*cases\s*$", low, flags=re.IGNORECASE):
                    section = "positive"
                    current = None
                    i += 1
                    continue
                if re.match(r"^##\s*negative\s*test\s*cases\s*$", low, flags=re.IGNORECASE):
                    section = "negative"
                    current = None
                    i += 1
                    continue
                m = re.match(r"^###\s*AC\s*(\d+).*", line, flags=re.IGNORECASE)
                if m and section in ("positive", "negative"):
                    # push previous
                    if current:
                        cases[section].append(current)
                    ac_num = int(m.group(1))
                    current = {"ac": ac_num, "ac_text": None, "title": None, "steps": [], "expected": None}
                    i += 1
                    continue
                if current is not None:
                    # Field bullets
                    if low.startswith("- acceptance criteria:"):
                        current["ac_text"] = line.split(":", 1)[1].strip()
                        i += 1
                        continue
                    if low.startswith("- test case title:") or low.startswith("- testcase title:"):
                        current["title"] = line.split(":", 1)[1].strip()
                        i += 1
                        continue
                    if low.startswith("- steps:"):
                        # collect subsequent numbered or dashed lines
                        i += 1
                        while i < len(lines):
                            step_line = lines[i]
                            step_stripped = step_line.strip()
                            if re.match(r"^(###|##)\s*", step_stripped) or step_stripped.lower().startswith("- expected result:") or step_stripped.lower().startswith("- acceptance criteria:") or step_stripped.lower().startswith("- test case title:"):
                                break
                            if re.match(r"^\d+\.|^- ", step_stripped):
                                # remove leading bullet/number
                                step_val = re.sub(r"^(\d+\.|-\s+)", "", step_stripped).strip()
                                if step_val:
                                    current["steps"].append(step_val)
                            i += 1
                        continue
                    if low.startswith("- expected result:"):
                        current["expected"] = line.split(":", 1)[1].strip()
                        i += 1
                        continue
                i += 1
            # push last
            if current and section in ("positive", "negative"):
                cases[section].append(current)
            # sort by ac
            for k in ("positive", "negative"):
                cases[k].sort(key=lambda x: x.get("ac", 0))
            return cases

        def format_md(cases):
            def render_section(name):
                buf = [f"## {name} Test Cases"]
                entries = cases["positive" if name == "Positive" else "negative"]
                for entry in entries:
                    ac_num = entry.get("ac")
                    ac_text = entry.get("ac_text") or (ac_items[ac_num-1] if isinstance(ac_num, int) and 0 < ac_num <= len(ac_items) else None)
                    title = entry.get("title") or (f"{name} - AC {ac_num}")
                    steps = entry.get("steps") or []
                    expected = entry.get("expected") or ""
                    buf.append(f"\n### AC {ac_num}")
                    buf.append(f"- Acceptance Criteria: {ac_text or ''}")
                    buf.append(f"- Test case Title: {title}")
                    buf.append(f"- Steps:")
                    if steps:
                        for idx, s in enumerate(steps, 1):
                            buf.append(f"  {idx}. {s}")
                    else:
                        buf.append("  1. ")
                    buf.append(f"- Expected Result: {expected}")
                return "\n".join(buf)

            # build final markdown
            pos = render_section("Positive")
            neg = render_section("Negative")
            return f"{pos}\n\n{neg}"

        try:
            cases = parse(raw_text)
            # If nothing parsed, return raw
            if not cases["positive"] and not cases["negative"]:
                return raw_text
            return format_md(cases)
        except Exception:
            return raw_text

    def generate_test_cases(self, description: str, acceptance_criteria: str, use_knowledge: bool = True) -> str:
        """
        Generate test cases using AI-ONLY approach with HuggingFace embeddings + Ollama LLM
        
        Args:
            description (str): The user story or feature description
            acceptance_criteria (str): The acceptance criteria for the feature
            use_knowledge (bool): Whether to use domain knowledge (default: True)
            
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

        if use_knowledge and hasattr(self, 'vector_store') and self.vector_store:
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

                logger.info("[AI] 🤗 Using semantic search")

            except Exception as e:
                logger.error(f"❌ FAISS vector store failed: {str(e)}")
                # In AI-only mode, we don't fallback - we fail
                raise RuntimeError(f"AI-only mode: FAISS vector store failed: {str(e)}")

        ac_items = self.enumerate_criteria(acceptance_criteria)
        criteria_list = '\n'.join([f"{i+1}. {item}" for i, item in enumerate(ac_items)])

        # Enrich domain_knowledge and similar_examples from vector store if available
        domain_knowledge = ""
        similar_examples = ""
        if use_knowledge and hasattr(self, 'vector_store') and self.vector_store:
            try:
                query = f"{description}\n{acceptance_criteria}"
                domain_context = self.vector_store.get_relevant_context(query=query, max_tokens=1000)
                similar_docs = self.vector_store.similarity_search(query=f"test cases examples for {description}", k=3)
                domain_knowledge = domain_context if domain_context != "No relevant context found." else ""
                if similar_docs:
                    similar_examples = "\n---\n".join([
                        f"Example from {doc.metadata.get('filename', 'knowledge base')}:\n{doc.page_content[:500]}..."
                        for doc in similar_docs
                    ])
            except Exception as e:
                logger.error(f"❌ FAISS vector store failed: {str(e)}")
                raise RuntimeError(f"AI-only mode: FAISS vector store failed: {str(e)}")

        all_outputs = []
        chunked_criteria = self._chunk_criteria(ac_items, chunk_size=5)
        start_idx = 0
        previous_criteria = []
        for chunk in chunked_criteria:
            chunk_list = '\n'.join([f"{i+1+start_idx}. {item}" for i, item in enumerate(chunk)])
            # Context bridging: pass previous criteria as context
            context_bridge = '\n'.join([f"{i+1}. {item}" for i, item in enumerate(previous_criteria)]) if previous_criteria else "None"
            prompt_input = {
                "user_story": description,
                "acceptance_criteria": '\n'.join(chunk),
                "domain_knowledge": domain_knowledge,
                "similar_examples": similar_examples,
                "criteria_list": chunk_list,
                "criteria_count": len(chunk),
                "previous_criteria": context_bridge
            }
            prompt_str = str(prompt_input)
            if self._prompt_length(prompt_str) > SAFE_PROMPT_TOKENS:
                logger.info("Prompt too long, but not summarizing. Using full input.")
            # Retry logic for LLM invocation
            for attempt in range(3):
                try:
                    result = self.chain.invoke(prompt_input)
                    if isinstance(result, dict) and 'content' in result:
                        output = result['content']
                    else:
                        output = str(result)
                    all_outputs.append(output)
                    break
                except Exception as e:
                    logger.error(f"LLM invocation failed (attempt {attempt+1}): {str(e)}")
                    if attempt == 2:
                        raise RuntimeError(f"AI-only mode: Test case generation failed after retries: {str(e)}")
            start_idx += len(chunk)
            previous_criteria.extend(chunk)
    # Aggregate all outputs
    combined = '\n\n'.join(all_outputs)
    # Enforce structure and readability before returning
    return self._enforce_output_structure(combined, ac_items)

    def generate_test_cases_with_metadata(self, description: str, acceptance_criteria: str, use_knowledge: bool = True) -> Dict[str, Any]:
        """
        Generate test cases and return with metadata about the AI-only generation process.

        Args:
            description (str): User story description
            acceptance_criteria (str): Acceptance criteria
            use_knowledge (bool): Whether to use knowledge base

        Returns:
            dict: Dictionary with test cases and metadata
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
        Search for similar test cases in the vector store (AI mode only).

        Args:
            query (str): Search query
            k (int): Number of results to return

        Returns:
            list: List of similar test cases with metadata
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
        Get statistics about the test case generation system.

        Returns:
            dict: Dictionary with system statistics
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
