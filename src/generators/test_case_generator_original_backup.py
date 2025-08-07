import os
print(f"[DEBUG] Loaded test_case_generator.py from: {os.path.abspath(__file__)}")
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
    Generate test cases from user stories and acceptance criteria
    """
    
    def __init__(self, knowledge_base=None, retriever=None, llm=None):
        """
        Initialize the test case generator
        
        Args:
            knowledge_base: Knowledge base for domain knowledge
            retriever: Retriever for similar test cases
            llm: Language model for test case generation
        """
        self.knowledge_base = knowledge_base
        self.retriever = retriever
        
        # Initialize LLM if not provided
        if llm is None:
            try:
                # Only use Ollama LLM (Llama 2)
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(model="llama2")
                logger.info("Initialized Ollama LLM with llama2 model")
                print("[DEBUG] Llama 2 LLM is being used for test case generation via Ollama.")
            except ImportError:
                logger.error("Could not import langchain_ollama. LLM-based test case generation will not work.")
                self.llm = None
        else:
            self.llm = llm
            logger.info(f"Using provided LLM: {type(llm).__name__}")
        
        # If knowledge_base is not provided, try to import and initialize it
        if self.knowledge_base is None:
            try:
                # Import from project root
                from src.ingestion.knowledge_base import KnowledgeBase
                self.knowledge_base = KnowledgeBase()
                logger.info("Initialized knowledge base")
            except Exception as e:
                logger.warning(f"Failed to initialize knowledge base: {str(e)}")
                self.knowledge_base = None
        
        # Enhanced prompt template with domain knowledge for more detailed test cases with 100% coverage
        from langchain_core.prompts import PromptTemplate
        self.prompt = PromptTemplate(
            input_variables=["user_story", "acceptance_criteria", "domain_knowledge"],
            template="""
You are an expert test engineer. Given the following user story and acceptance criteria, generate detailed, comprehensive test cases that ensure 100% coverage.

- First, generate all positive test cases (valid scenarios) that directly satisfy each acceptance criterion.
- After listing all positive test cases, provide a separate section titled “Negative and Edge Cases” containing test cases for invalid input, error handling, boundary values, and unusual user actions.
- Test cases must be specific, not generic.

For each test case, include:
- A clear, descriptive title
- Step-by-step test steps (numbered)
- The expected result (clear and verifiable)

Ensure that:
- Each acceptance criteria is explicitly covered by at least one positive test case.
- Negative and edge cases are grouped together in their own section after the positive test cases.
- All test cases are specific to the requirements and context, not generic.

Example Test Case:
Title: Verify successful login with valid credentials
Preconditions: User is on the login page
Steps:
1. Enter a valid username in the username field
2. Enter a valid password in the password field
3. Click the "Login" button
4. Observe the navigation to the dashboard page
Expected Result: The user is redirected to the dashboard and sees a welcome message

Follow this level of detail for all test steps.

User Story:
{user_story}

Acceptance Criteria:
{acceptance_criteria}

Domain Knowledge:
{domain_knowledge}

IMPORTANT INSTRUCTIONS:
1. Ensure EVERY SINGLE POINT in the acceptance criteria has at least one corresponding test case.
2. Break down complex acceptance criteria into multiple test cases if needed.
3. Include both positive and negative test cases for each feature.
4. For UI elements, include tests for visibility, functionality, and content verification.
5. For data flows, include validation of inputs, processing, and outputs.
6. For error conditions, verify appropriate error messages and recovery paths.
7. Ensure test cases are specific to the actual requirements, not generic.

Return the test cases in this format:
- Title: [Full descriptive title without abbreviations or truncation]
- Test Steps:
  1. [Specific, actionable step with exact details]
  2. [Specific, actionable step with exact details]
  3. [Continue with specific steps]
- Expected Result: [Clear, verifiable outcome]
"""
        )
        
        # LangChain LLM chain
        # Updated to use the recommended approach instead of deprecated LLMChain
        from langchain_core.runnables import RunnablePassthrough
        
        # Create a runnable sequence
        self.chain = self.prompt | self.llm

    def generate_test_cases(self, description: str, acceptance_criteria: str, use_knowledge: bool = True) -> str:
        """
        Generate test cases using Llama 2 via Ollama.
        Args:
            description (str): The user story or feature description.
            acceptance_criteria (str): The acceptance criteria for the feature.
            use_knowledge (bool): Whether to use domain knowledge (default: True).
        Returns:
            str: The generated test cases as a string.
        """
        if not self.llm:
            raise RuntimeError("LLM is not initialized. Cannot generate test cases.")

        # Retrieve domain knowledge and test case examples from the knowledge base
        domain_knowledge = ""
        if use_knowledge and self.knowledge_base:
            kb_results = self.knowledge_base.search_knowledge(f"{description}\n{acceptance_criteria}", k=3)
            print("[DEBUG] Knowledge base search results:", kb_results)
            if kb_results:
                domain_knowledge = "\n\n".join([item['content'] for item in kb_results])
            else:
                domain_knowledge = ""
        else:
            domain_knowledge = ""

        prompt = f"""
You are an expert test engineer. Given the following user story and acceptance criteria, generate detailed, comprehensive test cases that ensure 100% coverage.

- Each test case title must start with 'Verify' or 'Validate'.
- After each test case, add one blank line before the next test case.
- If an error message or user prompt message is mentioned in the user story or acceptance criteria, include the exact message in the Expected Result for the relevant test case.
- Do not invent any details. Only use information provided in the user story and acceptance criteria. If a message or scenario is not mentioned, do not include it.
- Do NOT include any test cases related to login, authentication, or user access unless such scenarios are explicitly mentioned in the user story or acceptance criteria.

For each test case, include:
- A clear, descriptive title (starting with 'Verify' or 'Validate')
- Step-by-step test steps (numbered)
- The expected result (clear, verifiable, and using the exact error or prompt message from the input if applicable)

Ensure that:
- Each acceptance criterion is explicitly covered by at least one positive test case.
- Negative and edge cases are grouped together in their own section after the positive test cases.
- All test cases are specific to the requirements and context, not generic.

Example Test Case:
Title: Verify successful login with valid credentials
Preconditions: User is on the login page
Steps:
1. Enter a valid username in the username field
2. Enter a valid password in the password field
3. Click the "Login" button
4. Observe the navigation to the dashboard page
Expected Result: The user is redirected to the dashboard and sees a welcome message

Follow this level of detail for all test cases.

User Story:
{description}

Acceptance Criteria:
{acceptance_criteria}

Domain Knowledge:
{domain_knowledge}
"""
        print("[DEBUG] Prompt sent to LLM:\n", prompt)
        result = self.llm.invoke(prompt)
        if isinstance(result, dict) and 'content' in result:
            return result['content']
        return str(result)
