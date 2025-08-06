import re
import logging

# Set up logging
logger = logging.getLogger(__name__)

def post_process_test_cases(test_cases: str) -> str:
    print("[DEBUG] post_process_test_cases CALLED")  # Debug marker
    """
    Post-process test cases to remove unwanted fields (Test Data, Priority, Description)
    
    Args:
        test_cases (str): Generated test cases
        
    Returns:
        str: Cleaned test cases
    """
    # Remove any line containing Description, Test Data, or Priority (case-insensitive, anywhere in the line)
    keywords = ["description", "test data", "priority"]
    lines = test_cases.split('\n')
    cleaned_lines = []
    for line in lines:
        if any(kw in line.lower() for kw in keywords):
            continue
        cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines)
    # Remove extra blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'\n +\n', '\n\n', result)
    return result.strip()
