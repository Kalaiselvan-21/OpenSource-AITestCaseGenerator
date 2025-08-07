import re
import os
import json
import traceback
from datetime import datetime

class TokenCounter:
    """
    A utility class to count tokens and track usage statistics
    """
    
    def __init__(self, log_file_path=None):
        """
        Initialize the token counter
        
        Args:
            log_file_path: Path to the log file for token usage
        """
        if log_file_path is None:
            # Default log file in the same directory as this script
            self.log_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'token_usage_log.json'
            )
        else:
            self.log_file_path = log_file_path
            
        # Initialize or load the log file
        self._init_log_file()
        
    
    def _init_log_file(self):
        """Initialize or load the log file"""
        try:
            if not os.path.exists(self.log_file_path):
                # Create a new log file with initial structure
                initial_data = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "requests": [],
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                
                with open(self.log_file_path, 'w') as f:
                    json.dump(initial_data, f, indent=2)
                    
                print(f"Created new token usage log at {self.log_file_path}")
            else:
                # Verify the file is readable and has valid JSON
                try:
                    with open(self.log_file_path, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Token usage log at {self.log_file_path} contains invalid JSON. Creating new file.")
                    # Backup the corrupted file
                    backup_path = f"{self.log_file_path}.bak"
                    os.rename(self.log_file_path, backup_path)
                    
                    # Create a new log file
                    initial_data = {
                        "total_tokens": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "requests": [],
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat()
                    }
                    
                    with open(self.log_file_path, 'w') as f:
                        json.dump(initial_data, f, indent=2)
        except Exception as e:
            print(f"Error initializing token usage log: {str(e)}")
            traceback.print_exc()
    
    def count_tokens(self, text):
        """
        Count the number of tokens in the given text
        This is a simple approximation based on word count
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Approximate token count
        """
        # If text is None or empty, return 0
        if not text:
            return 0
            
        # Simple approximation: 1 token ≈ 0.75 words
        # This is a rough estimate; actual tokenization varies by model
        
        # Split by whitespace and punctuation
        words = re.findall(r'\w+', text)
        
        # Use a more accurate ratio: 1 token ≈ 0.6 words (or 1.67 tokens per word)
        # This is closer to GPT's actual tokenization for English text
        token_count = int(len(words) * 1.67)
        
        return token_count
    
    def log_request(self, request_type, prompt_text, completion_text=None, metadata=None):
        """
        Log a request with token counts
        
        Args:
            request_type: Type of request (e.g., 'test_case_generation')
            prompt_text: The prompt text
            completion_text: The completion text (if available)
            metadata: Additional metadata about the request
            
        Returns:
            dict: Token usage statistics for this request
        """
        try:
            prompt_tokens = self.count_tokens(prompt_text)
            completion_tokens = self.count_tokens(completion_text) if completion_text else 0
            total_tokens = prompt_tokens + completion_tokens
            
            
            # Create the request entry
            request_entry = {
                "timestamp": datetime.now().isoformat(),
                "request_type": request_type,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
            # Add metadata if provided
            if metadata:
                request_entry["metadata"] = metadata
                
            # Update the log file
            try:
                # Read the current log data
                with open(self.log_file_path, 'r') as f:
                    log_data = json.load(f)
                    
                # Update totals
                log_data["total_tokens"] += total_tokens
                log_data["prompt_tokens"] += prompt_tokens
                log_data["completion_tokens"] += completion_tokens
                log_data["requests"].append(request_entry)
                log_data["last_updated"] = datetime.now().isoformat()
                
                # Write the updated log data
                with open(self.log_file_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
                    
                
            except Exception as e:
                print(f"Error updating token log: {str(e)}")
                traceback.print_exc()
                
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        except Exception as e:
            print(f"Error in log_request: {str(e)}")
            traceback.print_exc()
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "error": str(e)
            }
    
    def get_usage_stats(self):
        """
        Get the current token usage statistics
        
        Returns:
            dict: Token usage statistics
        """
        try:
            with open(self.log_file_path, 'r') as f:
                log_data = json.load(f)
                
            # Calculate additional statistics
            num_requests = len(log_data["requests"])
            avg_tokens_per_request = log_data["total_tokens"] / num_requests if num_requests > 0 else 0
            
            # Calculate estimated costs based on OpenAI pricing
            # GPT-4 pricing: $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
            gpt4_cost = (log_data["prompt_tokens"] / 1000 * 0.03) + (log_data["completion_tokens"] / 1000 * 0.06)
            
            # GPT-3.5 Turbo pricing: $0.0015 per 1K prompt tokens, $0.002 per 1K completion tokens
            gpt35_cost = (log_data["prompt_tokens"] / 1000 * 0.0015) + (log_data["completion_tokens"] / 1000 * 0.002)
            
            stats = {
                "total_tokens": log_data["total_tokens"],
                "prompt_tokens": log_data["prompt_tokens"],
                "completion_tokens": log_data["completion_tokens"],
                "num_requests": num_requests,
                "avg_tokens_per_request": avg_tokens_per_request,
                "estimated_costs": {
                    "gpt4": gpt4_cost,
                    "gpt35_turbo": gpt35_cost
                },
                "created_at": log_data["created_at"],
                "last_updated": log_data["last_updated"]
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting usage stats: {str(e)}")
            traceback.print_exc()
            return {
                "error": str(e),
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "num_requests": 0,
                "estimated_costs": {
                    "gpt4": 0,
                    "gpt35_turbo": 0
                }
            }

# Example usage
if __name__ == "__main__":
    counter = TokenCounter()
    
    # Example logging
    counter.log_request(
        "test_case_generation",
        "Generate test cases for user login functionality",
        "Here are the test cases: 1. Verify user can login with valid credentials...",
        {"user_story_id": "US-123"}
    )
    
    # Print stats
    print(json.dumps(counter.get_usage_stats(), indent=2))
