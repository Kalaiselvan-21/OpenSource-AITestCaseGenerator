#!/usr/bin/env python3
"""
Simple script to add test tokens to the token usage log file.
This can be used to verify that the token tracking system is working correctly.
"""

import os
import sys
import json
from datetime import datetime

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, 'token_usage_log.json')

def add_test_tokens(prompt_tokens=500, completion_tokens=300):
    """Add test tokens to the log file"""
    print(f"Adding test tokens to {log_file_path}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    
    # Create the request entry
    request_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_type": "manual_test",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "metadata": {
            "manual": True,
            "source": "add_test_tokens.py"
        }
    }
    
    try:
        # Read the current log data
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
        
        # Update totals
        log_data["total_tokens"] += prompt_tokens + completion_tokens
        log_data["prompt_tokens"] += prompt_tokens
        log_data["completion_tokens"] += completion_tokens
        log_data["requests"].append(request_entry)
        log_data["last_updated"] = datetime.now().isoformat()
        
        # Write the updated log data
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Successfully added {prompt_tokens + completion_tokens} tokens to the log")
        print(f"New totals: {log_data['total_tokens']} total tokens")
        print(f"           {log_data['prompt_tokens']} prompt tokens")
        print(f"           {log_data['completion_tokens']} completion tokens")
        
        return True
    except Exception as e:
        print(f"Error adding test tokens: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def print_usage_stats():
    """Print the current token usage statistics"""
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
        
        # Calculate additional statistics
        num_requests = len(log_data["requests"])
        avg_tokens_per_request = log_data["total_tokens"] / num_requests if num_requests > 0 else 0
        
        # Calculate estimated costs based on OpenAI pricing
        # GPT-4 pricing: $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
        gpt4_cost = (log_data["prompt_tokens"] / 1000 * 0.03) + (log_data["completion_tokens"] / 1000 * 0.06)
        
        # GPT-3.5 Turbo pricing: $0.0015 per 1K prompt tokens, $0.002 per 1K completion tokens
        gpt35_cost = (log_data["prompt_tokens"] / 1000 * 0.0015) + (log_data["completion_tokens"] / 1000 * 0.002)
        
        print("\nToken Usage Statistics:")
        print(f"Total tokens: {log_data['total_tokens']}")
        print(f"Prompt tokens: {log_data['prompt_tokens']}")
        print(f"Completion tokens: {log_data['completion_tokens']}")
        print(f"Number of requests: {num_requests}")
        print(f"Average tokens per request: {avg_tokens_per_request:.2f}")
        print(f"Estimated GPT-4 cost: ${gpt4_cost:.2f}")
        print(f"Estimated GPT-3.5 Turbo cost: ${gpt35_cost:.2f}")
        print(f"Created at: {log_data['created_at']}")
        print(f"Last updated: {log_data['last_updated']}")
        
        return True
    except Exception as e:
        print(f"Error getting usage stats: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Add test tokens to the token usage log file')
    parser.add_argument('--prompt', type=int, default=500, help='Number of prompt tokens to add')
    parser.add_argument('--completion', type=int, default=300, help='Number of completion tokens to add')
    parser.add_argument('--stats', action='store_true', help='Print token usage statistics')
    
    args = parser.parse_args()
    
    if args.stats:
        print_usage_stats()
    else:
        add_test_tokens(args.prompt, args.completion)
        print_usage_stats()
