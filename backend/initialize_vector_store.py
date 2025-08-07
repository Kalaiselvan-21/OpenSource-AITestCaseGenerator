#!/usr/bin/env python3
"""
Script to initialize the FAISS vector store with knowledge base documents
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store import initialize_vector_store, get_vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Initialize the FAISS vector store
    """
    print("🚀 Initializing FAISS Vector Store for AI Test Case Generator")
    print("=" * 60)
    
    # Check if Ollama is running
    print("🔍 Checking Ollama status...")
    import subprocess
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is running")
            models = result.stdout.strip().split('\n')
            print(f"📋 Available models: {len(models)}")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model}")
        else:
            print("❌ Ollama is not responding")
            print("   Please start Ollama with: ollama serve")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Ollama command timed out")
        return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        print("   Please install Ollama from: https://ollama.ai/")
        return False
    
    # Check knowledge base directory
    knowledge_base_path = "./knowledge_base"
    print(f"\n🔍 Checking knowledge base at: {knowledge_base_path}")
    
    if not os.path.exists(knowledge_base_path):
        print(f"❌ Knowledge base directory not found: {knowledge_base_path}")
        return False
    
    # Count documents
    doc_count = 0
    for root, dirs, files in os.walk(knowledge_base_path):
        for file in files:
            if file.endswith('.txt'):
                doc_count += 1
    
    print(f"📚 Found {doc_count} text documents in knowledge base")
    
    if doc_count == 0:
        print("⚠️  No text documents found in knowledge base")
        print("   Please add .txt files to the knowledge_base directory")
        return False
    
    # Initialize vector store
    print(f"\n🔧 Initializing FAISS vector store...")
    print("   This may take a few minutes depending on the size of your knowledge base...")
    
    start_time = datetime.now()
    
    try:
        # Initialize with hybrid approach
        success = initialize_vector_store()
        
        if success:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ Vector store initialized successfully!")
            print(f"⏱️  Initialization took: {duration:.2f} seconds")
            
            # Get and display statistics
            vs = get_vector_store()
            stats = vs.get_stats()
            
            print(f"\n📊 Vector Store Statistics:")
            print(f"   - Vector store initialized: {stats.get('vector_store_initialized', 'Unknown')}")
            print(f"   - Index size: {stats.get('index_size', 'Unknown')}")
            
            if 'metadata' in stats:
                metadata = stats['metadata']
                print(f"   - Total chunks: {metadata.get('total_chunks', 'Unknown')}")
                print(f"   - Documents processed: {metadata.get('documents_processed', 'Unknown')}")
                print(f"   - Chunk size: {metadata.get('chunk_size', 'Unknown')}")
                print(f"   - Created at: {metadata.get('created_at', 'Unknown')}")
            
            print(f"\n🎉 FAISS vector store is ready for use!")
            print(f"   You can now start the enhanced Flask app with:")
            print(f"   python app_enhanced.py")
            
            return True
            
        else:
            print("❌ Vector store initialization failed")
            print("   Check the logs above for error details")
            return False
            
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        return False

def test_vector_store():
    """
    Test the vector store with a sample query
    """
    print("\n🧪 Testing vector store with sample query...")
    
    try:
        vs = get_vector_store()
        
        # Test query
        test_query = "user login test cases"
        results = vs.similarity_search(test_query, k=3)
        
        print(f"🔍 Query: '{test_query}'")
        print(f"📋 Found {len(results)} similar documents:")
        
        for i, doc in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   - Source: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   - Type: {doc.metadata.get('doc_type', 'Unknown')}")
            print(f"   - Content preview: {doc.page_content[:100]}...")
        
        print("\n✅ Vector store test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Starting vector store initialization at {datetime.now()}")
    
    # Initialize vector store
    if main():
        # Test the vector store
        test_vector_store()
        print("\n🎯 All done! Your AI Test Case Generator is ready with FAISS integration.")
    else:
        print("\n💥 Initialization failed. Please check the errors above and try again.")
        sys.exit(1)
