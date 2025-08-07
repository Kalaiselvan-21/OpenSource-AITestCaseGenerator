# FAISS Vector Store Integration

This document explains the FAISS vector store integration for the AI Test Case Generator.

## 🎯 What This Integration Adds

### **Enhanced Context Retrieval**
- **Semantic Search**: Instead of basic keyword matching, FAISS finds contextually similar content
- **Better Test Case Quality**: Generated test cases are more relevant and domain-specific
- **Intelligent Context**: The system understands relationships between concepts

### **New Capabilities**
- **Vector-based similarity search** for finding related test cases and documentation
- **Context-aware test generation** using relevant knowledge base content
- **Semantic understanding** of test requirements and domain knowledge

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌──────────────────┐               │
│  │   Flask App     │    │  FAISS Vector    │               │
│  │  (Enhanced)     │◄──►│     Store        │               │
│  └─────────────────┘    └──────────────────┘               │
│           │                       │                        │
│           ▼                       ▼                        │
│  ┌─────────────────┐    ┌──────────────────┐               │
│  │ Enhanced Test   │    │  Knowledge Base  │               │
│  │   Generator     │◄──►│   Documents      │               │
│  └─────────────────┘    └──────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ Ollama LLM      │                                       │
│  │ (Llama2/Mistral)│                                       │
│  └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 New Files Added

### **Core Components**
- `vector_store.py` - FAISS vector store implementation
- `enhanced_test_generator.py` - Enhanced test generator with FAISS integration
- `app_enhanced.py` - Enhanced Flask app with new endpoints

### **Utilities**
- `initialize_vector_store.py` - Script to set up the vector store
- `FAISS_INTEGRATION.md` - This documentation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Initialize Vector Store
```bash
cd backend
python initialize_vector_store.py
```

### 3. Start Enhanced Server
```bash
python app_enhanced.py
```

## 🔧 Detailed Setup

### **Step 1: Verify Prerequisites**
```bash
# Check Ollama is running
ollama list

# Should show llama2 and/or mistral models
```

### **Step 2: Initialize FAISS Vector Store**
```bash
cd backend
python initialize_vector_store.py
```

This script will:
- ✅ Check Ollama connectivity
- ✅ Scan knowledge base documents
- ✅ Create FAISS embeddings
- ✅ Save vector store to disk
- ✅ Run test queries

### **Step 3: Start Enhanced Application**
```bash
python app_enhanced.py
```

The enhanced app includes all original functionality plus:
- 🔍 Vector-based context retrieval
- 📊 Vector store statistics
- 🔄 Dynamic vector store reinitialization

## 🌐 API Endpoints

### **Enhanced Endpoints**

#### **POST /generate-test-cases** (Enhanced)
Now uses FAISS for context retrieval:

```json
{
  "description": "User login functionality",
  "acceptance_criteria": "User should be able to login with email and password",
  "use_knowledge": true
}
```

**Response includes:**
```json
{
  "success": true,
  "test_cases": "Generated test cases...",
  "metadata": {
    "vector_store_used": true,
    "vector_store_stats": {...},
    "generation_time_seconds": 2.34
  }
}
```

#### **POST /vector-search** (New)
Search for similar content:

```json
{
  "query": "user authentication test cases",
  "k": 5
}
```

#### **GET /vector-stats** (New)
Get vector store statistics:

```json
{
  "success": true,
  "stats": {
    "vector_store_initialized": true,
    "index_size": 1250,
    "metadata": {
      "total_chunks": 45,
      "documents_processed": 8
    }
  }
}
```

#### **POST /reinitialize-vector-store** (New)
Reinitialize vector store (useful after adding documents):

```json
{
  "force_recreate": true
}
```

### **Health Check Enhanced**
The `/health` endpoint now includes:
- ✅ FAISS vector store status
- ✅ Required package versions
- ✅ Vector store statistics

## 🔍 How It Works

### **1. Document Processing**
```python
# Documents are loaded from knowledge_base/
documents = load_documents()

# Split into chunks for better context
chunks = text_splitter.split_documents(documents)

# Create embeddings using Ollama
embeddings = OllamaEmbeddings(model="llama2")

# Store in FAISS vector database
vector_store = FAISS.from_documents(chunks, embeddings)
```

### **2. Context Retrieval**
```python
# When generating test cases
query = f"{test_type} test cases for {requirements}"

# Find semantically similar content
context = vector_store.get_relevant_context(query, max_tokens=1500)

# Use context in LLM prompt
enhanced_prompt = create_prompt_with_context(requirements, context)
```

### **3. Test Case Generation**
```python
# Enhanced generator uses both:
# 1. Semantic context from FAISS
# 2. Similar test case examples
# 3. Domain-specific knowledge

test_cases = enhanced_generator.generate_test_cases(
    description=user_story,
    acceptance_criteria=criteria,
    use_knowledge=True  # Enables FAISS integration
)
```

## 📊 Performance Benefits

### **Before FAISS Integration**
- ❌ Basic keyword matching
- ❌ Limited context awareness
- ❌ Generic test cases

### **After FAISS Integration**
- ✅ Semantic similarity search
- ✅ Context-aware generation
- ✅ Domain-specific test cases
- ✅ Better coverage of edge cases

## 🛠️ Configuration

### **Vector Store Settings**
```python
# In vector_store.py
TestCaseVectorStore(
    knowledge_base_path="./knowledge_base",
    embeddings_model="llama2",  # or "mistral"
    vector_store_path="./vector_store",
    chunk_size=1000,
    chunk_overlap=200
)
```

### **Customization Options**
- **Chunk Size**: Adjust for your document types
- **Overlap**: Control context continuity
- **Model**: Switch between llama2/mistral for embeddings
- **Search Results**: Configure number of similar documents

## 🔧 Troubleshooting

### **Common Issues**

#### **1. Vector Store Initialization Fails**
```bash
# Check Ollama is running
ollama serve

# Verify models are available
ollama list

# Reinitialize with force
python initialize_vector_store.py
```

#### **2. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check FAISS installation
python -c "import faiss; print(faiss.__version__)"
```

#### **3. Memory Issues**
```python
# Reduce chunk size in vector_store.py
chunk_size=500  # Instead of 1000
```

#### **4. Slow Performance**
```python
# Reduce context size
max_tokens=1000  # Instead of 1500

# Limit search results
k=3  # Instead of 5
```

## 📈 Monitoring

### **Vector Store Health**
```bash
curl http://localhost:5000/health
curl http://localhost:5000/vector-stats
```

### **Performance Metrics**
- Generation time with/without FAISS
- Context relevance scores
- Token usage statistics

## 🔄 Maintenance

### **Adding New Documents**
1. Add `.txt` files to `knowledge_base/`
2. Reinitialize vector store:
   ```bash
   curl -X POST http://localhost:5000/reinitialize-vector-store \
        -H "Content-Type: application/json" \
        -d '{"force_recreate": true}'
   ```

### **Updating Models**
1. Pull new Ollama models: `ollama pull mistral`
2. Update configuration in `vector_store.py`
3. Reinitialize vector store

## 🎯 Next Steps

### **Potential Enhancements**
- 🔄 **Automatic reindexing** when documents change
- 📊 **Analytics dashboard** for vector store performance
- 🔍 **Advanced filtering** by document type/metadata
- 🚀 **GPU acceleration** with faiss-gpu
- 🌐 **Multi-language support** with different embedding models

### **Integration Options**
- 📱 **Frontend integration** for real-time search
- 🔗 **API versioning** for backward compatibility
- 📈 **Metrics collection** for usage analytics
- 🔐 **Authentication** for enterprise deployment

---

**🎉 Your AI Test Case Generator now has semantic search superpowers!**
