# AI Test Case Generator

An intelligent test case generation system powered by open-source LLM models (Ollama) and FAISS vector database for semantic search and context-aware test case creation.

## 🚀 Features

- **AI-Powered Test Generation**: Leverages Ollama LLM models (Llama2, Mistral) for intelligent test case creation
- **Vector-Based Context Search**: Uses FAISS for semantic similarity search in knowledge base
- **Knowledge Base Integration**: Processes documentation and examples to generate contextually relevant test cases
- **RESTful API**: Flask-based backend with CORS support
- **Token Usage Tracking**: Built-in token counter for monitoring LLM usage
- **Extensible Architecture**: Modular design for easy integration and customization

## 🏗️ Architecture

```
AITestCaseGenerator/
├── backend/                    # Flask backend application
│   ├── app.py                 # Main Flask application
│   ├── vector_store.py        # FAISS vector store implementation
│   ├── token_counter.py       # Token usage tracking
│   ├── knowledge_base/        # Documentation and examples
│   │   ├── overall_functionality.txt
│   │   ├── test_case_examples/
│   │   └── best_practices/
│   ├── vector_store/          # FAISS index files (generated)
│   └── requirements.txt       # Python dependencies
├── AITestAgent/               # Frontend application
└── README.md
```

## 🛠️ Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Git (for version control)

### Ollama Models Required
- `llama2:latest`
- `mistral:latest`

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AITestCaseGenerator
   ```

2. **Set up Python virtual environment**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and start Ollama**
   ```bash
   # Install Ollama (macOS)
   brew install ollama
   
   # Start Ollama service
   ollama serve
   
   # Pull required models
   ollama pull llama2
   ollama pull mistral
   ```

5. **Initialize the vector store**
   ```bash
   python -c "from vector_store import initialize_vector_store; initialize_vector_store()"
   ```

## 🚀 Usage

### Starting the Backend Server

```bash
cd backend
source venv/bin/activate
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### Health Check
```bash
GET /health
```
Returns system status including Ollama connectivity and token usage statistics.

#### Generate Test Cases
```bash
POST /generate-test-cases
Content-Type: application/json

{
  "requirements": "User login functionality with email validation",
  "test_type": "functional",
  "complexity": "medium"
}
```

### Vector Store Operations

The FAISS vector store automatically processes documents from the `knowledge_base/` directory:

- **Functionality specifications**: `overall_functionality.txt`
- **Test case examples**: `test_case_examples/`
- **Best practices**: `best_practices/`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
OLLAMA_BASE_URL=http://localhost:11434
VECTOR_STORE_PATH=./vector_store
KNOWLEDGE_BASE_PATH=./knowledge_base
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Ollama Configuration

Ensure Ollama is running and accessible:

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Test model availability
ollama list
```

## 📊 Monitoring

### Token Usage Tracking

The system tracks token usage for cost monitoring:

```bash
# View token usage statistics
curl http://localhost:5000/health
```

### Vector Store Statistics

```python
from vector_store import get_vector_store

vs = get_vector_store()
stats = vs.get_stats()
print(stats)
```

## 🧪 Development

### Adding New Knowledge Base Content

1. Add documents to appropriate directories in `knowledge_base/`
2. Reinitialize the vector store:
   ```python
   from vector_store import initialize_vector_store
   initialize_vector_store(force_recreate=True)
   ```

### Testing Vector Search

```python
from vector_store import get_vector_store

vs = get_vector_store()
results = vs.similarity_search("user authentication test cases", k=5)
for doc in results:
    print(f"Score: {doc.metadata}")
    print(f"Content: {doc.page_content[:200]}...")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   ps aux | grep ollama
   
   # Restart Ollama
   ollama serve
   ```

2. **Vector Store Initialization Failed**
   ```bash
   # Check knowledge base files exist
   ls -la backend/knowledge_base/
   
   # Recreate vector store
   python -c "from vector_store import initialize_vector_store; initialize_vector_store(force_recreate=True)"
   ```

3. **FAISS Import Error**
   ```bash
   # Reinstall FAISS
   pip uninstall faiss-cpu
   pip install faiss-cpu==1.11.0
   ```

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review Ollama documentation for LLM-related issues

## 🔄 Version History

- **v1.0.0** - Initial release with basic test case generation
- **v1.1.0** - Added FAISS vector store integration
- **v1.2.0** - Enhanced context-aware generation

---

**Built with ❤️ using Ollama, FAISS, and Flask**
