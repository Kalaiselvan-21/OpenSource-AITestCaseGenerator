# AI Test Case Generator Backend

This is the backend service for the AI Test Case Generator, which uses Ollama and Langchain to generate test cases based on user stories and acceptance criteria.

## Setup Instructions

1. **Install Ollama**:
   - Download from: https://ollama.ai/
   - Install it on your system

2. **Set up the Python environment**:
   ```bash
   # Create and activate a virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Pull the LLM model**:
   ```bash
   ollama pull llama2
   ```
   
   You can also use other models like `mistral` or `phi` by setting the `OLLAMA_MODEL` environment variable.

4. **Start the backend service**:
   ```bash
   ./run.sh
   ```
   
   This will start the Flask server on port 5002.

## Usage with Forge App

Since Forge apps can't directly connect to localhost URLs in production, you have two options:

1. **For development**:
   - Use a service like ngrok to expose your local server:
     ```bash
     ngrok http 5002
     ```
   - Update the Forge app's config.js with the ngrok URL

2. **For production**:
   - Deploy this backend to a server with a public HTTPS endpoint
   - Update the Forge app's config.js with the production URL

## Environment Variables

- `OLLAMA_MODEL`: The Ollama model to use (default: llama2)
- `PORT`: The port to run the server on (default: 5002)

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /generate-test-cases`: Generate test cases based on user story and acceptance criteria
