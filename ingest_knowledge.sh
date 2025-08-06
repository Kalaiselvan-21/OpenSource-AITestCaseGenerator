#!/bin/bash
# ingest_knowledge.sh

# Activate virtual environment
source venv/bin/activate

# Load environment variables
source .env

# Split the KNOWLEDGE_FILES by comma
IFS=',' read -ra FILES <<< "$KNOWLEDGE_FILES"

# Loop through each file and ingest knowledge
for file in "${FILES[@]}"; do
  echo "Ingesting knowledge from $file..."
  ./run.py knowledge ingest --source "$file" --name "$(basename "$file")"
done

echo "Knowledge ingestion complete!"
