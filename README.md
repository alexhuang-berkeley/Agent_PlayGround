# Single-Agent LLM Playground

This project provides a minimal playground for experimenting with a single-agent LLM workflow. It is implemented with **Streamlit** for quick iteration.

## Features
- Upload documents (`.pdf`, `.txt`, `.docx`) indexed into a FAISS vector store using OpenAI embeddings
- Upload a single CSV file that is automatically loaded into an in-memory SQLite database
- The agent generates SQL queries against this database and shows both the query and results
- Use the **Schema Prompt** field to provide table schema or instructions before each text-to-SQL call. The schema prompt is injected into the SQL generation step.
- Live editable system and schema prompts influence the agent immediately
- Simple chat interface with optional conversation memory
- Basic model selector (uses Amazon Bedrock models)
- Powered by **LangChain**, exposing tools for SQL querying and vector-based document search
- View raw conversation history as a lightweight "chain of thought"

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Running
Start the Streamlit app:
```bash
streamlit run app.py
```
Configure AWS credentials (via environment variables or shared config) and set your `AWS_REGION`. The app uses Amazon Bedrock to generate responses.

## Notes
This is a simplified prototype meant for local experimentation. It does not persist uploads or chat history between sessions.
The agent logic now runs through LangChain, using tools for SQL queries and vector-based document retrieval. Uploaded documents are embedded into a FAISS vector store in memory for semantic search.
