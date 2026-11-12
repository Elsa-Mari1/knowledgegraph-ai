# KnowledgeGraph AI: Enterprise RAG & Agentic Knowledge Assistant

An enterprise knowledge assistant featuring stateful LangGraph agent workflows, hybrid vector retrieval (`pgvector`), structured metadata filtering, and full observability.

## Tech Stack
- **Backend:** Python, FastAPI, SQLAlchemy, LangChain, LangGraph
- **Database:** PostgreSQL + pgvector
- **AI Models:** Google Gemini API & Amazon Bedrock adapters
- **Observability:** LangSmith

## Quickstart
1. Copy `.env.example` to `.env` and fill in your API keys.
2. Spin up local database via `docker-compose up -d`.
3. Activate Python venv and run the FastAPI development server.
