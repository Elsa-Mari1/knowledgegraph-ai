import os
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="KnowledgeGraph AI API", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status":"healthy", "service": "knowledgegraph-ai-backend"}

@app.get("/")
async def root():
    return {"message": " Welcome to KnowledgeGraph AI API. Go to /docs for Swagger UI."}

if __name__== "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)