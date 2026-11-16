import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.models.database import SessionLocal, DocumentChunk, Document

load_dotenv()

class VectorSearchEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")

        #Initialize Gemini embedding model for query vectorization
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            output_dimensionality=768,
            google_api_key=api_key
        )

    def search(self, query_text:str, top_k: int = 3) -> List[Dict[str, Any]]:
        db: Session = SessionLocal()
        try:
            # 1. Generate embeddings for the user's query
            query_vector = self.embeddings_model.embed_query(query_text)
            # 2. Format query vector as a pgvector-compatible string
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        
            sql_query = text("""
                SELECT
                    dc.id AS chunk_id,
                    dc.chunk_index,
                    dc.text AS chunk_text,
                    d.filename,
                    (dc.embedding <=> :query_vector) AS distance
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                ORDER BY dc.embedding <=> :query_vector
                LIMIT :top_k;
            """)

            results = db.execute(
                sql_query,
                {"query_vector": vector_str, "top_k": top_k}
            ).fetchall()

            #3. Format results into clean dictionaries
            formatted_results = []
            for row in results:
                # Convert cosine distance to a similarity score (1.0 - distance)
                similarity_score = 1.0 - float(row.distance)
                formatted_results.append({
                    "filename": row.filename,
                    "chunk_index": row.chunk_index,
                    "text": row.chunk_text,
                    "similarity_score": round(similarity_score,4)
                })

            return formatted_results
        except Exception as e:
            print(f"Error during vector search: {e}")
            raise e
        finally:
            db.close()

if __name__ == "__main__":
    #Test vector search with a sample query
    engine = VectorSearchEngine()
    test_query = "What is LangGraph used for?"
    print(f"Searching for : '{test_query}'...\n")

    hits = engine.search(test_query, top_k = 2)
    for idx, hit in enumerate(hits,1):
        print(f"Match {idx} [Score: {hit['similarity_score']}] from {hit['filename']}:")
        print(f"\"{hit['text']}\"n")
