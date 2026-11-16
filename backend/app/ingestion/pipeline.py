import os
import hashlib
from typing import List
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.models.database import SessionLocal, Document, DocumentChunk

load_dotenv()

class IngestionPipeline:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")

        #Initialize Google GenAI Embeddings (768 dimensions for text-embedding-004)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            output_dimensionality=768,
            google_api_key=api_key
        )
        
        #Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def compute_checksum(self, text:str) -> str:
        """Generate SHA-256 checksum to prevent duplicate document ingestion."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def ingest_text(self, filename: str, content: str, content_type: str = "text/plain") -> int:
        """ process raw text, chunk it, embed it via Gemini, and store in pgvector DB."""
        db: Session = SessionLocal()
        try:
            checksum = self.compute_checksum(content)

            # Check if document already exists
            existing_doc = db.query(Document).filter(Document.checksum == checksum).first()
            if existing_doc:
                print(f"Document '{filename}' already ingested (checksum match). Skipping.")
                return existing_doc.id

            db_doc = Document(
                filename=filename,
                content_type=content_type,
                checksum=checksum
            )
            db.add(db_doc)
            db.commit()
            db.refresh(db_doc)

            #Split text into chunks
            raw_chunks = self.text_splitter.split_text(content)
            print(f"Split '{filename}' into {len(raw_chunks)} chunks. Generating Gemini embedding...")

            # Generate embedding in batch using Gemini
            embeddings = self.embeddings_model.embed_documents(raw_chunks)

            #Store chunks in vectors
            for idx, (chunk_text, embedding) in enumerate(zip(raw_chunks, embeddings)):
                db_chunk = DocumentChunk(
                    document_id=db_doc.id,
                    chunk_index=idx,
                    text=chunk_text,
                    embedding=embedding
                )
                db.add(db_chunk)

            db.commit()
            print(f"Successfully ingested and embedded {len(raw_chunks)} chunks for '{filename}'.")
            return db_doc.id
        except Exception as e:
            db.rollback()
            print(f"Error during ingestion pipeline: {e}")
            raise e
        finally:
            db.close()
if __name__ == "__main__":
    from pathlib import Path
    # Read the updated architecture document
    file_path = Path("data/architecture_overview.txt")
    content = file_path.read_text(encoding="utf-8")

    # Initialise the ingestion pipeline
    pipeline = IngestionPipeline()

    # Ingest the document
    document_id = pipeline.ingest_text(
        filename=file_path.name,
        content=content
    )

    print(f"Successfully ingested {file_path.name}!")
    print(f"Document ID: {document_id}")