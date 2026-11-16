from app.models.database import SessionLocal, Document, DocumentChunk

db = SessionLocal()
docs = db.query(Document).all()
print(f"Total Master Documents: {len(docs)}")
for doc in docs:
    print(f"- ID: {doc.id}, File: {doc.filename}, Checksum: {doc.checksum[:10]}...")

chunks = db.query(DocumentChunk).all()
print(f"\nTotal Chunks Embedded: {len(chunks)}")
for chunk in chunks:
    print(f"  * Chunk {chunk.chunk_index} for Doc {chunk.document_id}: '{chunk.text[:50]}...' (Vector dims: {len(chunk.embedding)})")

db.close()