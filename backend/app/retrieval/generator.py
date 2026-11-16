import os 
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.retrieval.search import VectorSearchEngine

load_dotenv()

class RAGGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables")

        #Initialize our vector search engine
        self.search_engine = VectorSearchEngine()

        # Initialise Gemini chat model for synthesis
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-4-hydrogen",
            google_api_key=api_key,
            temperature=0.2
        )

    def answer_query(self, user_query: str, top_k: int = 3)-> Dict[str, Any]:
        retrieved_hits = self.search_engine.search(user_query, top_k=top_k)

        if not retrieved_hits:
            return {
                "answer": "I couldn't fine any relevant documentation to answer your question.",
                "sources": []
            }
        
        context_blocks = []
        sources= []
        for idx, hit in enumerate(retrieved_hits, 1):
            context_blocks.append(f"Source [{idx}] ({hit['filename']} - Chunk {hit['chunk_index']}):\n{hit['text']}")
            sources.append({
                "filename": hit["filename"],
                "chunk_index":hit["chunk_index"],
                "similarity_score": hit["similarity_score"]
            })
        context_string = "\n\n".join(context_blocks)
        print(context_string)

        system_prompt = (
        "You are an expert technical assistant for an internal RAG and agentic AI platform. "
        "Your job is to answer user queries accurately using ONLY the provided context chunks below. "
        "If the answer cannot be found in the context, explicitly state that you don't know "
        "rather than fabricating information. Cite your source chunk numbers when applicable."
        )

        human_prompt = f"""
        Context Information:
        {context_string}

        User Question: {user_query}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = self.chat_model.invoke(messages)
        return {
            "answer": response.content,
            "sources": sources
        }

if __name__ == "__main__":
    # Test end-to-end RAG generation
    generator = RAGGenerator()
    query = "What is LangGraph used for?"
    
    print(f"Query: '{query}'\n")
    result = generator.answer_query(query)
    
    print("Generated Answer:")
    print(result["answer"])
    print("\nRetrieved Sources:")
    for src in result["sources"]:
        print(f"- {src['filename']} (Chunk {src['chunk_index']}, Score: {src['similarity_score']})")