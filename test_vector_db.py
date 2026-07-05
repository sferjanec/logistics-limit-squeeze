#!/usr/bin/env python3
"""
Lightweight PGVector DB & Gemini Embedding Connection Test
---------------------------------------------------------
This diagnostic script ignores agent orchestration and only verifies:
1. Connection to your running local pgvector Docker container.
2. Generating embeddings using Google's current "gemini-embedding-001" model.

Author: Creative Portfolio Agent
Date: July 5, 2026
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

def main():
    print("================================================================")
    print("🧪 DIAGNOSTIC CONNECTIVITY TEST INITIALIZED")
    print("================================================================")
    
    # 1. Initialize modern 2026 Google Embeddings Model
    print("📡 Initializing 'models/gemini-embedding-001' embedder...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",  # Google's current mainline model
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        print("   -> 🟢 Embedder initialized successfully.")
    except Exception as e:
        print(f"   -> ❌ Embedder Initialization Failed: {e}")
        return

    # 2. Database Connection Parameters (Matches docker-compose.yml)
    CONNECTION_STRING = "postgresql+psycopg://postgres:postgres@localhost:5432/logistics_ai"
    COLLECTION_NAME = "diagnostic_test"

    print("\n🐘 Connecting to pgvector Docker Container...")
    try:
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )
        print("   -> 🟢 Successfully established communication channel with PostgreSQL.")
    except Exception as e:
        print(f"   -> ❌ Database Connection Refused: {e}")
        print("      Is your Docker container running? (Run 'docker ps' to verify)")
        return

    # 3. Perform Write and Read Operation
    print("\n📝 Attempting simple vector store loop...")
    test_text = "Standard Operating Capacity Squeeze Verified."
    
    try:
        # Write/Embed data
        print("   [Write] Sending test text to embedder and saving to Postgres...")
        vector_store.add_texts(
            texts=[test_text],
            metadatas=[{"source": "Diagnostic_Test", "type": "Test_Run"}]
        )
        print("   -> 🟢 Vector successfully saved.")

        # Read/Search data back
        print("   [Read] Retrieving vector using similarity search query...")
        results = vector_store.similarity_search("Capacity Squeeze", k=1)
        
        if results and len(results) > 0:
            print("\n================================================================")
            print("🎉 SUCCESS! SYSTEM FULLY INTEGRATED")
            print("================================================================")
            print(f"• Query Match:  '{results[0].page_content}'")
            print(f"• Metadata:     {results[0].metadata}")
            print("================================================================\n")
        else:
            print("   -> ❌ Read operation returned empty dataset.")
            
    except Exception as e:
        print(f"   -> ❌ Vector loop crashed: {e}")

if __name__ == "__main__":
    main()