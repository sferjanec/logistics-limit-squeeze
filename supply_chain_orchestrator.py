#!/usr/bin/env python3
"""
LangGraph Logistics State Machine with PGVector RAG
---------------------------------------------------
This architecture implements a persistent Retrieval-Augmented Generation (RAG) loop.
It uses Google's embedding models and a local PostgreSQL (pgvector) container 
to detect 'Strategy Drift' across execution runs.

Author: Creative Portfolio Agent
Date: July 5, 2026
"""

import os
import json
from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

# 1. Initialize the LLM and Embedding Model
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    api_key=os.environ.get("GEMINI_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# 2. Database Connection String (Matches docker-compose.yml)
CONNECTION_STRING = "postgresql+psycopg://postgres:postgres@localhost:5432/logistics_ai"
COLLECTION_NAME = "executive_memos"

# Initialize PGVector Store
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# 3. Define the State Graph Memory
class AgentState(TypedDict):
    limit_data: str
    historical_context: str
    final_memo: str

# 4. Define the Nodes
def analyst_node(state: AgentState):
    print("🤖 Analyst: Evaluating mathematical limits...")
    # In a real run, this reads sourcing_config.json. Hardcoded for demo stability.
    limits = "Max Safe Scaling: 4.68. Violation penalty: $320k capital decay."
    return {"limit_data": limits}

def memory_retrieval_node(state: AgentState):
    print("🤖 Retriever: Querying PGVector for historical strategy drift...")
    try:
        # We query the vector database for memos similar to our current limit topic
        docs = vector_store.similarity_search("scaling limits capital loss", k=1)
        if docs:
            history = docs[0].page_content
            print("   -> 📂 Found previous executive memo in Postgres!")
        else:
            history = "No historical memos found. This is the first recorded incident."
            print("   -> 📭 Postgres is empty. No strategy drift detected.")
    except Exception as e:
        history = f"Memory retrieval failed. Postgres may not be running. Error: {e}"
        print(f"   -> ⚠️ {history}")
        
    return {"historical_context": history}

def coo_node(state: AgentState):
    print("🤖 COO: Drafting Strategy Memo...")
    prompt = (
        f"Current Math Limits: {state['limit_data']}\n"
        f"Historical Memory: {state['historical_context']}\n\n"
        "Draft a 2-paragraph executive memo. Paragraph 1 should state the current limits. "
        "Paragraph 2 should compare this to the 'Historical Memory' to identify any 'Strategy Drift'."
    )
    
    response = gemini_llm.invoke([HumanMessage(content=prompt)])
    return {"final_memo": response.content}

def archivist_node(state: AgentState):
    print("🤖 Archivist: Embedding and saving final memo to PGVector...")
    memo = state['final_memo']
    try:
        # Save the new memo into the vector database for future runs
        vector_store.add_texts(
            texts=[memo],
            metadatas=[{"source": "COO_Agent", "type": "Scaling_Memo"}]
        )
        print("   -> 💾 Successfully committed to Postgres.")
    except Exception as e:
        print(f"   -> ⚠️ Failed to save to Postgres: {e}")
    return state # State remains unchanged, we just caused a side-effect

# 5. Define the Graph Routing
workflow = StateGraph(AgentState)

workflow.add_node("analyst", analyst_node)
workflow.add_node("retriever", memory_retrieval_node)
workflow.add_node("coo", coo_node)
workflow.add_node("archivist", archivist_node)

# The Execution Flow
workflow.set_entry_point("analyst")
workflow.add_edge("analyst", "retriever")
workflow.add_edge("retriever", "coo")
workflow.add_edge("coo", "archivist") # After drafting, save it!
workflow.add_edge("archivist", END)

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    print("\n" + "="*64)
    print("🌐 BOOTING LANGGRAPH: PGVECTOR RAG PIPELINE")
    print("="*64)
    
    initial_state = {"limit_data": "", "historical_context": "", "final_memo": ""}
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*64)
    print("📥 FINAL EXECUTIVE BOARD MEMO")
    print("="*64)
    print(final_state["final_memo"])
    print("="*64 + "\n")