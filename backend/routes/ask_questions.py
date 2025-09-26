from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, StreamingResponse
from backend.modules.llm import get_llm_chain
from backend.modules.query_handlers import query_chain
from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from backend.logger import logger
from backend.config.config import config
from backend.modules.load_vectorstore import get_pinecone_index
import json
import os

os.environ.setdefault("PINECONE_API_KEY", config.PINECONE_API_KEY)

PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX = config.PINECONE_INDEX_NAME
VOYAGEAI_API_KEY = config.VOYAGE_API_KEY

router = APIRouter()


@router.post("/ask/")
async def ask_question(query: str = Form(...)):
    try:
        logger.info(f"Raw user query: {query}")

        # --- Normalize query ---
        if isinstance(query, dict):
            query = query.get("query", "")
        elif query.strip().startswith("{") and "query" in query:
            query = json.loads(query)["query"]

        logger.info(f"Final query after normalization: '{query}'")

        # --- Ensure Pinecone index exists ---
        index = get_pinecone_index()

        # --- Embedding model ---
        embed_model = VoyageAIEmbeddings(
            voyage_api_key=VOYAGEAI_API_KEY,
            model="voyage-3.5"
        )

        # --- Pinecone vector store ---
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX,
            embedding=embed_model
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # --- LLM chain ---
        chain = get_llm_chain(retriever)

        # --- Retrieve docs ---
        retrieved_docs = retriever.invoke(query)

        sources = [
            {
                "source": d.metadata.get("source", ""),
                "page": d.metadata.get("page"),
                "content": d.page_content,
                "extra": d.metadata,
            }
            for d in retrieved_docs
        ]

        # --- Run chain ---
        result = query_chain(chain, {"input": query})

        logger.info("Query successful")

        return JSONResponse(
            content={
                "response": result["response"],
                "sources": sources
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@router.post("/ask/stream")
async def ask_question_stream(query: str = Form(...)):
    try:
        logger.info(f"Raw user query (stream): {query}")

        # Normalize
        if isinstance(query, dict):
            query = query.get("query", "")
        elif query.strip().startswith("{") and "query" in query:
            query = json.loads(query)["query"]

        logger.info(f"Final query after normalization: '{query}'")

        # Pinecone + embeddings
        index = get_pinecone_index()
        embed_model = VoyageAIEmbeddings(
            voyage_api_key=VOYAGEAI_API_KEY,
            model="voyage-3.5"
        )
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX,
            embedding=embed_model
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Chain (returns plain text now, thanks to Option A)
        chain = get_llm_chain(retriever)

        # Retrieve docs separately for sources
        retrieved_docs = retriever.invoke(query)
        sources = [
            {
                "source": d.metadata.get("source", ""),
                "page": d.metadata.get("page"),
                "content": d.page_content,
                "extra": d.metadata,
            }
            for d in retrieved_docs
        ]

        # Streaming generator
        def generate():
            try:
                for token in chain.stream({"input": query}):
                    yield token  # directly stream text tokens
                yield "\n\n[SOURCES]" + json.dumps(sources)
            except Exception as e:
                yield f"\n\n[ERROR]{str(e)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in ask_question_stream: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
