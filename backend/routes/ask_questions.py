from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from backend.modules.llm import get_llm_chain
from backend.modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.logger import logger
from backend.config.config import config
from backend.modules.load_vectorstore import index
from backend.modules.retriever import SimpleRetriever
import json

PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX = config.PINECONE_INDEX_NAME
GOOGLE_API_KEY = config.GOOGLE_API_KEY

router = APIRouter()

@router.post("/ask/")
async def ask_question(query: str = Form(...)):
    try:
        logger.info(f"user query: {query}")

        # Normalize query input
        if isinstance(query, dict):
            query = query.get("query", "")
        elif query.strip().startswith("{") and "query" in query:
            query = json.loads(query)["query"]

        # Embed query
        embed_model = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="models/text-embedding-004"
        )
        embedded_query = embed_model.embed_query(query)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        # Convert Pinecone results into LangChain Documents
        docs = [
            Document(
                page_content=match.metadata.get("page_content", ""),
                metadata=match.metadata
            )
            for match in res.matches
        ]

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)

        # Pass aligned keys: question + context
        result = query_chain(
            chain,
            {
                "question": query,
                "context": "\n\n".join([doc.page_content for doc in docs]),
            }
        )

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
