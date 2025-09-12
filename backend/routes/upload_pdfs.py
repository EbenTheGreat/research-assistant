from fastapi import APIRouter, UploadFile, File
from typing import List
from fastapi.responses import JSONResponse
from backend.logger import logger
from backend.modules.load_vectorstore import load_vectorstore


router = APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info("Received uploaded files")
        load_vectorstore(files)
        logger.info("Document added to vectorstore")
        return{"message": "File processed and vector store updated"}

    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})







