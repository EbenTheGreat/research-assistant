import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from backend.logger import logger
from backend.modules.load_vectorstore import load_vectorstore_from_docs
from backend.modules.ocr_loader import load_pdf_with_hybrid_ocr

UPLOAD_DIR = "./uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()


@router.post("/upload_pdfs/")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)

            # --- Save uploaded file ---
            with open(file_path, "wb") as f:
                f.write(await file.read())
            logger.info(f"Saved file: {file.filename}")

            # --- Extract text with hybrid OCR ---
            docs = load_pdf_with_hybrid_ocr(file_path)

            if not docs:
                logger.warning(f"No text extracted from {file.filename}")
                continue

            # --- Push docs directly to vectorstore ---
            load_vectorstore_from_docs(docs, file_path)
            logger.info(f"Document added to vectorstore: {file.filename}")

        return JSONResponse(
            content={"message": "Files processed and added to vectorstore"},
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Error in upload_pdfs: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )
