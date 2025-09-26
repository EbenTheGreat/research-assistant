from pathlib import Path
import os
import io
import pickle
import base64
import tempfile
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from google.cloud import vision
from google.oauth2 import service_account

# -------------------------
# CACHE
# -------------------------
CACHE_DIR = "ocr_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# POPPLER PATH HANDLING
# -------------------------
POPPLER_PATH = None
if os.name == "nt":  # Windows (local dev)
    POPPLER_PATH = r"C:\Users\user\Desktop\fastapi\Release-25.07.0-0\poppler-25.07.0\Library\bin"
    print(f"[INFO] Using Windows Poppler path: {POPPLER_PATH}")
else:
    # Linux (Render) → use system poppler-utils (from apt.txt)
    POPPLER_PATH = None
    print("[INFO] Using system Poppler (Linux)")

# -------------------------
# GOOGLE VISION CREDS
# -------------------------
def load_google_credentials():
    """
    Load Google Vision credentials.
    - If GOOGLE_APPLICATION_CREDENTIAL contains a file path -> use it directly.
    - If it looks like base64 -> decode it into a temp JSON file.
    """
    raw_value = os.environ.get("GOOGLE_APPLICATION_CREDENTIAL")

    if not raw_value:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIAL not set")

    # Case 1: Local file path
    if os.path.exists(raw_value):
        print("Using Google credentials file:", raw_value)
        return service_account.Credentials.from_service_account_file(raw_value)

    # Case 2: Base64 string
    try:
        creds_bytes = base64.b64decode(raw_value)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(creds_bytes)
        tmp.flush()
        tmp.close()
        print("Using Google credentials from BASE64 env var")
        return service_account.Credentials.from_service_account_file(tmp.name)
    except Exception as e:
        raise RuntimeError(
            "Invalid GOOGLE_APPLICATION_CREDENTIAL value. Must be a file path or base64 JSON."
        ) from e


credentials = load_google_credentials()
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# -------------------------
# HELPERS
# -------------------------
def _get_cache_path(file_path: str) -> str:
    return os.path.join(CACHE_DIR, Path(file_path).stem + ".pkl")


def load_pdf_with_hybrid_ocr(file_path: str):
    """Try extracting text normally first; if none, fallback to OCR."""
    cache_path = _get_cache_path(file_path)

    # 1. Return cached results
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    documents = []

    # 2. Try PyPDFLoader first
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if documents and any(doc.page_content.strip() for doc in documents):
            for d in documents:
                d.metadata.update({"filename": Path(file_path).name})
            print(f"[INFO] Extracted text with PyPDFLoader from {Path(file_path).name}")
            with open(cache_path, "wb") as f:
                pickle.dump(documents, f)
            return documents
        else:
            print(f"[WARN] No digital text in {Path(file_path).name}, trying OCR...")
    except Exception as e:
        print(f"[ERROR] PyPDFLoader failed: {e}")

    # 3. Fallback: pdf2image + Google Vision OCR
    try:
        images = convert_from_path(file_path, dpi=200, poppler_path=POPPLER_PATH)
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            content = buf.getvalue()

            response = vision_client.text_detection(image=vision.Image(content=content))
            page_text = response.full_text_annotation.text if response.full_text_annotation else ""

            documents.append(Document(
                page_content=page_text.strip(),
                metadata={
                    "source": str(file_path),
                    "filename": Path(file_path).name,
                    "page": i + 1
                }
            ))

        print(f"[INFO] OCR extraction complete for {Path(file_path).name}")

        with open(cache_path, "wb") as f:
            pickle.dump(documents, f)

    except Exception as e:
        print(f"[ERROR] OCR failed for {file_path}: {e}")

    return documents
