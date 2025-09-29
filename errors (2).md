# 🐛 Debugging Journey — Research Assistant with FastAPI, OCR & Pinecone

This file documents every major error encountered while building the project, along with the **cause, fix, and lesson learned**.  
It serves as a debugging diary to help others avoid the same pitfalls.

---

## 1. `'Document' object has no attribute 'filename'`

**Error:**
```text
'Document' object has no attribute 'filename'
```

**Cause:**  
LangChain `Document` objects don’t have a `.filename` property — only FastAPI’s `UploadFile` does.

**Fix:**  
Use `UploadFile.filename` when saving files instead of trying to access `.filename` on `Document`.

**Lesson Learned:**  
LangChain `Document` ≠ FastAPI `UploadFile`. Keep their roles separate.

---

## 2. Pinecone Connection Error

**Error:**
```text
HTTPSConnectionPool(host='api.pinecone.io', port=443): Max retries exceeded...
Failed to resolve 'api.pinecone.io' ([Errno 11002] getaddrinfo failed)
```

**Cause:**  
Your local machine couldn’t resolve Pinecone’s API hostname (DNS/network issue).

**Fix:**  
- Verified Pinecone API key.  
- Checked internet/DNS.  
- Eventually resolved when the connection stabilized.

**Lesson Learned:**  
Not all errors are code-related. Sometimes it’s DNS or network.

---

## 3. Google Vision OCR Billing Disabled

**Error:**
```text
OCR failed ... This API method requires billing to be enabled.
```

**Cause:**  
Google Cloud Vision API requires billing enabled even for free-tier usage.

**Fix:**  
Enabled billing on the Google Cloud project.

**Lesson Learned:**  
Google Vision won’t work without billing enabled. Always check API requirements.

---

## 4. Action Unsuccessful `[OR_BACR2_44]`

**Error:**
```text
Action unsuccessful
This action couldn't be completed. [OR_BACR2_44]
```

**Cause:**  
Error came from Google Cloud console when trying to set up credentials.

**Fix:**  
Generated a **new service account key**:
```
rag-assistant-473015-664df326dd89.json
```

**Lesson Learned:**  
When credentials break, regenerating a new key is often the fastest fix.

---

## 5. Google Credentials File Not Found

**Error:**
```text
google.auth.exceptions.DefaultCredentialsError:
File C:\...\backend\config.json was not found.
```

**Cause:**  
The app was pointing to `backend/config.json`, but the real service account file was at:

```
backend/config/rag-assistant-473015-664df326dd89.json
```

**Fix:**  
Corrected the `.env` variable:

```env
GOOGLE_APPLICATION_CREDENTIALS=C:/Users/user/Desktop/fastapi/Agentic AI Course/week 3/research-assistaant/backend/config/rag-assistant-473015-664df326dd89.json
```

**Lesson Learned:**  
- Always double-check file paths.  
- On Windows, prefer **forward slashes (`/`)** in `.env`.

---

## 6. Wrong Config Path Still Being Picked Up

**Log Output:**
```text
Using Google credentials file: ...\backend\config.json
FileNotFoundError: No such file or directory: '...\backend\config.json'
```

**Cause:**  
Even after updating `.env`, code in `ocr_loader.py` was still hardcoded to look for `config.json`.

**Fix:**  
- Searched the project for `"config.json"`.  
- Updated `ocr_loader.py` to use the environment variable `GOOGLE_APPLICATION_CREDENTIALS`.  

**Lesson Learned:**  
Environment variables should be the **single source of truth**. Avoid hardcoding sensitive paths.

---

## 7. `ModuleNotFoundError: No module named 'frontend'`

**Error:**
```text
ModuleNotFoundError: No module named 'frontend'
```

**Cause:**  
Python couldn’t recognize `frontend` as a package when deployed on Render, even though it worked locally. This happened because `__init__.py` files were missing and packaging was inconsistent.

**Fix:**  
- Added empty `__init__.py` files inside `backend`, `frontend`, `ocr_cache`, `uploaded_documents`.  
- Created a `pyproject.toml` with `setuptools` configuration to ensure proper packaging.

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "research-assistant"
version = "0.1.0"
description = "RAG app with FastAPI + Streamlit"
dependencies = []

[tool.setuptools]
packages = ["backend", "frontend", "ocr_cache", "uploaded_documents"]
```

**Lesson Learned:**  
Deployment environments are stricter than local dev. Always add `__init__.py` and a proper packaging file (`pyproject.toml`).

---

## 8. `requests.exceptions.HTTPError` from Streamlit

**Error:**
```text
requests.exceptions.HTTPError: This app has encountered an error...
```

**Cause:**  
The frontend’s `ask_questions_stream` was calling the backend, but either:
- Wrong `API_URL`, or
- The backend endpoint `/ask/stream` returned an error (status 500).

**Fix:**  
- Verified `API_URL` in `frontend/config.py`:

```python
# API_URL = "http://127.0.0.1:8000"
API_URL = "https://research-assistant-oe9n.onrender.com/"
```

- Tested backend with Restfox/Postman to confirm `/ask/stream` works.  
- Ensured `requests.post(..., data={"query": question}, stream=True)` matches the backend’s `Form` parameter.

**Lesson Learned:**  
- Always test backend endpoints directly with Postman/Restfox before blaming the frontend.  
- `HTTPError` often means status `500` from backend, not a frontend bug.

---

## 9. Trailing Slash Bug in API_URL

**Error:**  
Frontend crashed with `requests.exceptions.HTTPError`, even though the backend was live.

**Cause:**  
In `frontend/config.py`, `API_URL` had a trailing slash:

```python
API_URL = "https://research-assistant-oe9n.onrender.com/"
```

So when concatenating:

```python
f"{API_URL}/ask/stream"
```

It became:

```
https://research-assistant-oe9n.onrender.com//ask/stream
```

(double `//`), which is treated as a different route.

**Fix:**  
Remove the trailing slash:

```python
API_URL = "https://research-assistant-oe9n.onrender.com"
```

Or normalize:

```python
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
```

**Lesson Learned:**  
Tiny string issues (`/` vs no `/`) can break APIs. Normalize URLs before concatenation.

---

## 10. ✅ Restfox/Postman Debugging Workflow

To debug `/ask/stream`, we used Restfox:

- **Method:** `POST`  
- **URL:**  
  ```
  https://research-assistant-oe9n.onrender.com/ask/stream
  ```
- **Body:** (set to `x-www-form-urlencoded`)  
  ```
  query=nux mg 30
  ```

If the backend is working correctly, you should see a **streamed response** (chunks of text).  
If you get a `500`, the issue is in the backend’s `/ask/stream` logic, not the frontend.

**Lesson Learned:**  
When debugging API errors:
1. Test with Restfox/Postman.  
2. Check backend logs for stack traces.  
3. Only after confirming the backend works, connect the frontend.  

---

# 🎉 Final Outcome

After fixing all these errors:

- ✅ PDF uploads worked  
- ✅ OCR ran with Google Vision  
- ✅ Vector embeddings stored in Pinecone  
- ✅ Queries returned correct results from the knowledge base  
- ✅ Streamlit frontend successfully communicated with FastAPI backend  

The system now works end-to-end 🎊
