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

# 🎉 Final Outcome

After fixing all these errors:

- ✅ PDF uploads worked  
- ✅ OCR ran with Google Vision  
- ✅ Vector embeddings stored in Pinecone  
- ✅ Queries returned correct results from the knowledge base  

The system now works end-to-end 🎊

---
