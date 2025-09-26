import requests
from frontend.config import API_URL

def upload_pdfs(files):
    files_payload = [("files", (f.name, f.read(), "application/pdf")) for f in files]
    return requests.post(f"{API_URL}/upload_pdfs/", files=files_payload)


def ask_questions(question):
    return requests.post(f"{API_URL}/ask/", data={"query": question})


def ask_questions_stream(question):
    """Stream response from the backend."""
    with requests.post(
        f"{API_URL}/ask/stream",
        data={"query": question},  # must match backend Form param
        stream=True,
    ) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk













