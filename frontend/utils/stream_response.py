import requests
from api import API_URL

def stream_answer(question):
    with requests.post(f"{API_URL}/ask/stream", data={"query": question}, stream=True) as r:
        for line in r.iter_lines(decode_unicode=True):
            if line:
                yield line
