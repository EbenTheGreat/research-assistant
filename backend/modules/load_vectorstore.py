import os
import time
from pathlib import Path
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.config.config import config

GOOGLE_API_KEY = config.GOOGLE_API_KEY
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_ENV = config.PINECONE_ENVIRONMENT
PINECONE_INDEX_BASE = config.PINECONE_INDEX_NAME

UPLOAD_DIR = "./uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_pinecone_index():
    """Return a Pinecone index, creating one if needed."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

    # get all existing indexes
    existing_indexes = {i["name"]: i for i in pc.list_indexes()}

    # check for one that starts with your base name
    matching_index = [name for name in existing_indexes if name.startswith(PINECONE_INDEX_BASE)]

    if matching_index:
        index_name = matching_index[0]
        print(f" Using existing Pinecone index: {index_name}")
    else:
        print(f" Creating new Pinecone index: {PINECONE_INDEX_BASE}")
        pc.create_index(
            name=PINECONE_INDEX_BASE,
            dimension=768,
            metric="cosine",
            spec=spec,
        )

        while not pc.describe_index(PINECONE_INDEX_BASE).status["ready"]:
            time.sleep(1)

        index_name = [i["name"] for i in pc.list_indexes() if i["name"].startswith(PINECONE_INDEX_BASE)][0]
        print(f"Created Pinecone index: {index_name}")

    return pc.Index(index_name)


index = get_pinecone_index()

from itertools import islice

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def load_vectorstore(uploaded_files):
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    file_paths = []

    # 1. Save uploaded files
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
            file_paths.append(str(save_path))

    # 2. Split and embed
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Texts & metadata
        texts = [chunk.page_content for chunk in chunks]
        metadata = []
        for chunk in chunks:
            md = chunk.metadata.copy()
            md["page_content"] = chunk.page_content  # 🔑 store the text in metadata
            metadata.append(md)

        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        # 3. Embedding
        print("Embedding chunks...")
        embeddings = embedding_model.embed_documents(texts)

        # 4. Upsert
        print("Upserting embeddings...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            vectors = list(zip(ids, embeddings, metadata))

            # Send in smaller chunks 
            for batch in chunked_iterable(vectors, 100):
                index.upsert(vectors=batch)

            progress.update(len(embeddings))

        print(f" Upload complete for {file_path}")
