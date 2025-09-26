import time
from pathlib import Path
from itertools import islice
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings
from backend.config.config import config

# ==========================================================
# Lazy Globals
# ==========================================================
_pinecone_index = None


# ==========================================================
# Helpers
# ==========================================================
def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def embed_texts_in_batches(embedding_model, texts, batch_size: int = 50):
    """Safely embed texts in smaller batches to avoid API resets."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = embedding_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {e}")
            # Retry one by one
            for text in batch:
                try:
                    single_embedding = embedding_model.embed_documents([text])[0]
                    all_embeddings.append(single_embedding)
                except Exception as inner_e:
                    print(f"Failed on single text: {inner_e}")
    return all_embeddings


# ==========================================================
# Pinecone
# ==========================================================
def get_pinecone_index():
    """Return Pinecone index, creating it if needed."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        spec = ServerlessSpec(cloud="aws", region=config.PINECONE_ENVIRONMENT)

        existing_indexes = {i["name"]: i for i in pc.list_indexes()}
        matching_index = [name for name in existing_indexes if name.startswith(config.PINECONE_INDEX_NAME)]

        if matching_index:
            index_name = matching_index[0]
            print(f" Using existing Pinecone index: {index_name}")
        else:
            print(f" Creating new Pinecone index: {config.PINECONE_INDEX_NAME}")
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=spec,
            )
            while not pc.describe_index(config.PINECONE_INDEX_NAME).status["ready"]:
                time.sleep(1)
            index_name = config.PINECONE_INDEX_NAME
            print(f"Created Pinecone index: {index_name}")

        _pinecone_index = pc.Index(index_name)
    return _pinecone_index


# ==========================================================
# Vectorstore Loader (Docs → Embeddings → Pinecone)
# ==========================================================
def load_vectorstore_from_docs(docs: list[Document], source: str):
    """
    Takes extracted Document objects and pushes them to Pinecone.
    `source` is the file path of the original PDF.
    """
    embedding_model = VoyageAIEmbeddings(
        voyage_api_key=config.VOYAGE_API_KEY,
        model="voyage-3.5"
    )

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    print(f"Embedding {len(chunks)} chunks from {Path(source).name}...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_texts_in_batches(embedding_model, texts, batch_size=50)

    # Format for Pinecone
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append((
            f"{Path(source).stem}-{i}",
            embedding,
            {
                "source": str(source),
                "page": chunk.metadata.get("page", None),
                "text": chunk.page_content
            }
        ))

    # Upsert to Pinecone
    index = get_pinecone_index()
    with tqdm(total=len(vectors), desc="Upserting to Pinecone") as progress:
        for batch in chunked_iterable(vectors, 100):
            index.upsert(vectors=batch)
            progress.update(len(batch))

    print(f"Upload complete for {Path(source).name}")
