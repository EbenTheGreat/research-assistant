from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from typing import Union

import hashlib
import os
import torch


load_dotenv()


# __________________________________STEPS__________________________#

# __________________STEP 1 - Setting Up the Knowledge Base (PINECONE)__________________#

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "ml-publications"

# create index if not already present
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# set up embeddings
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# connect langchain to pincecone
INDEX = pc.Index(index_name)
vector_store = PineconeVectorStore(INDEX, EMBEDDINGS, text_key="text")


# __________________Step 2: Loading the Publications__________________#
def load_research_publications(documents_path):
    documents = []
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()

                for doc in loaded_docs:
                    doc.metadata["source"] = file

                documents.extend(loaded_docs)
                print(f"Successfully loaded: {file}")

            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


# ______________________Step 3: Chunk The Publications________________#

def generate_id(text: str | Union, title: str) -> str:
    """Generate a stable hash ID for a text chunk."""
    content_to_hash = (title + text).encode("utf-8")
    return hashlib.sha256(content_to_hash).hexdigest()


def chunk_research_paper(paper_content, title):
    # chunk the document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents=paper_content)

    # add metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "id": generate_id(chunk, title),
            "content": chunk,
            "title": title,
        })

    return chunk_data


# _____________________Step 4 – Creating Embeddings__________________#
def embed_documents(documents: list[str]) -> list[list[float]]:
    """Embed documents using HuggingFace embeddings"""
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    return model.embed_documents(documents)

# _____________________Step 5 – Store Embeddings in Pinecone__________________#
def insert_publications(index: type(INDEX), publications, embeddings_model):
    """Insert chunked publications into Pinecone with metadata"""
    for publication_id, publication in enumerate(publications):
        chunked_publication = chunk_research_paper(publication, f"publication_id: {publication_id}")
        # extract texts
        texts = [chunk["content"] for chunk in chunked_publication]
        # Generate embeddings
        vectors = embeddings_model.embed_documents(texts)

        # Format Pinecone upsert payload
        pinecone_vectors = []
        for chunk, vector in zip(chunked_publication, vectors):
            pinecone_vectors.append({
                "id": chunk["id"],
                "values": vector,
                "metadata": {
                    "title": chunk["title"],
                    "content": chunk["content"]
                }
            })

        # Upsert into Pinecone (insert or update if exists)
        index.upsert(vectors=pinecone_vectors)


# _______________________Step 6 – Intelligent Retrieval_____________________#
def search_research_db(query, index: type(INDEX), embeddings: type(EMBEDDINGS), top_k=5):
    """Find most relevant research chunks from Pinecone"""
    query_vector = embeddings.embed_query(query)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    relevant_chunks = []
    for match in results["matches"]:
        relevant_chunks.append({
            "content": match["metadata"].get("content", ""),
            "title": match["metadata"].get("title", ""),
            "similarity": match["score"],
        })

    return relevant_chunks

def answer_research_question(query, index: type(INDEX), embeddings: type(EMBEDDINGS), llm):
    relevant_chunks = search_research_db(query, index, embeddings, top_k=3)

    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following research findings, answer the researcher's question:

Research Context:
{context}

Researcher's Question: {question}

Answer: Provide a comprehensive answer based on the research findings above.
"""
    )

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, relevant_chunks


# Run an example
llm = ChatGroq(model="llama3-8b-8192")

answer, sources = answer_research_question(
    "What are effective techniques for handling class imbalance?",
    index,
    embeddings,
    llm
)

print("AI Answer:", answer)
print("\nBased on sources:")
for source in sources:
    print(f"- {source['title']}")









