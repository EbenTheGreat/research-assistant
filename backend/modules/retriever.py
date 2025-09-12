from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Any, Optional

class PineconeRetriever(BaseRetriever, BaseModel):
    index: Any
    embeddings: Any = Field(...)  # declare as a field
    top_k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        response = self.index.query(vector=vector, top_k=self.top_k, include_metadata=True)
        return [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in response["matches"]
        ]

class SimpleRetriever(BaseRetriever):
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[dict] = Field(default_factory=dict)

    def __init__(self, documents: List[Document]):
        super().__init__()
        self._docs = documents

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._docs

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._docs

