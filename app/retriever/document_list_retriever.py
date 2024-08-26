from typing import List, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from pydantic import BaseModel, Field


class DocumentListRetriever(BaseRetriever):
    documents: List[Document] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, documents: List[Document]):
        super().__init__()
        self.documents = documents

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        return self.documents

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        return self.documents
