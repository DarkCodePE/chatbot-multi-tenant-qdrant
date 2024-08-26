from typing import List, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from pydantic import BaseModel, Field


class CustomQdrantRetrieverConfig(BaseModel):
    client: QdrantClient
    collection_name: str
    embeddings: Any
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True


class CustomQdrantRetriever(BaseRetriever):
    config: CustomQdrantRetrieverConfig

    def __init__(self, config: CustomQdrantRetrieverConfig):
        super().__init__(config=config)

    def _get_relevant_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        query_vector = self.config.embeddings.embed_query(query)

        filter_conditions = [FieldCondition(key="type", match=MatchValue(value="document"))]

        results = self.config.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=filter_conditions),
            limit=self.config.k
        )

        documents = []
        for result in results:
            content = result.payload.get("content", "")
            metadata = {
                "id": result.id,
                "score": result.score,
                "course_id": result.payload.get("course_id"),
                "topic_id": result.payload.get("topic_id"),
                **result.payload.get("metadata", {})
            }
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    async def _aget_relevant_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
