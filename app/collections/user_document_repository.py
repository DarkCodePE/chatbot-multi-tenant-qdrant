import os
from pathlib import Path
from qdrant_client import QdrantClient, models
import logging
from dotenv import load_dotenv
from langchain.schema import Document
from uuid import uuid4
# Configuration
load_dotenv()
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHAT_HISTORY_COLLECTION = os.getenv("CHAT_HISTORY_COLLECTION")
USER_DOCUMENTS_COLLECTION = os.getenv("USER_DOCUMENTS_COLLECTION")
TOPIC_COLLECTION = os.getenv("TOPIC_COLLECTION")

DOCS_FOLDER = Path("documents")

class UserDocumentRepository:
    def __init__(self, qdrant_client, embeddings):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = USER_DOCUMENTS_COLLECTION

    async def add_user_document(self, session_id: str, document: Document):
        vector = await self.embeddings.aembed_query(document.page_content)
        try:
            self.qdrant_client.upsert(
                collection_name=USER_DOCUMENTS_COLLECTION,
                points=[models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "content": document.page_content,
                        "metadata": document.metadata
                    }
                )]
            )
            logging.info(f"Documento de usuario añadido para la sesión {session_id}")
        except Exception as e:
            logging.error(f"Error al añadir documento de usuario: {str(e)}")
            raise

    async def search_user_documents(self, query: str, session_id: str, k: int = 5):
        query_vector = await self.embeddings.aembed_query(query)
        try:
            results = self.qdrant_client.search(
                collection_name=USER_DOCUMENTS_COLLECTION,
                query_vector=query_vector,
                filter=models.Filter(
                    must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
                ),
                limit=k
            )
            return [Document(page_content=result.payload["content"], metadata=result.payload["metadata"]) for result in
                    results]
        except Exception as e:
            logging.error(f"Error al buscar documentos de usuario: {str(e)}")
            raise