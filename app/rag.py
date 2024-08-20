import os
import asyncio
from pathlib import Path
from uuid import uuid4

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from qdrant_client import QdrantClient, models

import logging
from dotenv import load_dotenv

from app.collections import TopicRepository, UserDocumentRepository
from langchain.schema import Document
from pydantic import BaseModel, ConfigDict
from langsmith.wrappers import wrap_openai

from app.database import Database
from app.schema.schema import TopicInfo, QuestionV2
from app.services.rag_service import RAGService
import openai

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

# Wrapping OpenAI client
openai_client = wrap_openai(openai.Client())


class RAG:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", client=openai_client)
        self.topic_repo = TopicRepository(self.qdrant_client, self.embeddings)
        self.user_doc_repo = UserDocumentRepository(self.qdrant_client, self.embeddings)
        self.database = Database()
        self.rag_service = RAGService(self.database, self, self.llm)
        self.collection_name = TOPIC_COLLECTION

    async def initialize(self):
        collections = [TOPIC_COLLECTION, USER_DOCUMENTS_COLLECTION]
        for collection in collections:
            try:
                self.qdrant_client.get_collection(collection)
            except Exception:
                self.qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
                )
            self.qdrant_client.create_payload_index(
                collection_name=collection,
                field_name="course_id" if collection == TOPIC_COLLECTION else "session_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def add_topic(self, topic: TopicInfo):
        return await self.topic_repo.add_topic(topic)

    async def add_user_document(self, session_id: str, document: Document):
        return await self.user_doc_repo.add_user_document(session_id, document)

    async def add_document_to_topic(self, topic_id: str, document: Document):
        return await self.topic_repo.add_document(topic_id, document)

    async def start_chat_session(self, user_id: str, course_id: str = None, topic_id: str = None):
        return await self.rag_service.start_chat_session(user_id, course_id, topic_id)

    async def end_chat_session(self, chat_session_id: str):
        return await self.rag_service.end_chat_session(chat_session_id)

    async def process_google_drive_folder(self, folder_name: str, course_id: str, topic_id: str):
        """
        Método para procesar documentos de Google Drive y añadirlos a un tema específico.
        Este método no se ejecuta automáticamente al iniciar un chat, sino que debe ser llamado explícitamente.
        """
        return await self.topic_repo.process_google_drive_documents(folder_name, course_id, topic_id)
async def main():
    rag_instance = RAG()
    await rag_instance.initialize()


if __name__ == "__main__":
    asyncio.run(main())
