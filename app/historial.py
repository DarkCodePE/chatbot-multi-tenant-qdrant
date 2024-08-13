import os
from pathlib import Path
from uuid import uuid4
from qdrant_client import QdrantClient, models
import logging
from dotenv import load_dotenv
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, messages_from_dict, messages_to_dict
from typing import List, Dict, Any, Optional
from qdrant_client.http.exceptions import UnexpectedResponse

# Configuration
load_dotenv()
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHAT_HISTORY_COLLECTION = "chat_history"

DOCS_FOLDER = Path("documents")

class QdrantChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, client: QdrantClient, collection_name: str):
        self.session_id = session_id
        self.client = client
        self.collection_name = collection_name
        self.configure_collection()

    def configure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            logging.info(f"La colección {self.collection_name} ya existe.")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logging.info(f"La colección {self.collection_name} no existe. Creándola...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
                )
            else:
                raise

        # Crear índice para session_id si no existe
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="session_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as e:
            if "already exists" not in str(e):
                raise

    def add_message(self, message: BaseMessage) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(
                id=str(uuid4()),
                payload={
                    "session_id": self.session_id,
                    "message": messages_to_dict([message])[0]
                },
                vector=[0.0] * 1536  # Placeholder vector
            )]
        )

    def clear(self) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=self.session_id)
                        )
                    ]
                )
            )
        )

    @property
    def messages(self) -> List[BaseMessage]:
        response = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=self.session_id)
                    )
                ]
            ),
            limit=1000  # Ajusta según sea necesario
        )
        message_dicts = [point.payload["message"] for point in response[0]]
        return messages_from_dict(message_dicts)