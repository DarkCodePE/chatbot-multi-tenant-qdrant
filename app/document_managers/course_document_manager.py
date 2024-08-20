# app/document_managers/course_document_manager.py

import io
from datetime import datetime
from uuid import uuid4
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
import logging


class CourseDocumentManager:
    def __init__(self, qdrant_client: QdrantClient, embeddings: OpenAIEmbeddings):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.drive_service = self._initialize_drive_service()
        self.collection_name = "course_documents"

    def _initialize_drive_service(self):
        credentials = service_account.Credentials.from_service_account_file(
            'path/to/service_account.json',
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=credentials)

    async def process_course_document(self, file_id: str, course_id: str, topic_id: str):
        try:
            # Obtener metadatos del archivo
            file_metadata = self.drive_service.files().get(fileId=file_id).execute()

            # Descargar el contenido del archivo
            request = self.drive_service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                _, done = downloader.next_chunk()

            content = file.getvalue().decode('utf-8')

            # Generar embedding
            vector = await self.embeddings.aembed_query(content)

            # Preparar punto para Qdrant
            point = models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "file_id": file_id,
                    "course_id": course_id,
                    "topic_id": topic_id,
                    "content": content,
                    "metadata": {
                        "name": file_metadata.get('name'),
                        "mimeType": file_metadata.get('mimeType'),
                        "createdTime": file_metadata.get('createdTime'),
                        "modifiedTime": file_metadata.get('modifiedTime')
                    }
                }
            )

            # Insertar en Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logging.info(f"Documento del curso procesado y almacenado: {file_metadata.get('name')}")
            return True
        except Exception as e:
            logging.error(f"Error al procesar documento del curso: {str(e)}")
            return False

    async def get_course_documents(self, course_id: str, topic_id: str = None, limit: int = 10):
        try:
            filter_conditions = [
                models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id))
            ]
            if topic_id:
                filter_conditions.append(models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)))

            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * 1536,  # Dummy vector para búsqueda por filtro
                query_filter=models.Filter(must=filter_conditions),
                limit=limit
            )

            return [
                {
                    "id": point.id,
                    "content": point.payload.get("content"),
                    "metadata": point.payload.get("metadata")
                } for point in search_result
            ]
        except Exception as e:
            logging.error(f"Error al recuperar documentos del curso: {str(e)}")
            return []

    async def delete_course_document(self, document_id: str):
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[document_id])
            )
            logging.info(f"Documento del curso eliminado: {document_id}")
            return True
        except Exception as e:
            logging.error(f"Error al eliminar documento del curso: {str(e)}")
            return False

    async def update_course_document(self, document_id: str, new_content: str):
        try:
            # Obtener el documento existente
            existing_doc = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id]
            )
            if not existing_doc:
                raise ValueError(f"Documento no encontrado: {document_id}")

            # Generar nuevo embedding
            new_vector = await self.embeddings.aembed_query(new_content)

            # Actualizar el punto en Qdrant
            updated_point = models.PointStruct(
                id=document_id,
                vector=new_vector,
                payload={
                    **existing_doc[0].payload,
                    "content": new_content,
                    "metadata": {
                        **existing_doc[0].payload.get("metadata", {}),
                        "modifiedTime": datetime.now().isoformat()
                    }
                }
            )

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )

            logging.info(f"Documento del curso actualizado: {document_id}")
            return True
        except Exception as e:
            logging.error(f"Error al actualizar documento del curso: {str(e)}")
            return False