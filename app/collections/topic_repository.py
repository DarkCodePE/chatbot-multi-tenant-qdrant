import os
import io
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PyPDF2 import PdfReader
from googleapiclient.errors import HttpError
from qdrant_client import QdrantClient, models
import logging
from dotenv import load_dotenv
from langchain.schema import Document
from pydantic import BaseModel, ConfigDict
from langchain_openai import OpenAIEmbeddings
from app.schema.schema import TopicInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

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


class TopicRepository:
    def __init__(self, qdrant_client: QdrantClient, embeddings: OpenAIEmbeddings):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = TOPIC_COLLECTION
        self.drive_service = self._initialize_drive_service()

    def _initialize_drive_service(self):
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS no está configurado en las variables de entorno")

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=credentials)

    async def add_topic(self, topic: TopicInfo):
        vector = await self.embeddings.aembed_query(topic.description)
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=topic.topic_id,
                    vector=vector,
                    payload={
                        "name": topic.name,
                        "course_id": topic.course_id,
                        "description": topic.description,
                        "type": "topic_metadata"
                    }
                )]
            )
            logging.info(f"Tópico añadido: {topic.name}")
        except Exception as e:
            logging.error(f"Error al añadir tópico: {str(e)}")
            raise

    async def add_document(self, course_id: str, topic_id: str, document: Document):
        vector = await self.embeddings.aembed_query(document.page_content)
        vector_id = str(uuid4())
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload={
                        "course_id": course_id,
                        "topic_id": topic_id,
                        "content": document.page_content,
                        "metadata": {
                            "document_id": document.metadata.get("document_id"),
                            "file_name": document.metadata.get("file_name"),
                            "type": document.metadata.get("type", "article"),
                            "language": document.metadata.get("language", "en"),
                            **document.metadata
                        },
                    }
                )]
            )
            logging.info(f"Documento añadido al curso {course_id}, tópico {topic_id}")
            return vector_id
        except Exception as e:
            logging.error(f"Error al añadir documento: {str(e)}")
            raise

    def get_folder_id(self, folder_name):
        try:
            results = self.drive_service.files().list(
                q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'",
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            folders = results.get('files', [])
            if not folders:
                print(f'No se encontró la carpeta: {folder_name}')
                return None
            return folders[0]['id']
        except Exception as e:
            print(f'Ocurrió un error al buscar la carpeta: {str(e)}')
            return None

    def list_accessible_folders(self):
        """
        Lista todas las carpetas a las que la cuenta de servicio tiene acceso.

        :return: Lista de diccionarios con información de las carpetas accesibles
        """
        try:
            results = []
            page_token = None
            while True:
                response = self.drive_service.files().list(
                    q="mimeType='application/vnd.google-apps.folder'",
                    spaces='drive',
                    fields='nextPageToken, files(id, name, owners, permissions)',
                    pageToken=page_token
                ).execute()

                for folder in response.get('files', []):
                    folder_info = {
                        'id': folder['id'],
                        'name': folder['name'],
                        'owner': folder['owners'][0]['emailAddress'] if folder.get('owners') else 'Unknown',
                        'permissions': [
                            {
                                'type': perm['type'],
                                'role': perm['role'],
                                'emailAddress': perm.get('emailAddress', 'N/A')
                            } for perm in folder.get('permissions', [])
                        ]
                    }
                    results.append(folder_info)
                    logging.info(f"Carpeta accesible: {folder['name']} (ID: {folder['id']})")
                    for perm in folder_info['permissions']:
                        logging.info(f"  - Permiso: {perm['type']} {perm['role']} {perm['emailAddress']}")

                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break

            return results
        except Exception as error:
            logging.error(f"Error al listar carpetas accesibles: {error}")
            return []

    def check_folder_permissions(self, folder_id):
        """
        Verifica y muestra los permisos de una carpeta de Google Drive.

        :param folder_id: ID de la carpeta a verificar
        :return: Lista de permisos o None si ocurre un error
        """
        try:
            # Primero, verificamos si podemos acceder a la información de la carpeta
            folder = self.drive_service.files().get(fileId=folder_id,
                                                    fields="name, owners, sharingUser, permissions").execute()
            logging.info(f"Nombre de la carpeta: {folder['name']}")
            logging.info(f"Propietario: {folder['owners'][0]['emailAddress']}")
            logging.info(f"Compartido por: {folder.get('sharingUser', {}).get('emailAddress', 'N/A')}")

            # Luego, listamos los permisos
            permissions = folder.get('permissions', [])

            logging.info(f"Permisos para la carpeta {folder_id}:")
            for permission in permissions:
                logging.info(f"ID del permiso: {permission['id']}")
                logging.info(f"Rol: {permission['role']}")
                logging.info(f"Tipo: {permission['type']}")
                logging.info(f"Dirección de correo: {permission.get('emailAddress', 'N/A')}")
                logging.info("-" * 40)

            return permissions
        except HttpError as error:
            if error.resp.status == 404:
                logging.error(
                    f"La carpeta con ID {folder_id} no se encontró. Verifica que el ID sea correcto y que la carpeta no haya sido eliminada.")
            elif error.resp.status == 403:
                logging.error(
                    f"No tienes permiso para acceder a la carpeta con ID {folder_id}. Verifica que la cuenta de servicio tenga los permisos necesarios.")
            else:
                logging.error(f"Ocurrió un error al acceder a la carpeta: {error}")
            return None
        except Exception as error:
            logging.error(f"Error inesperado al acceder a la carpeta: {error}")
            return None

    async def process_google_drive_documents(self, folder_id: str, course_id: str, topic_id: str):
        try:
            # Verificar permisos antes de procesar
            # permissions = self.check_folder_permissions(folder_id)
            # if permissions is None:
            #     logging.error(f"No se pudo acceder a los permisos de la carpeta {folder_id}. Abortando procesamiento.")
            #     return False
            # Verificar si la carpeta es accesible
            # access = self.list_accessible_folders()
            # logging.info(f"Carpetas accesibles: {access}")

            logging.info(f"Iniciando procesamiento de documentos para curso {course_id} y tópico {topic_id}")
            logging.info(f"Buscando archivos en la carpeta de Google Drive con ID: {folder_id}")

            results = self.drive_service.files().list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name, mimeType, createdTime, modifiedTime)"
            ).execute()
            files = results.get('files', [])

            logging.info(f"Archivos encontrados en la carpeta '{folder_id}':")
            for file in files:
                logging.info(f"- {file['name']} (ID: {file['id']}, Tipo: {file['mimeType']})")

            total_files = len(files)
            processed_files = 0
            failed_files = 0

            logging.info(f"Iniciando procesamiento de {total_files} archivos")

            for file in files:
                try:
                    file_id = file['id']
                    request = self.drive_service.files().get_media(fileId=file_id)
                    file_content = io.BytesIO()
                    downloader = MediaIoBaseDownload(file_content, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()

                    if file['mimeType'] == 'application/pdf':
                        # Procesar PDF
                        pdf_reader = PdfReader(file_content)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text()
                    else:
                        # Para otros tipos de archivo, intentar decodificar como texto
                        text_content = file_content.getvalue().decode('utf-8', errors='ignore')

                    # Crear embedding y almacenar en Qdrant
                    vector = await self.embeddings.aembed_query(text_content)

                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[models.PointStruct(
                            id=str(uuid4()),
                            vector=vector,
                            payload={
                                "course_id": course_id,
                                "topic_id": topic_id,
                                "content": text_content,
                                "metadata": {
                                    "name": file.get('name'),
                                    "mimeType": file.get('mimeType'),
                                    "createdTime": file.get('createdTime'),
                                    "modifiedTime": file.get('modifiedTime')
                                },
                                "type": "document"
                            }
                        )]
                    )
                    processed_files += 1
                    logging.info(f"Procesado archivo {processed_files}/{total_files}: {file.get('name')}")

                except Exception as e:
                    logging.error(f"Error al procesar archivo {file.get('name')}: {str(e)}")

            success_rate = (processed_files / total_files) * 100 if total_files > 0 else 0

            logging.info(f"Procesamiento completado para curso {course_id} y tópico {topic_id}:")
            logging.info(f"Total de archivos: {total_files}")
            logging.info(f"Archivos procesados exitosamente: {processed_files}")
            logging.info(f"Archivos fallidos: {failed_files}")
            logging.info(f"Tasa de éxito: {success_rate:.2f}%")

            return processed_files > 0
        except Exception as e:
            logging.error(f"Error procesando documentos de Google Drive: {str(e)}")
            return False

    async def search_documents(self, query: str, course_id: str = None, topic_id: str = None, k: int = 5):
        query_vector = await self.embeddings.aembed_query(query)
        try:
            filter_conditions = [models.FieldCondition(key="type", match=models.MatchValue(value="document"))]
            if course_id:
                filter_conditions.append(
                    models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id)))
            if topic_id:
                filter_conditions.append(models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)))

            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                filter=models.Filter(must=filter_conditions),
                limit=k
            )
            return [Document(page_content=result.payload["content"],
                             metadata={**result.payload["metadata"], "course_id": result.payload["course_id"],
                                       "topic_id": result.payload["topic_id"]})
                    for result in results]
        except Exception as e:
            logging.error(f"Error al buscar documentos: {str(e)}")
            raise

    async def get_topics_for_course(self, course_id: str):
        try:
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id)),
                        models.FieldCondition(key="type", match=models.MatchValue(value="topic_metadata"))
                    ]
                ),
                limit=100
            )
            return [TopicInfo(topic_id=result.id, name=result.payload["name"],
                              course_id=result.payload["course_id"], description=result.payload["description"])
                    for result in results[0]]
        except Exception as e:
            logging.error(f"Error al obtener tópicos del curso: {str(e)}")
            raise

    async def update_document(self, document_id: str, new_content: str):
        try:
            existing_doc = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id]
            )
            if not existing_doc:
                raise ValueError(f"Documento no encontrado: {document_id}")

            new_vector = await self.embeddings.aembed_query(new_content)

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

            logging.info(f"Documento actualizado: {document_id}")
            return True
        except Exception as e:
            logging.error(f"Error al actualizar documento: {str(e)}")
            return False

    async def delete_document(self, document_id: str):
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[document_id])
            )
            logging.info(f"Documento eliminado: {document_id}")
            return True
        except Exception as e:
            logging.error(f"Error al eliminar documento: {str(e)}")
            return False
