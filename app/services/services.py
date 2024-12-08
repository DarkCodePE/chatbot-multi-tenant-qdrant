# app/services.py
import io
import os
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict
from uuid import uuid4
from fastapi import HTTPException, BackgroundTasks
from googleapiclient.http import MediaIoBaseDownload
from langchain.schema import Document
from langchain.schema import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START, END
from langgraph.errors import NodeInterrupt
from langgraph.graph import StateGraph
from pydantic import Field
from IPython.display import Image, display
from app.collections import TopicRepository
from app.generator.rag import RAG, TopicInfo
import logging
import asyncio
from app.model import User as UserModel, Course as CourseModel, Topic as TopicModel, Question as QuestionModel, \
    ChatSession, Document as DocumentModel, Course, Topic, ProcessedDocument, user_course
from app.retriever.custom_qdrant_retriever import CustomQdrantRetriever, CustomQdrantRetrieverConfig
from app.retriever.document_list_retriever import DocumentListRetriever
from app.schema.schema import UserLogin, UserResponse, DocumentCreate, CourseAssignment, CourseCreate, CourseResponse, \
    TopicCreate, TopicResponse, Question, Feedback, QuestionV2, DocumentAddToTopic, ChatSessionStart, ChatListResponse, \
    ChatListItem, UploadDocument, UserCreate, CourseUpdate, UserBase, State
from sqlalchemy.orm import Session
from app.model import User as UserModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langsmith import traceable
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from app.historial import QdrantChatMessageHistory
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
import openai
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from app.services.util import get_password_hash, verify_password
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

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
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
if not GOOGLE_DRIVE_FOLDER_ID:
    raise ValueError("GOOGLE_DRIVE_FOLDER_ID no está configurado en las variables de entorno")

# Wrapping OpenAI client
openai_client = wrap_openai(openai.Client())


def get_session_history(session_id: str):
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantChatMessageHistory(session_id, qdrant_client, CHAT_HISTORY_COLLECTION)


# Singleton para RAG
class RAGSingleton:
    _instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = RAG()
                await cls._instance.initialize()
            return cls._instance


class UserService:
    def __init__(self, database):
        self.database = database

    async def get_unassigned_users(self, course_id: str, db: Session) -> List[UserResponse]:
        """
        Obtiene la lista de usuarios que no están asignados a un curso específico.

        Args:
            course_id: ID del curso
            db: Sesión de base de datos

        Returns:
            Lista de usuarios no asignados al curso
        """
        try:
            # Verificar que el curso existe
            course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
            if not course:
                raise HTTPException(status_code=404, detail="Course not found")

            # Subconsulta para obtener los IDs de usuarios ya asignados al curso
            assigned_users = (
                db.query(UserModel.id)
                .join(user_course)
                .filter(user_course.c.course_id == course_id)
                .subquery()
            )

            # Consulta principal para obtener usuarios no asignados
            unassigned_users = (
                db.query(UserModel)
                .filter(UserModel.id.notin_(assigned_users))
                .all()
            )

            # Transformar los resultados al formato de respuesta
            return [
                UserResponse(
                    id=user.id,
                    name=user.name,
                    email=user.email,
                    group_id=user.group_id,
                    session_id=user.session_id,
                    courses=[course.name for course in user.courses]
                ) for user in unassigned_users
            ]

        except HTTPException as e:
            raise e
        except Exception as e:
            logging.error(f"Error getting unassigned users: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def register_user(self, user: UserCreate, db: Session):
        db_user = self.database.get_user_by_email(db, user.email)
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = get_password_hash(user.password)
        new_user = UserModel(
            name=user.name,
            email=user.email,
            hashed_password=hashed_password,
            group_id="1",
            session_id=str(uuid4())
        )
        db_user = self.database.create_user(db, new_user)

        return self.create_user_response(db_user)

    async def login_user(self, user: UserLogin, db: Session):
        db_user = self.database.get_user_by_email(db, user.email)
        if db_user is None or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect email or password")

        db_user.session_id = str(uuid4())
        db.commit()

        return self.create_user_response(db_user)

    def create_user_response(self, db_user):
        user_courses = [course.name for course in db_user.courses]
        return {
            "id": db_user.id,
            "name": db_user.name,
            "email": db_user.email,
            "session_id": db_user.session_id,
            "courses": user_courses
        }

    def get_user(self, user_id: str, db: Session):
        db_user = self.database.get_user_by_id(db, user_id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return db_user

    async def add_user_document(self, user_id: str, document: DocumentCreate, db: Session):
        db_user = self.database.get_user_by_id(db, user_id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")

        try:
            langchain_document = Document(
                page_content=document.content,
                metadata=document.metadata
            )
            rag_instance = await RAGSingleton.get_instance()
            await rag_instance.add_user_document(user_id, langchain_document)
            return {"message": "Document added successfully"}
        except Exception as e:
            logging.error(f"Error adding document for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_user_courses(self, user_id: str, db: Session):
        db_user = self.database.get_user_by_id(db, user_id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return [CourseResponse.from_orm(course) for course in db_user.courses]

    def assign_course_to_user(self, assignment: CourseAssignment, db: Session):
        db_user = self.database.get_user_by_id(db, assignment.user_id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")

        db_course = self.database.get_course_by_id(db, assignment.course_id)
        if db_course is None:
            raise HTTPException(status_code=404, detail="Course not found")

        if db_course not in db_user.courses:
            db_user.courses.append(db_course)
            db.commit()
            return {"message": f"User {db_user.name} assigned to course {db_course.name}"}
        else:
            return {"message": "User already assigned to this course"}

    #get_course_folder_id
    def get_course_folder_id(self, course_id: str, db: Session):
        course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        return course.google_drive_folder_id


class CourseService:
    def __init__(self, database):
        self.database = database

    async def remove_user_from_course(self, course_id: str, user_id: str, db: Session):
        """
        Desasigna un usuario de un curso.
        """
        try:
            course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
            if not course:
                raise HTTPException(status_code=404, detail="Course not found")

            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            if user in course.users:
                course.users.remove(user)
                db.commit()
                return {"message": f"User {user.name} removed from course {course.name}"}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="User is not assigned to this course"
                )

        except HTTPException as e:
            raise e
        except Exception as e:
            db.rollback()
            logging.error(f"Error removing user from course: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def upload_document(self, file: UploadDocument, db: Session):
        try:
            # Inicializar el TopicRepository para acceder a los métodos de Google Drive
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            embeddings = OpenAIEmbeddings()
            topic_repository = TopicRepository(qdrant_client, embeddings)
            course = self.database.get_course_by_id(db, file.course_id)
            if course is None:
                raise HTTPException(status_code=404, detail="Course not found")
            new_processed_doc = await topic_repository.upload_document_to_drive(file.course_id,
                                                                                course.google_drive_folder_id,
                                                                                file.file_name, file.file_content,
                                                                                file.mime_type)
            logging.info(f"Documento subido: {new_processed_doc}")
            db.add(new_processed_doc)
            db.commit()
            db.refresh(new_processed_doc)
            return new_processed_doc
        except Exception as e:
            logging.error(f"Error al subir el documento: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def download_document(self, document_id: str, db: Session) -> tuple[bytes, str]:
        """
        Descarga un documento desde Google Drive.

        Args:
            document_id: ID del documento en la base de datos
            db: Sesión de base de datos

        Returns:
            Tupla con el contenido del archivo y el nombre del archivo
        """
        try:
            # Obtener el documento procesado de la base de datos
            processed_doc = db.query(ProcessedDocument).filter(
                ProcessedDocument.id == document_id
            ).first()

            if not processed_doc:
                raise HTTPException(status_code=404, detail="Document not found")

            topic_repository = TopicRepository(QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY), OpenAIEmbeddings())
            # Obtener el archivo desde Google Drive
            request = topic_repository.drive_service.files().get_media(
                fileId=processed_doc.google_file_id
            )

            file_metadata = topic_repository.drive_service.files().get(
                fileId=processed_doc.google_file_id,
                fields='name, mimeType'
            ).execute()

            # Descargar el contenido del archivo
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            return file_content.getvalue(), file_metadata.get('name', 'downloaded_file')

        except Exception as e:
            logging.error(f"Error downloading document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading document: {str(e)}")

    async def delete_document(self, document_id: str, db: Session) -> dict:
        """
        Elimina un documento de Google Drive y de la base de datos.

        Args:
            document_id: ID del documento en la base de datos
            db: Sesión de base de datos

        Returns:
            Diccionario con mensaje de confirmación
        """
        try:
            # Obtener el documento procesado de la base de datos
            processed_doc = db.query(ProcessedDocument).filter(
                ProcessedDocument.id == document_id
            ).first()

            if not processed_doc:
                raise HTTPException(status_code=404, detail="Document not found")

            topic_repository = TopicRepository(QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY), OpenAIEmbeddings())
            # Eliminar el archivo de Google Drive
            try:
                topic_repository.drive_service.files().delete(
                    fileId=processed_doc.google_file_id
                ).execute()
            except Exception as e:
                logging.error(f"Error deleting file from Google Drive: {str(e)}")
                # Continuamos incluso si falla la eliminación en Google Drive
                # para mantener la consistencia en nuestra base de datos

            # Eliminar el registro de la base de datos
            db.delete(processed_doc)
            db.commit()

            return {
                "message": "Document deleted successfully",
                "document_id": document_id
            }

        except Exception as e:
            db.rollback()
            logging.error(f"Error deleting document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

    async def create_course(self, course: CourseCreate, db: Session):
        try:
            # Inicializar el TopicRepository para acceder a los métodos de Google Drive
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            embeddings = OpenAIEmbeddings()
            topic_repository = TopicRepository(qdrant_client, embeddings)

            # Buscar la carpeta en Google Drive
            folder_id = topic_repository.get_folder_id(course.name)
            logging.info(f"Carpeta encontrada, folder_id {folder_id}")
            if not folder_id:
                # Si la carpeta no existe, la creamos
                folder_metadata = {
                    'name': course.name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = (topic_repository.drive_service
                          .files()
                          .create(body=folder_metadata, fields='id')
                          .execute())
                folder_id = folder.get('id')
                logging.info(f"Carpeta creada para el curso {course.name}: {folder_id}")
                # Compartir la carpeta con el usuario
                permission = {
                    'type': 'user',
                    'role': 'writer',
                    'emailAddress': 'orlandokuanb@gmail.com'
                }
                topic_repository.drive_service.permissions().create(
                    fileId=folder_id,
                    body=permission,
                    fields='id',
                ).execute()
                folder_id = folder.get('id')
                logging.info(f"Carpeta creada y compartida para el curso {course.name}: {folder_id}")
            else:
                logging.info(f"Carpeta encontrada para el curso {course.name}: {folder_id}")

            # Crear el curso en la base de datos
            new_course = CourseModel(name=course.name, google_drive_folder_id=folder_id)
            db.add(new_course)
            db.commit()
            db.refresh(new_course)

            return CourseResponse.from_orm(new_course)
        except Exception as e:
            db.rollback()
            logging.error(f"Error al crear el curso: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al crear el curso: {str(e)}")

    def get_all_courses(self, db: Session):
        return [CourseResponse.from_orm(course) for course in db.query(CourseModel).all()]

    async def assign_topic_to_course(self, course_id: str, topic_id: str, db: Session):
        course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
        topic = db.query(TopicModel).filter(TopicModel.id == topic_id).first()

        if not course or not topic:
            raise HTTPException(status_code=404, detail="Course or Topic not found")

        course.topics.append(topic)
        db.commit()
        return {"message": f"Topic {topic.name} assigned to course {course.name}"}

    async def update_course(self, course_id: str, course_update: CourseUpdate, db: Session):
        try:
            course = self.database.get_course_by_id(db, course_id)
            if not course:
                raise HTTPException(status_code=404, detail="Course not found")

            # Actualizar el nombre del curso
            course.name = course_update.name

            # Opcional: Actualizar la carpeta en Google Drive si el nombre cambia
            if course_update.name != course.name:
                topic_repository = TopicRepository(QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY),
                                                   OpenAIEmbeddings())
                folder_metadata = {'name': course_update.name}
                updated_folder = (topic_repository.drive_service
                                  .files()
                                  .update(
                    fileId=course.google_drive_folder_id,
                    body=folder_metadata
                ).execute())
                logging.info(f"Carpeta de Google Drive actualizada: {updated_folder.get('id')}")

            db.commit()
            db.refresh(course)
            return CourseResponse.from_orm(course)
        except HTTPException as e:
            raise e
        except Exception as e:
            logging.error(f"Error al actualizar el curso: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_course(self, course_id: str, db: Session):
        try:
            course = self.database.get_course_by_id(db, course_id)
            if not course:
                raise HTTPException(status_code=404, detail="Course not found")

            # Eliminar la carpeta en Google Drive
            topic_repository = TopicRepository(QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY), OpenAIEmbeddings())
            (topic_repository.drive_service
             .files()
             .delete(fileId=course.google_drive_folder_id)
             .execute())
            logging.info(f"Carpeta de Google Drive eliminada: {course.google_drive_folder_id}")

            # Eliminar el curso de la base de datos
            db.delete(course)
            db.commit()
            return {"message": f"Course '{course.name}' has been deleted successfully."}
        except HTTPException as e:
            raise e
        except Exception as e:
            logging.error(f"Error al eliminar el curso: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    async def update_course_documents(self, course_id: str, db: Session):
        course = db.query(Course).filter(Course.id == course_id).first()
        if not course or not course.google_drive_folder:
            raise ValueError("Course not found or no Google Drive folder associated")

        rag_instance = await RAGSingleton.get_instance()
        await rag_instance.process_google_drive_folder(course.google_drive_folder, course.id, None)

        return {"message": "Course documents updated successfully"}


def transform_description_to_string(description):
    return str(description) if description else ""


class TopicService:
    def __init__(self, database):
        self.database = database

    async def create_topic(self, topic: TopicCreate, db: Session):
        course = db.query(CourseModel).filter(CourseModel.id == topic.course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")

        # Crear un nuevo tópico en la base de datos
        new_topic = TopicModel(id=str(uuid4()), name=topic.name, description=topic.description)
        db.add(new_topic)

        # Asociar el tópico con el curso
        course.topics.append(new_topic)

        db.commit()
        db.refresh(new_topic)
        logging.info(f"New topic created: {new_topic}")

        # Crear un TopicInfo para RAG
        topic_info = TopicInfo(
            topic_id=new_topic.id,
            name=new_topic.name,
            course_id=course.id,
            description=transform_description_to_string(new_topic.description)
        )

        # Añadir el tópico a RAG
        rag_instance = await RAGSingleton.get_instance()
        await rag_instance.add_topic(topic_info)

        return TopicResponse.from_orm(new_topic)

    async def add_document_to_topic(self, doc: DocumentAddToTopic, db: Session):
        topic = db.query(TopicModel).filter(TopicModel.id == doc.topic_id).first()
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        course = topic.course
        if not course:
            raise HTTPException(status_code=404, detail="Course not found for this topic")

        # Crear un documento en la base de datos SQL
        db_document = DocumentModel(
            title=doc.file_name,
            content=doc.content,
            topic_id=topic.id,
            type=doc.metadata.get('type', 'article'),  # Default to 'article' if not specified
            language=doc.metadata.get('language', 'en')  # Default to 'en' if not specified
        )
        db.add(db_document)

        # Crear un documento de Langchain
        langchain_doc = Document(
            page_content=doc.content,
            metadata={**doc.metadata, "file_name": doc.file_name}
        )

        # Añadir el documento al tópico en RAG
        rag_instance = await RAGSingleton.get_instance()
        vector_id = await rag_instance.add_document(course.id, topic.id, langchain_doc)
        # Update the document in SQL database with the vector_id
        db_document.vector_id = vector_id
        db.commit()

        return {"message": f"Document {doc.file_name} added to topic {topic.name}"}

    async def get_topics_for_course(self, course_id: str, db: Session):
        course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")

        rag_instance = await RAGSingleton.get_instance()
        topics = await rag_instance.get_topics_for_course(course_id)

        return [TopicResponse(id=topic.topic_id, name=topic.name, description=topic.description) for topic in topics]

    async def update_topic_documents(self, topic_id: str, db: Session):
        topic = db.query(TopicModel).filter(TopicModel.id == topic_id).first()
        if not topic or not topic.google_drive_folder:
            raise ValueError("Topic not found or no Google Drive folder associated")

        rag_instance = await RAGSingleton.get_instance()
        await rag_instance.process_google_drive_folder(topic.google_drive_folder, topic.course_id, topic.id)

        return {"message": "Topic documents updated successfully"}

    async def update_topic(self, topic_id: str, updated_topic: TopicCreate, db: Session):
        topic = db.query(Topic).filter(Topic.id == topic_id).first()
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        logging.info(f"Updating topic {topic_id} with new data: {updated_topic}")
        topic.name = updated_topic.name
        topic.description = updated_topic.description
        topic.course_id = updated_topic.course_id  # Asegúrate de actualizar el course_id si es necesario
        db.commit()

        # Actualizar el tópico en RAG
        # rag_instance = await RAGSingleton.get_instance()
        # topic_info = TopicInfo(
        #     topic_id=topic.id,
        #     name=topic.name,
        #     course_id=topic.course_id,
        #     description=topic.description
        # )
        # await rag_instance.add_topic(topic_info)  # Esto actualizará el tópico existente

        return TopicResponse.from_orm(topic)

    def update_topic_sync(self, topic_id: str, updated_topic: TopicCreate, db: Session):
        topic = db.query(Topic).filter(Topic.id == topic_id).first()
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topic.name = updated_topic.name
        topic.description = updated_topic.description
        topic.course_id = updated_topic.course_id
        db.commit()
        return TopicResponse.from_orm(topic)

    def get_topic_by_id(self, topic_id: str, db: Session):
        return db.query(Topic).filter(Topic.id == topic_id).first()

    async def delete_topic(self, topic_id: str, db: Session):
        topic = db.query(TopicModel).filter(TopicModel.id == topic_id).first()
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        # Eliminar el tópico de la base de datos SQL
        db.delete(topic)
        db.commit()

        # Eliminar el tópico y sus documentos asociados de RAG
        rag_instance = await RAGSingleton.get_instance()
        await rag_instance.delete_topic(topic_id)

        return {"message": f"Topic {topic.name} deleted successfully"}

    async def search_documents_service(self, query: str, course_id: str, topic_id: str = None, k: int = 5):
        rag_instance = await RAGSingleton.get_instance()
        documents = await rag_instance.search_documents(query, course_id, topic_id, k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in documents
        ]


class QuestionService:
    def __init__(self, database, checkpointer: PostgresSaver, store: PostgresStore):
        self.database = database
        #self.llm = ChatOpenAI(model_name="gpt-4o-mini", client=openai_client)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.llm_judge = ChatOpenAI(model="gpt-4o")
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=TOPIC_COLLECTION,
            embedding=self.embeddings
        )
        self.topic_repository = TopicRepository(self.qdrant_client, self.embeddings)
        self.topic_service = TopicService(database)
        retriever_config = CustomQdrantRetrieverConfig(
            client=self.qdrant_client,
            collection_name=TOPIC_COLLECTION,
            embeddings=self.embeddings,
            k=5
        )
        self.retriever = CustomQdrantRetriever(config=retriever_config)
        self.checkpointer = checkpointer
        self.store = store
        # Instanciar TavilySearchResults
        self.web_search_tool = TavilySearchResults(k=3)

    # async def retrieve(self, state: State) -> Dict[str, Any]:
    #     question = state["input"]
    #     course_id = state["course_id"]
    #     filters = Filter(
    #         must=[
    #             FieldCondition(key="course_id", match=MatchValue(value=course_id))
    #         ]
    #     )
    #     # Recuperar documentos relevantes de manera asíncrona
    #     relevant_docs = await self.retriever.ainvoke(question, filters=filters)
    #     logging.info(f"Retrieved {len(relevant_docs)} relevant documents")
    #     for doc in relevant_docs:
    #         logging.info(
    #             f"Document ID: {doc.metadata.get('id')}, "
    #             f"Course ID: {doc.metadata.get('course_id')}, "
    #             f"Topic ID: {doc.metadata.get('topic_id')}, "
    #             f"Score: {doc.metadata.get('score')}, "
    #             f"Content preview: {doc.page_content[:100]}..."
    #         )
    #     document_list_retriever = DocumentListRetriever(relevant_docs)
    #
    #     return {
    #         "input": question,
    #         "chat_history": state["chat_history"],
    #         "documents": document_list_retriever,
    #         "web_search": "No"
    #     }
    def retrieve(self, state: State) -> Dict[str, Any]:
        question = state["input"]
        course_id = state.get("course_id")
        # Definir filtros si course_id está presente
        filters = None
        if course_id:
            filters = Filter(
                must=[
                    FieldCondition(key="course_id", match=MatchValue(value=course_id))
                ]
            )
        logging.debug(f"Aplicando filtros: course_id={course_id}")

        relevant_docs = self.retriever.get_relevant_documents(question, filters=filters)
        return {
            "input": question,
            "chat_history": state["chat_history"],
            "documents": relevant_docs,
            "web_search": "No"
        }

    def grade_documents(self, state: State) -> Dict[str, Any]:
        question = state["input"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un evaluador que determina si un documento es relevante para una pregunta."),
            ("human",
             "Pregunta: {question}\n\nDocumento: {document}\n\n¿Es este documento relevante? Responde 'Sí' o 'No'.")
        ])

        for doc in documents:
            grade_chain = grade_prompt | self.llm_judge | StrOutputParser()
            grade = grade_chain.invoke({"question": question, "document": doc.page_content})
            if "sí" in grade.lower():
                filtered_docs.append(doc)
            else:
                web_search = "Yes"

        return {
            "input": question,
            "chat_history": state["chat_history"],
            "documents": filtered_docs,
            "web_search": web_search
        }

    def decide_to_generate(self, state: State) -> str:
        if state.get("web_search") == "Yes":
            return "transform_query"
        else:
            return "generate"

    def transform_query(self, state: State) -> Dict[str, Any]:
        question = state["input"]

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un asistente que reformula preguntas para optimizar búsquedas web."),
            ("human", "Pregunta original: {question}\n\nReformula esta pregunta para optimizarla para la búsqueda web:")
        ])

        rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
        new_question = rewrite_chain.invoke({"question": question})

        return {
            "input": new_question,
            "chat_history": state["chat_history"],
            "documents": state["documents"],
            "web_search": state["web_search"]
        }

    def perform_web_search(self, state: State) -> Dict[str, Any]:
        question = state["input"]
        search_results = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in search_results])
        web_document = Document(page_content=web_results)
        documents = state["documents"] + [web_document]

        return {
            "input": question,
            "chat_history": state["chat_history"],
            "documents": documents,
            "web_search": "No"
        }

    def check_ambiguity(self, state: State) -> State:
        question = state["input"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un sistema que determina si la pregunta del usuario es ambigua."),
            ("human", f"La pregunta del usuario es: '{question}'. "
                      f"Responde 'AMBIGUO' si es ambigua y 'CLARO' si está clara.")
        ])
        result = (prompt | self.llm_judge | StrOutputParser()).invoke({})

        if "AMBIGUO" in result.upper():
            # Si es ambigua, lanzamos interrupción
            raise NodeInterrupt("La pregunta es ambigua, se requiere feedback humano.")
        # Si no es ambigua, seguimos normal
        return state

    def merge_feedback(self, state: State) -> State:
        # Si user_feedback no está vacío, lo fusionamos con el input original
        # Podrías definir la lógica de fusión:
        # Por ejemplo, simplemente:
        if state["user_feedback"]:
            # Suponiendo que la retroalimentación sea una explicación de cómo aclarar la pregunta,
            # Podríamos sobrescribir el input con la retroalimentación directamente, o combinarlos.
            # Aquí un ejemplo simple:
            # "El usuario originalmente preguntó X, feedback: Y. Nueva pregunta: Y"
            # Pero para simplificar, digamos que user_feedback es ya la versión clara de la pregunta.
            state["input"] = state["user_feedback"]
            # Opcional: puedes limpiar user_feedback después
            # state["user_feedback"] = ""
        return state

    def generate(self, state: State) -> Dict[str, Any]:
        question = state["input"]
        documents = state["documents"]

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Eres un asistente para tareas de preguntas y respuestas. Usa los siguientes documentos para responder la pregunta. Si no sabes la respuesta, indica que no lo sabes. Usa tres oraciones como máximo y mantén la respuesta concisa.\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(DocumentListRetriever(documents), qa_chain)

        response = rag_chain.invoke({
            "input": question,
            "chat_history": state["chat_history"],
            "context": "\n\n".join([doc.page_content for doc in documents])
        })

        return {
            "input": question,
            "chat_history": state["chat_history"] + [HumanMessage(content=question),
                                                     AIMessage(content=response["answer"])],
            "context": response["context"],
            "answer": response["answer"]
        }

    #@traceable(run_type="chain")
    async def process_question(self, question: QuestionV2, db: Session):
        try:
            chat_session = db.query(ChatSession).filter(ChatSession.id == question.chat_session_id).first()
            if not chat_session:
                raise HTTPException(status_code=404, detail="Chat session not found")

            # Usar los resultados del retriever personalizado para generar la respuesta
            response = await self.generate_response_agent(question.text, chat_session)

            # Guardar la pregunta en la base de datos
            db_question = QuestionModel(
                text=question.text,
                chat_session_id=chat_session.id,
                user_id=chat_session.user_id,
                course_id=chat_session.course_id,
                topic_id=chat_session.topic_id,
                answer=response  # Guardar la respuesta del bot
            )
            db.add(db_question)
            db.commit()

            return {
                "response": response,
                "user": chat_session.user.name,
                "course": chat_session.course.name if chat_session.course else None,
                "topic": chat_session.topic.name if chat_session.topic else None
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def generate_response_agent(self, question: str, chat_session: ChatSession) -> str:
        try:
            logging.info(f"Generando respuesta para la pregunta: {question}")

            # Definir el flujo de LangGraph con los nuevos nodos
            workflow = StateGraph(state_schema=State)
            # Añadir los nodos
            workflow.add_node("check_ambiguity", self.check_ambiguity)
            workflow.add_node("merge_feedback", self.merge_feedback)
            workflow.add_node("retrieve", self.retrieve)
            workflow.add_node("grade_documents", self.grade_documents)
            workflow.add_node("transform_query", self.transform_query)
            workflow.add_node("perform_web_search", self.perform_web_search)  # Nodo renombrado
            workflow.add_node("generate", self.generate)

            # Definir las transiciones
            workflow.add_edge(START, "check_ambiguity")
            workflow.add_edge("check_ambiguity", "merge_feedback")
            workflow.add_edge("merge_feedback", "retrieve")
            workflow.add_edge("retrieve", "grade_documents")
            workflow.add_conditional_edges(
                "grade_documents",
                self.decide_to_generate,
                {
                    "transform_query": "transform_query",
                    "generate": "generate",
                }
            )
            workflow.add_edge("transform_query", "perform_web_search")
            workflow.add_edge("perform_web_search", "generate")
            workflow.add_edge("generate", END)

            # Compilar el grafo con el checkpointer
            app = workflow.compile(checkpointer=self.checkpointer)
            # Graficar el flijo
            display(Image(app.get_graph().draw_mermaid_png()))
            # Obtener el historial de chat
            chat_history = get_session_history(chat_session.id).messages

            # Estado inicial
            state = {
                "input": question,
                "chat_history": chat_history,
                "context": "",
                "answer": "",
                "documents": [],
                "web_search": "No",
                "course_id": chat_session.course_id,
                "user_feedback": ""
            }
            config = {
                "configurable": {
                    "thread_id": chat_session.id
                }
            }
            # Ejecutar el grafo
            result = app.invoke(state, config)

            return result['answer']
        except NodeInterrupt as e:
            # Aquí se detectó ambigüedad
            # Detenemos y pedimos al usuario aclaración
            # Luego de obtener clarificación:
            # graph.update_state(config, {"user_feedback": "Pregunta ya aclarada"}, as_node="human_input")
            # Y continuar:
            # result = app.invoke(None, config)
            #
            # Pero esto lo harías en tu lógica externa
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logging.error(f"Error durante la generación de respuesta: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error generating response")

    async def start_chat_session(self, session_start: ChatSessionStart, db: Session):
        try:
            # Iniciar transacción
            db.begin_nested()  # Crear un savepoint

            # Verificar usuario
            user = db.query(UserModel).filter(UserModel.id == session_start.user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            # Crear tópico temporal
            temp_topic = TopicCreate(
                id=str(uuid4()),
                name=f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                description=session_start.initial_question,
                course_id=session_start.course_id
            )

            try:
                created_topic = await self.topic_service.create_topic(temp_topic, db)
            except Exception as e:
                db.rollback()
                logging.error(f"Error creating topic: {str(e)}")
                raise HTTPException(status_code=500, detail="Error creating chat topic")

            # Crear sesión de chat
            try:
                chat_session = self.database.create_chat_session(
                    db,
                    user_id=session_start.user_id,
                    course_id=session_start.course_id,
                    topic_id=created_topic.id
                )
            except Exception as e:
                db.rollback()
                logging.error(f"Error creating chat session: {str(e)}")
                raise HTTPException(status_code=500, detail="Error creating chat session")

            # Procesar pregunta inicial
            try:
                initial_question = QuestionV2(
                    text=session_start.initial_question,
                    user_id=session_start.user_id,
                    chat_session_id=chat_session.id
                )
                answer = await self.process_question(initial_question, db)
            except Exception as e:
                db.rollback()
                logging.error(f"Error processing initial question: {str(e)}")
                raise HTTPException(status_code=500, detail="Error processing initial question")

            # Iniciar tarea de generación de título
            try:
                from app.event.tasks import generate_and_update_title
                logging.info(f"Sending generate_and_update_title task for topic_id: {created_topic.id}")
                task = generate_and_update_title.delay(created_topic.id, session_start.initial_question)
            except Exception as e:
                logging.error(f"Error starting title generation task: {str(e)}")
                # No hacemos rollback aquí porque la generación del título es una tarea secundaria
                task = None

            # Si todo fue exitoso, hacer commit de la transacción
            db.commit()

            return {
                "chat_session_id": chat_session.id,
                "topic_id": created_topic.id,
                "topic_title": created_topic.name,
                "initial_answer": answer,
                "title_task_id": task.id if task else None
            }

        except HTTPException as he:
            # Propagar excepciones HTTP
            raise he
        except Exception as e:
            # Rollback en caso de cualquier otro error
            db.rollback()
            logging.error(f"Unexpected error in start_chat_session: {str(e)}")
            raise HTTPException(status_code=500, detail="Error starting chat session")
        finally:
            # Asegurarse de que la sesión está limpia
            db.close()

    async def get_title_task_status(self, task_id: str):
        from app.event.tasks import generate_and_update_title
        task = generate_and_update_title.AsyncResult(task_id)
        if task.state == 'PENDING':
            return {'state': task.state, 'status': 'Task is pending...'}
        elif task.state != 'FAILURE':
            return {
                'state': task.state,
                'status': 'Task completed' if task.state == 'SUCCESS' else 'Task is in progress',
                'result': task.result
            }
        else:
            return {'state': task.state, 'status': 'Task failed', 'error': str(task.info)}

    async def end_chat_session(self, chat_session_id: str, db: Session):
        chat_session = self.database.end_chat_session(db, chat_session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return {"message": "Chat session ended successfully"}

    async def submit_feedback(self, feedback: Feedback, db: Session):
        try:
            db_feedback = self.database.create_feedback(db, feedback)
            logging.info(f"Feedback received for chat session: {feedback.chat_session_id}")
            return {"message": "Feedback received successfully"}
        except Exception as e:
            logging.error(f"Error submitting feedback: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_chat_list(self, user_id: str, course_id: str, db: Session) -> ChatListResponse:
        # Obtener todas las sesiones de chat para el usuario y curso
        logging.info(f"Obteniendo lista de chats para usuario {user_id} y curso {course_id}")
        chat_sessions = db.query(ChatSession).filter(
            ChatSession.user_id == user_id,
            ChatSession.course_id == course_id
        ).order_by(ChatSession.start_time.desc()).all()

        chat_list_items = []
        for session in chat_sessions:
            # Obtener el tópico asociado a la sesión
            topic = db.query(Topic).filter(Topic.id == session.topic_id).first()

            chat_list_items.append(ChatListItem(
                id=session.id,
                topic_title=topic.name if topic else "Unknown Topic",
                timestamp=session.start_time
            ))

        return ChatListResponse(
            user_id=user_id,
            course_id=course_id,
            chats=chat_list_items
        )

    async def get_chat_history(self, chat_id: str, db: Session):
        logging.info(f"Obteniendo historial de chat para sesión {chat_id}")
        chat_session = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Obtener todas las preguntas y respuestas asociadas a esta sesión de chat
        questions = db.query(QuestionModel).filter(QuestionModel.chat_session_id == chat_id).order_by(
            QuestionModel.created_at).all()

        history = []
        for question in questions:
            history.append({"type": "user", "content": question.text})
            if question.answer:
                history.append({"type": "bot", "content": question.answer})

        return history
