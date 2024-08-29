# app/services.py
import os
from pathlib import Path
from typing import List, Any
from uuid import uuid4
from fastapi import HTTPException
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field

from app.collections import TopicRepository
from app.rag import RAG, TopicInfo
import logging
import asyncio
from app.model import User as UserModel, Course as CourseModel, Topic as TopicModel, Question as QuestionModel, \
    ChatSession, Document as DocumentModel, Course, Topic
from app.retriever.custom_qdrant_retriever import CustomQdrantRetriever, CustomQdrantRetrieverConfig
from app.retriever.document_list_retriever import DocumentListRetriever
from app.schema.schema import UserLogin, UserResponse, DocumentCreate, CourseAssignment, CourseCreate, CourseResponse, \
    TopicCreate, TopicResponse, Question, Feedback, QuestionV2, DocumentAddToTopic, ChatSessionStart, ChatListResponse, \
    ChatListItem
from sqlalchemy.orm import Session
from app.model import User as UserModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langsmith import traceable
from langchain_qdrant import QdrantVectorStore
from sqlalchemy import func
from qdrant_client import QdrantClient, models
from app.historial import QdrantChatMessageHistory
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
import openai
from summa import keywords
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

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

    async def login_user(self, user: UserLogin, db: Session):
        db_user = self.database.get_user_by_name(db, user.name)
        if db_user is None:
            db_user = UserModel(
                id=str(uuid4()),
                name=user.name,
                session_id=str(uuid4())
            )
            db_user = self.database.create_user(db, db_user)
        else:
            db_user.session_id = str(uuid4())
            db.commit()

        user_courses = [course.name for course in db_user.courses]
        #course_collections = [course.collection_name for course in db_user.courses]

        return UserResponse(
            id=db_user.id,
            name=db_user.name,
            session_id=db_user.session_id,
            courses=user_courses,
            #course_collections=course_collections
        )

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


class CourseService:
    def __init__(self, database):
        self.database = database

    async def create_course(self, course: CourseCreate, db: Session):
        new_course = CourseModel(name=course.name)
        db.add(new_course)
        db.commit()
        db.refresh(new_course)
        return CourseResponse.from_orm(new_course)

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
        topic = db.query(TopicModel).filter(TopicModel.id == topic_id).first()
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        topic.name = updated_topic.name
        topic.description = updated_topic.description
        db.commit()

        # Actualizar el tópico en RAG
        rag_instance = await RAGSingleton.get_instance()
        topic_info = TopicInfo(
            topic_id=topic.id,
            name=topic.name,
            course_id=topic.course_id,
            description=topic.description
        )
        await rag_instance.add_topic(topic_info)  # Esto actualizará el tópico existente

        return TopicResponse.from_orm(topic)

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


def generate_topic_title(question: str) -> str:
    key_phrases = keywords.keywords(question, words=5).split('\n')
    title = " ".join(key_phrases).title()
    return title


class QuestionService:
    def __init__(self, database):
        self.database = database
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", client=openai_client)
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

    @traceable(run_type="chain")
    async def process_question(self, question: QuestionV2, db: Session):
        try:
            chat_session = db.query(ChatSession).filter(ChatSession.id == question.chat_session_id).first()
            if not chat_session:
                raise HTTPException(status_code=404, detail="Chat session not found")

            # Crear un retriever específico para este curso y tema
            #relevant_docs = self.retriever.get_relevant_documents(chat_session.course_id, chat_session.topic_id)
            relevant_docs = await self.retriever.ainvoke(question.text)
            logging.info(f"Retrieved {len(relevant_docs)} relevant documents")
            for doc in relevant_docs:
                logging.info(
                    f"Document ID: {doc.metadata['id']}, Score: {doc.metadata['score']}, Content preview: {doc.page_content[:100]}...")
            # Crear un DocumentListRetriever con los documentos relevantes
            document_list_retriever = DocumentListRetriever(relevant_docs)

            # Búsqueda con el retriever personalizado
            # custom_results = await retriever.aget_relevant_documents(question.text)
            # logging.info(f"Custom retriever results: {custom_results}")

            # Búsqueda directa con Qdrant
            # qdrant_results = self.qdrant_client.search(
            #     collection_name=TOPIC_COLLECTION,
            #     query_vector=self.embeddings.embed_query(question.text),
            #     query_filter=models.Filter(
            #         must=[
            #             models.FieldCondition(key="type", match=models.MatchValue(value="document")),
            #             models.FieldCondition(key="course_id", match=models.MatchValue(value=chat_session.course_id)),
            #             models.FieldCondition(key="topic_id", match=models.MatchValue(value=chat_session.topic_id))
            #         ]
            #     ),
            #     limit=5
            # )
            # logging.info(f"Qdrant search results: {qdrant_results}")

            # Comparar resultados
            # custom_ids = set(doc.metadata.get('id') for doc in custom_results)
            # qdrant_ids = set(result.id for result in qdrant_results)
            # common_ids = custom_ids.intersection(qdrant_ids)
            # logging.info(f"Common document IDs: {common_ids}")
            # logging.info(f"Documents only in custom results: {custom_ids - qdrant_ids}")
            # logging.info(f"Documents only in Qdrant results: {qdrant_ids - custom_ids}")

            # Usar los resultados del retriever personalizado para generar la respuesta
            response = await self.generate_response(question.text, chat_session, document_list_retriever)

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

    @traceable(run_type="retriever")
    def create_filtered_retriever(self, course_id: str, topic_id: str) -> VectorStoreRetriever:
        filter_condition = {"must": [{"key": "type", "match": {"value": "document"}}]}

        if course_id:
            filter_condition["must"].append({"key": "course_id", "match": {"value": course_id}})
        if topic_id:
            filter_condition["must"].append({"key": "topic_id", "match": {"value": topic_id}})

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": filter_condition,
            }
        )

    async def sync_documents(self, course_id: str, topic_id: str):
        try:
            rag_instance = await RAGSingleton.get_instance()

            if GOOGLE_DRIVE_FOLDER_ID:
                success = await rag_instance.process_google_drive_folder(GOOGLE_DRIVE_FOLDER_ID, course_id, topic_id)
                if success:
                    logging.info(f"Documentos actualizados para curso {course_id} y tema {topic_id}")
                else:
                    logging.warning(f"No se pudieron actualizar documentos para curso {course_id} y tema {topic_id}")
            else:
                logging.warning("GOOGLE_DRIVE_FOLDER_ID no está configurado. No se sincronizaron documentos.")

        except Exception as e:
            logging.error(f"Error al sincronizar documentos: {str(e)}")

    @traceable(metadata={"model": "gpt-4o-mini"})
    async def generate_response(self, question: str, chat_session: ChatSession, retriever: BaseRetriever):
        try:
            logging.info(f"Generando respuesta para la pregunta: {question}")

            # Crear el prompt para contextualizar la pregunta
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question "
                           "which might reference context in the chat history, "
                           "formulate a standalone question which can be understood "
                           "without the chat history. Do NOT answer the question, "
                           "just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            # Crear un retriever consciente del historial
            history_aware_retriever = create_history_aware_retriever(
                self.llm,
                retriever,
                contextualize_q_prompt
            )

            # Crear el prompt para la cadena de preguntas y respuestas
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant for question-answering tasks. "
                           "Use the following pieces of retrieved context to answer "
                           "the question. If you don't know the answer, say that you "
                           "don't know. Use three sentences maximum and keep the "
                           "answer concise.\n\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            # Crear la cadena de documentos
            qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)

            # Combinar el retriever y la cadena de qa
            rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

            # Usar RunnableWithMessageHistory
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            # Invocar la cadena
            response = await conversational_rag_chain.ainvoke(
                {"input": question},
                config={"configurable": {"session_id": chat_session.id}}
            )
            # Obtener los documentos recuperados
            retrieved_docs = response.get('context', [])

            # Logging de los documentos recuperados
            for i, doc in enumerate(retrieved_docs):
                logging.info(f"Documento {i + 1}:")
                logging.info(f"  Contenido (primeros 100 caracteres): {doc.page_content[:100]}")
                logging.info(f"  Metadata: {doc.metadata}")

            logging.info(f"Respuesta generada: {response['answer']}")

            return response['answer']
        except Exception as e:
            logging.error(f"Error durante la generación de respuesta: {e}", exc_info=True)
            raise

    async def start_chat_session(self, session_start: ChatSessionStart, db: Session):
        user = db.query(UserModel).filter(UserModel.id == session_start.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Generar título del tópico basado en la pregunta inicial
        topic_title = generate_topic_title(session_start.initial_question)

        # Crear un nuevo tópico
        new_topic = TopicCreate(
            name=topic_title,
            description=session_start.initial_question,
            course_id=session_start.course_id
        )
        created_topic = await self.topic_service.create_topic(new_topic, db)

        chat_session = self.database.create_chat_session(
            db,
            user_id=session_start.user_id,
            course_id=session_start.course_id,
            topic_id=created_topic.id
        )

        # Sincronizar documentos al inicio de la sesión
        await self.sync_documents(session_start.course_id, created_topic.id)

        # Procesar la pregunta inicial
        question = QuestionV2(
            text=session_start.initial_question,
            user_id=session_start.user_id,
            chat_session_id=chat_session.id
        )
        answer = await self.process_question(question, db)

        # Guardar la pregunta inicial en la base de datos
        db_question = QuestionModel(
            id=str(uuid4()),
            text=session_start.initial_question,
            user_id=session_start.user_id,
            chat_session_id=chat_session.id,
            course_id=session_start.course_id,
            topic_id=created_topic.id
        )
        db.add(db_question)
        db.commit()

        return {
            "chat_session_id": chat_session.id,
            "topic_id": created_topic.id,
            "topic_title": topic_title,
            "initial_answer": answer
        }

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
