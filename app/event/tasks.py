# tasks.py
from app.celery_app import app
from langchain_core.pydantic_v1 import BaseModel, Field

from app.generator.rag import RAG
from app.schema.schema import TopicCreate
from app.services.services import TopicService, QuestionService
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from app.model import User as UserModel
from langchain_core.output_parsers import StrOutputParser
from app.database import SessionLocal
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Wrapping OpenAI client
openai_client = openai.Client()


# Define la estructura de salida esperada
class Title(BaseModel):
    title: str = Field(description="Un título conciso de 5 palabras o menos basado en la pregunta")


def generate_topic_title(question: str) -> str:
    logger.info(f"Generating title for question: {question}")
    try:
        # Inicializar el modelo de lenguaje
        model = ChatOpenAI(model="gpt-4o-mini")

        # Crear el prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Eres un asistente que genera títulos concisos."),
            ("human", "Generate a short and concise title (5 words or less) based on this question: {question}")
        ])

        # Crear el parser de salida
        parser = StrOutputParser()

        # Crear la cadena completa
        chain = prompt_template | model | parser

        # Invocar la cadena
        title = chain.invoke({"question": question})

        logger.info(f"Generated title: {title}")
        return title
    except Exception as e:
        logger.exception(f"Error generating title: {str(e)}")
        raise


@app.task(bind=True, name='app.event.tasks.generate_and_update_title',
          soft_time_limit=300, time_limit=600)
def generate_and_update_title(self, topic_id: str, question: str):
    logger.info(f"Starting task for topic_id: {topic_id}")
    self.update_state(state='PROGRESS', meta={'status': 'Generating title'})

    db = SessionLocal()
    try:
        logger.info("Database connection established")
        topic_service = TopicService(db)

        new_title = generate_topic_title(question)
        logger.info(f"Generated new title: {new_title}")

        # Obtener el tema existente
        existing_topic = topic_service.get_topic_by_id(topic_id, db)
        if not existing_topic:
            raise ValueError(f"Topic with id {topic_id} not found")

        # Actualizar el tema con el nuevo título y el course_id existente
        updated_topic = TopicCreate(
            name=new_title,
            course_id=existing_topic.course_id,
            description=existing_topic.description
        )
        topic_service.update_topic_sync(topic_id, updated_topic, db)
        logger.info(f"Updated topic {topic_id} with new title: {new_title}")

        return {'status': 'success', 'new_title': new_title}
    except Exception as exc:
        logger.exception(f"Unexpected error in generate_and_update_title: {str(exc)}")
        self.update_state(state='FAILURE', meta={'error': str(exc)})
        raise exc
    finally:
        db.close()
        logger.info("Database connection closed")


@app.task(name='app.event.tasks.sync_user_documents')
def sync_user_documents(user_id: str):
    logger.info(f"Starting document synchronization for user: {user_id}")
    db = SessionLocal()
    try:
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        # Inicializar una nueva instancia de RAG para esta tarea
        rag = RAG()
        rag.initialize()  # Asumiendo que initialize() es síncrono, si no, usar run_until_complete
        for course in user.courses:
            if course.google_drive_folder:
                logger.info(f"Synchronizing documents for course: {course.id}")
                rag.process_google_drive_folder(course.google_drive_folder_id, course.id, None)

        logger.info(f"Document synchronization completed for user: {user_id}")
        return {"status": "success", "message": "Documents synchronized successfully"}
    except Exception as exc:
        logger.exception(f"Error synchronizing documents for user {user_id}: {str(exc)}")
        raise exc
    finally:
        db.close()
