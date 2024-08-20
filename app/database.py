from uuid import uuid4

from sqlalchemy import create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

from app.model import User, Course, Topic, Question, Feedback, ChatSession
import logging

logging.basicConfig(level=logging.DEBUG)

# Importar aquí los modelos para evitar problemas de dependencia circular

# Configuration
# load_dotenv()
# Determinar el entorno actual
environment = os.getenv("ENVIRONMENT", "development")
# Cargar el archivo .env.prod adecuado
if environment == "production":
    load_dotenv(".env.prod")
else:
    load_dotenv(".env")

POSTGRES_HOST = os.getenv("DB_HOST", "localhost")
POSTGRES_PORT = os.getenv("DB_PORT", "5432")
POSTGRES_DB = os.getenv("DB_NAME")
POSTGRES_USER = os.getenv("DB_USER")
POSTGRES_PASSWORD = os.getenv("DB_PASSWORD")
# URL de la base de datos
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
print(SQLALCHEMY_DATABASE_URL)
# Configuración del engine y creación de la sesión
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    from .base import Base
    Base.metadata.create_all(bind=engine)

class Database:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def create_user(self, db: Session, user: User):
        db_user = User(id=user.id, name=user.name, session_id=user.session_id)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    def get_user_by_id(self, db: Session, user_id: str):
        return db.query(User).filter(User.id == user_id).first()

    def get_user_by_name(self, db: Session, name: str):
        return db.query(User).filter(User.name == name).first()

    def create_course(self, db: Session, course: Course):
        db_course = Course(id=course.id, name=course.name)
        db.add(db_course)
        db.commit()
        db.refresh(db_course)
        return db_course

    def get_course_by_id(self, db: Session, course_id: str):
        return db.query(Course).filter(Course.id == course_id).first()

    def create_topic(self, db: Session, topic: Topic):

        db_topic = Topic(id=topic.id, name=topic.name, course_id=topic.course_id)
        db.add(db_topic)
        db.commit()
        db.refresh(db_topic)
        return db_topic

    def get_topic_by_id(self, db: Session, topic_id: str):
        return db.query(Topic).filter(Topic.id == topic_id).first()

    def assign_course_to_user(self, db: Session, user_id: str, course_id: str):
        user = self.get_user_by_id(db, user_id)
        course = self.get_course_by_id(db, course_id)
        if user and course:
            user.courses.append(course)
            db.commit()
            return True
        return False

    def assign_topic_to_course(self, db: Session, course_id: str, topic_id: str):
        course = self.get_course_by_id(db, course_id)
        topic = self.get_topic_by_id(db, topic_id)
        if course and topic:
            course.topics.append(topic)
            db.commit()
            return True
        return False

    def create_feedback(self, db: Session, feedback: Feedback):
        db_feedback = Feedback(id=feedback.id, run_id=feedback.run_id, score=feedback.score)
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        return db_feedback

    def create_chat_session(self, db: Session, user_id: str, course_id: str = None, topic_id: str = None):
        chat_session = ChatSession(
            id=str(uuid4()),
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_id,
            status="active"
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        return chat_session

    def get_chat_session(self, db: Session, chat_session_id: str):
        return db.query(ChatSession).filter(ChatSession.id == chat_session_id).first()

    def end_chat_session(self, db: Session, chat_session_id: str):
        chat_session = self.get_chat_session(db, chat_session_id)
        if chat_session:
            chat_session.status = "completed"
            chat_session.end_time = func.now()
            db.commit()
            return chat_session
        return None

    def create_question(self, db: Session, question: Question):
        db_question = Question(
            id=question.id,
            text=question.text,
            chat_session_id=question.chat_session_id
        )
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        return db_question