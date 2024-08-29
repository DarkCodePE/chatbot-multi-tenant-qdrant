from sqlalchemy import Column, String, DateTime, Table, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from uuid import uuid4
from .base import Base

# Tablas de asociación existentes
user_course = Table('user_course', Base.metadata,
                    Column('user_id', String, ForeignKey('users.id')),
                    Column('course_id', String, ForeignKey('courses.id'))
                    )

course_topic = Table('course_topic', Base.metadata,
                     Column('course_id', String, ForeignKey('courses.id')),
                     Column('topic_id', String, ForeignKey('topics.id'))
                     )

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    name = Column(String, index=True)
    group_id = Column(String, index=True)
    session_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    chat_status = Column(String)  # Nuevo campo para indicar el estado de chat
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    courses = relationship("Course", secondary=user_course, back_populates="users")
    questions = relationship("Question", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, group_id={self.group_id})>"

class Course(Base):
    __tablename__ = "courses"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    name = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    users = relationship("User", secondary=user_course, back_populates="courses")
    topics = relationship("Topic", back_populates="course")
    questions = relationship("Question", back_populates="course")
    chat_sessions = relationship("ChatSession", back_populates="course")

    def __repr__(self):
        return f"<Course(id={self.id}, name={self.name})>"

class Topic(Base):
    __tablename__ = "topics"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    name = Column(String, index=True)
    description = Column(Text)
    vector_id = Column(String, unique=True)  # ID in Qdrant
    course_id = Column(String, ForeignKey('courses.id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    course = relationship("Course", back_populates="topics")
    documents = relationship("Document", back_populates="topic")
    questions = relationship("Question", back_populates="topic")
    chat_sessions = relationship("ChatSession", back_populates="topic")

    def __repr__(self):
        return f"<Topic(id={self.id}, name={self.name})>"

class Question(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    text = Column(String)
    answer = Column(String)  # Nuevo campo para la respuesta
    user_id = Column(String, ForeignKey('users.id'), index=True)
    course_id = Column(String, ForeignKey('courses.id'), index=True)
    topic_id = Column(String, ForeignKey('topics.id'), index=True)
    chat_session_id = Column(String, ForeignKey('chat_sessions.id'), index=True)  # Nuevo campo
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="questions")
    course = relationship("Course", back_populates="questions")
    topic = relationship("Topic", back_populates="questions")
    chat_session = relationship("ChatSession", back_populates="questions")

    def __repr__(self):
        return f"<Question(id={self.id}, user_id={self.user_id}, course_id={self.course_id}, topic_id={self.topic_id})>"

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    title = Column(String, index=True)
    content = Column(Text)
    vector_id = Column(String, unique=True)  # ID in Qdrant
    topic_id = Column(String, ForeignKey('topics.id'))
    type = Column(String)  # Nuevo campo: 'article', 'exercise', 'quiz', etc.
    language = Column(String)  # Nuevo campo
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    topic = relationship("Topic", back_populates="documents")

    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title}, type={self.type})>"

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(String, ForeignKey('users.id'), index=True)
    course_id = Column(String, ForeignKey('courses.id'), index=True)
    topic_id = Column(String, ForeignKey('topics.id'), index=True)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status = Column(String)  # 'active', 'completed', etc.

    user = relationship("User", back_populates="chat_sessions")
    course = relationship("Course", back_populates="chat_sessions")
    topic = relationship("Topic", back_populates="chat_sessions")
    questions = relationship("Question", back_populates="chat_session")
    feedbacks = relationship("Feedback", back_populates="chat_session")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, status={self.status})>"

class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    chat_session_id = Column(String, ForeignKey('chat_sessions.id'), index=True)
    score = Column(String)
    comment = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chat_session = relationship("ChatSession", back_populates="feedbacks")

    def __repr__(self):
        return f"<Feedback(id={self.id}, chat_session_id={self.chat_session_id}, score={self.score})>"
