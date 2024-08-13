from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from uuid import uuid4
from .base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    name = Column(String, index=True)
    group_id = Column(String, index=True)
    session_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, group_id={self.group_id})>"

class Question(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    text = Column(String)
    user_id = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Question(id={self.id}, user_id={self.user_id})>"

class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    run_id = Column(String, index=True)
    score = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Feedback(id={self.id}, run_id={self.run_id}, score={self.score})>"
