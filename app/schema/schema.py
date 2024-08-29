# app/schema.py
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UserLogin(BaseModel):
    name: str
    group_id: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    name: str
    session_id: str
    group_id: Optional[str] = None


class ChatSessionStart(BaseModel):
    user_id: str
    course_id: Optional[str] = None
    initial_question: Optional[str] = None


class ChatSessionEnd(BaseModel):
    chat_session_id: str


class QuestionV2(BaseModel):
    chat_session_id: str
    text: str


class FeedbackCreate(BaseModel):
    chat_session_id: str
    score: str
    comment: Optional[str] = None


class CourseCreate(BaseModel):
    name: str


class TopicCreate(BaseModel):
    name: str
    description: Optional[str] = None
    course_id: str


class TopicResponse(BaseModel):
    id: str
    name: str

    model_config = ConfigDict(from_attributes=True)


class TopicInfo(BaseModel):
    topic_id: str
    name: str
    course_id: str
    description: str = ""

class DocumentCreate(BaseModel):
    title: str
    content: str
    topic_id: str
    type: str
    language: str


class Feedback(BaseModel):
    run_id: str
    score: float


class CourseBase(BaseModel):
    name: str


class CourseResponse(CourseBase):
    id: str
    name: str
    model_config = ConfigDict(from_attributes=True)


class CourseAssignment(BaseModel):
    user_id: str
    course_id: str


class DocumentAddToTopic(BaseModel):
    topic_id: str
    content: str
    metadata: dict = {}


class Question(BaseModel):
    text: str
    user_id: str
    course_id: str
    topic_id: str

class ChatListItem(BaseModel):
    id: str
    topic_title: str
    timestamp: datetime

class ChatListResponse(BaseModel):
    user_id: str
    course_id: str
    chats: List[ChatListItem]