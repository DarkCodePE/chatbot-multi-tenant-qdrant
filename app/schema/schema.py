# app/schema.py
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from typing import List, TypedDict, Optional, Any, Dict, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from langchain_core.documents import Document


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    session_id: str
    courses: list[str]


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
    name: str = Field(..., min_length=3, max_length=200)


class CourseUpdate(BaseModel):
    name: str = Field(..., min_length=3, max_length=200)


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


class UploadDocument(BaseModel):
    course_id: str
    file_name: str
    file_content: bytes
    mime_type: str


class Feedback(BaseModel):
    run_id: str
    score: float


class CourseBase(BaseModel):
    name: str


class UserBase(BaseModel):
    id: str
    name: str
    email: str
    model_config = ConfigDict(from_attributes=True)


class CourseResponse(CourseBase):
    id: str
    name: str
    google_drive_folder_id: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    users: List[UserBase] = []
    model_config = ConfigDict(from_attributes=True)


class ProcessedDocumentResponse(BaseModel):
    id: str
    course_id: str
    google_file_id: str
    file_name: str
    last_modified: datetime
    qdrant_point_id: str


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


class State(TypedDict):
    input: str
    chat_history: Annotated[List[BaseMessage], "add_messages"]
    context: str
    answer: str
    documents: Optional[List[Document]]  # Para almacenar documentos recuperados
    web_search: Optional[str]  # Para decidir si realizar una búsqueda web
    course_id: str
