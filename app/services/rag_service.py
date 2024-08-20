import os
from pathlib import Path
from qdrant_client import QdrantClient, models
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langsmith import traceable
from sqlalchemy import func

from app.collections import TopicRepository, UserDocumentRepository
from app.historial import QdrantChatMessageHistory
from app.model import ChatSession, User, Question
from app.schema.schema import Feedback, QuestionV2
from sqlalchemy.orm import Session

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


#TODO: Reemplazar QrantClient por RedisClient
def get_session_history(session_id: str):
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantChatMessageHistory(session_id, qdrant_client, CHAT_HISTORY_COLLECTION)


class RAGService:
    def __init__(self, database, rag_instance, llm):
        self.database = database
        self.rag = rag_instance
        self.llm = llm

    @traceable(metadata={"model": "gpt-4o-mini"})
    async def rag(self, question: str, session_id: str, all_docs):
        try:
            context = "\n".join(doc.page_content for doc in all_docs)
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Responde a la pregunta del usuario utilizando solo la información proporcionada a continuación:\n\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            chain_with_history = RunnableWithMessageHistory(
                document_chain,
                get_session_history=get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
            response = await chain_with_history.ainvoke(
                {
                    "input": question,
                    "context": context,
                },
                config={"configurable": {"session_id": session_id}}
            )
            return response
        except Exception as e:
            logging.error(f"Error durante la invocación de chain_with_history: {e}", exc_info=True)
            raise

    @traceable(run_type="chain")
    async def process_question(self, question: QuestionV2, db: Session):
        try:
            chat_session = db.query(ChatSession).filter(ChatSession.id == question.chat_session_id).first()
            if not chat_session:
                raise ValueError("Chat session not found")

            rag_response = await self.rag.chatbot(
                question.text,
                chat_session.user.session_id,
                chat_session.course.id if chat_session.course else None,
                chat_session.topic.id if chat_session.topic else None
            )

            # Guardar la pregunta en la base de datos
            db_question = Question(text=question.text, chat_session_id=chat_session.id)
            db.add(db_question)
            db.commit()

            return {
                "response": rag_response,
                "user": chat_session.user.name,
                "course": chat_session.course.name if chat_session.course else None,
                "topic": chat_session.topic.name if chat_session.topic else None
            }
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            raise

    async def start_chat_session(self, user_id: str, course_id: str = None, topic_id: str = None, db: Session = None):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        chat_session = ChatSession(
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_id,
            status="active"
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)

        return chat_session

    async def end_chat_session(self, chat_session_id: str, db: Session):
        chat_session = db.query(ChatSession).filter(ChatSession.id == chat_session_id).first()
        if not chat_session:
            raise ValueError("Chat session not found")

        chat_session.status = "completed"
        chat_session.end_time = func.now()
        db.commit()

    async def submit_feedback(self, feedback: Feedback, db: Session):
        try:
            db_feedback = Feedback(
                chat_session_id=feedback.chat_session_id,
                score=feedback.score,
                comment=feedback.comment
            )
            db.add(db_feedback)
            db.commit()
            return {"message": "Feedback received successfully"}
        except Exception as e:
            logging.error(f"Error submitting feedback: {str(e)}")
            raise
