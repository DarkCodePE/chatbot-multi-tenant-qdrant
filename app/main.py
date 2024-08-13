from uuid import uuid4
import random
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.database import Database
from app.rag import RAG
from langsmith import Client
import logging
import time
from functools import lru_cache
import asyncio

from sqlalchemy.orm import Session
from app.model import User as UserModel
from app.database import init_db

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UserResponse(BaseModel):
    id: str
    name: str
    group_id: str
    session_id: str
# Modelos Pydantic
class Question(BaseModel):
    text: str
    user_id: str

class Feedback(BaseModel):
    run_id: str
    score: float

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

# Inicializa la base de datos (crea las tablas si no existen)
init_db()
# Inicialización de FastAPI
app = FastAPI()
database = Database()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserCreate(BaseModel):
    name: str
    group_id: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(database.get_db)):
    db_user = UserModel(
        id=str(uuid4()),
        name=user.name,
        group_id=user.group_id,
        session_id=str(uuid4())
    )
    return database.create_user(db, db_user)

@app.get("/users/{user_id}", response_model=UserResponse)
def read_user(user_id: str, db: Session = Depends(database.get_db)):
    db_user = database.get_user_by_id(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/ask")
async def ask_question(question: Question, db: Session = Depends(database.get_db)):
    start_time = time.time()
    try:
        logging.info(f"Received question: {question.text[:50]}...")

        # Verificar si el usuario existe
        db_user = database.get_user_by_id(db, question.user_id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        logging.info(f"User found ->: {db_user}")

        # Obtener la instancia de RAG y llamar al método chatbot
        rag_instance = await RAGSingleton.get_instance()
        response = await rag_instance.chatbot(
            question.text,
            db_user.session_id,
            db_user.group_id
        )

        total_time = time.time() - start_time
        logging.info(f"Total processing time: {total_time:.2f} seconds")

        return {
            "response": response,
            "user": db_user
        }
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: Feedback):
    try:
        ls_client = Client()
        ls_client.create_feedback(
            feedback.run_id,
            key="user-score",
            score=feedback.score
        )
        logging.info(f"Feedback received for run_id: {feedback.run_id}")
        return {"message": "Feedback recibido con éxito"}
    except Exception as e:
        logging.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)