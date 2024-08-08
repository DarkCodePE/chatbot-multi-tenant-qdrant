from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.rag import RAG
from langsmith import Client
import logging
import time
from functools import lru_cache
import asyncio

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modelos Pydantic
class Question(BaseModel):
    text: str
    group_id: str

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

# Inicialización de FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caché para respuestas
@lru_cache(maxsize=100)
def get_cached_response(question: str, group_id: str):
    return question, group_id
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
async def ask_question(question: Question):
    start_time = time.time()
    try:
        logging.info(f"Received question: {question.text[:50]}...")  # Log primeros 50 caracteres

        # Usar la caché solo para obtener los parámetros
        cached_question, cached_group_id = get_cached_response(question.text, question.group_id)

        # Obtener la instancia de RAG y llamar al método chatbot
        rag_instance = await RAGSingleton.get_instance()
        response, run_id = await rag_instance.chatbot(cached_question, cached_group_id)

        total_time = time.time() - start_time
        logging.info(f"Total processing time: {total_time:.2f} seconds")

        return {"response": response, "run_id": run_id}
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