from http.client import HTTPException
from typing import List
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.database import Database
import logging
from sqlalchemy.orm import Session
from app.database import init_db
from app.schema.schema import UserResponse, CourseResponse, TopicResponse, DocumentCreate, CourseCreate, UserLogin, \
    TopicCreate, CourseAssignment, QuestionV2, Feedback, DocumentAddToTopic, ChatSessionStart, ChatSessionEnd, \
    ChatListResponse
from app.services.services import UserService, CourseService, TopicService, QuestionService

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Inicializa la base de datos
init_db()

# Servicios
user_service = UserService(database)
course_service = CourseService(database)
topic_service = TopicService(database)
question_service = QuestionService(database)


# Rutas de la API

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/users/login", response_model=UserResponse)
async def user_login(user: UserLogin, db: Session = Depends(database.get_db)):
    return await user_service.login_user(user, db)


@app.post("/courses", response_model=CourseResponse)
async def create_course(course: CourseCreate, db: Session = Depends(database.get_db)):
    return await course_service.create_course(course, db)


@app.get("/courses", response_model=List[CourseResponse])
def get_courses(db: Session = Depends(database.get_db)):
    return course_service.get_all_courses(db)


@app.post("/topics", response_model=TopicResponse)
async def create_topic(topic: TopicCreate, db: Session = Depends(database.get_db)):
    return await topic_service.create_topic(topic, db)


@app.get("/users/{user_id}", response_model=UserResponse)
def read_user(user_id: str, db: Session = Depends(database.get_db)):
    return user_service.get_user(user_id, db)


@app.post("/users/{user_id}/documents")
async def add_user_document(user_id: str, document: DocumentCreate, db: Session = Depends(database.get_db)):
    return await user_service.add_user_document(user_id, document, db)


@app.get("/users/{user_id}/courses", response_model=List[CourseResponse])
def get_user_courses(user_id: str, db: Session = Depends(database.get_db)):
    return user_service.get_user_courses(user_id, db)


@app.post("/courses/{course_id}/topics/{topic_id}")
async def assign_topic_to_course(course_id: str, topic_id: str, db: Session = Depends(database.get_db)):
    return await course_service.assign_topic_to_course(course_id, topic_id, db)


@app.post("/users/assign-course")
def assign_course_to_user(assignment: CourseAssignment, db: Session = Depends(database.get_db)):
    return user_service.assign_course_to_user(assignment, db)


@app.post("/topics/{topic_id}/documents")
async def add_document_to_topic(topic_id: str, document: DocumentAddToTopic, db: Session = Depends(database.get_db)):
    document.topic_id = topic_id
    return await topic_service.add_document_to_topic(document, db)


# Nuevos endpoints para manejar sesiones de chat

@app.post("/chat/start")
async def start_chat_session(session_start: ChatSessionStart, db: Session = Depends(database.get_db)):
    return await question_service.start_chat_session(session_start, db)


@app.post("/chat/question")
async def process_question(question: QuestionV2, db: Session = Depends(database.get_db)):
    return await question_service.process_question(question, db)


@app.post("/chat/end")
async def end_chat_session(chat_end: ChatSessionEnd, db: Session = Depends(database.get_db)):
    return await question_service.end_chat_session(chat_end.chat_session_id, db)


@app.post("/feedback")
async def submit_feedback(feedback: Feedback, db: Session = Depends(database.get_db)):
    return await question_service.submit_feedback(feedback, db)


@app.post("/process-google-drive")
async def process_google_drive(folder_name: str, course_id: str, topic_id: str, db: Session = Depends(database.get_db)):
    try:
        result = await topic_service.process_google_drive_documents(folder_name, course_id, topic_id, db)
        return {"message": "Procesamiento de carpeta de Google Drive completado", "result": result}
    except Exception as e:
        raise HTTPException(str(e))


@app.get("/chats/{user_id}/{course_id}", response_model=ChatListResponse)
async def get_chat_list(user_id: str, course_id: str, db: Session = Depends(database.get_db)):
    try:
        return question_service.get_chat_list(user_id, course_id, db)
    except Exception as e:
        raise HTTPException(str(e))


@app.get("/chat/{chat_id}/history", response_model=List[dict])
async def get_chat_history(chat_id: str, db: Session = Depends(database.get_db)):
    try:
        return await question_service.get_chat_history(chat_id, db)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
