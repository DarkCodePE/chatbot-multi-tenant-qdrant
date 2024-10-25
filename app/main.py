import io
from http.client import HTTPException
from typing import List, Dict

from celery.result import AsyncResult
from fastapi import FastAPI, Depends, BackgroundTasks, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from app.database import Database
import logging
from sqlalchemy.orm import Session
from app.database import init_db
from app.schema.schema import UserResponse, CourseResponse, TopicResponse, DocumentCreate, CourseCreate, UserLogin, \
    TopicCreate, CourseAssignment, QuestionV2, Feedback, DocumentAddToTopic, ChatSessionStart, ChatSessionEnd, \
    ChatListResponse, UploadDocument, ProcessedDocumentResponse, UserCreate, CourseUpdate
from app.model import User as UserModel, Course as CourseModel, Topic as TopicModel, Question as QuestionModel, \
    ChatSession, Document as DocumentModel, Course, Topic, ProcessedDocument
from app.services.services import UserService, CourseService, TopicService, QuestionService
import asyncio
import json

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


@app.post("/users/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(database.get_db)):
    return await user_service.register_user(user, db)


@app.get("/courses/{course_id}/unassigned-users", response_model=List[UserResponse])
async def get_unassigned_users(course_id: str, db: Session = Depends(database.get_db)):
    """
    Endpoint para obtener usuarios no asignados a un curso específico.

    Args:
        course_id: ID del curso
        db: Sesión de base de datos

    Returns:
        Lista de usuarios no asignados al curso
    """
    return await user_service.get_unassigned_users(course_id, db)


@app.post("/courses", response_model=CourseResponse)
async def create_course(course: CourseCreate, db: Session = Depends(database.get_db)):
    return await course_service.create_course(course, db)


@app.get("/courses", response_model=List[CourseResponse])
def get_courses(db: Session = Depends(database.get_db)):
    return course_service.get_all_courses(db)


@app.put("/courses/{course_id}", response_model=CourseResponse)
async def update_course(course_id: str, course_update: CourseUpdate, db: Session = Depends(database.get_db)):
    try:
        updated_course = await course_service.update_course(course_id, course_update, db)
        return updated_course
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error al actualizar el curso: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.delete("/courses/{course_id}", response_model=Dict[str, str])
async def delete_course(course_id: str, db: Session = Depends(database.get_db)):
    try:
        result = await course_service.delete_course(course_id, db)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error al eliminar el curso: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/topics", response_model=TopicResponse)
async def create_topic(topic: TopicCreate, db: Session = Depends(database.get_db)):
    return await topic_service.create_topic(topic, db)


@app.get("/users/{user_id}", response_model=UserResponse)
def read_user(user_id: str, db: Session = Depends(database.get_db)):
    return user_service.get_user(user_id, db)


@app.post("/upload-document")
async def upload_document(
        course_id: str = Form(...),
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db)
):
    try:
        file_content = await file.read()
        upload_file = UploadDocument(course_id=course_id, file_name=file.filename, file_content=file_content,
                                     mime_type=file.content_type)
        result = await course_service.upload_document(file=upload_file, db=db)
        return result
    except Exception as e:
        logging.error(f"Error al subir el documento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-document/{document_id}")
async def download_document(
        document_id: str,
        db: Session = Depends(database.get_db)
):
    """
    Endpoint para descargar un documento.
    No requiere validación de usuario ya que es manejado por administradores.
    """
    try:
        # Descargar el documento directamente
        file_content, file_name = await course_service.download_document(
            document_id, db
        )

        # Crear un stream para el archivo
        stream = io.BytesIO(file_content)

        return StreamingResponse(
            stream,
            media_type='application/octet-stream',
            headers={
                'Content-Disposition': f'attachment; filename="{file_name}"'
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading document: {str(e)}"
        )


@app.delete("/delete-document/{document_id}")
async def delete_document(
        document_id: str,
        db: Session = Depends(database.get_db)
) -> Dict[str, str]:
    """
    Endpoint para eliminar un documento.
    No requiere validación de usuario ya que es manejado por administradores.
    """
    try:
        # Eliminar el documento directamente
        result = await course_service.delete_document(document_id, db)
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@app.post("/users/{user_id}/documents")
async def add_user_document(user_id: str, document: DocumentCreate, db: Session = Depends(database.get_db)):
    return await user_service.add_user_document(user_id, document, db)


@app.get("/courses/{course_id}/files", response_model=List[ProcessedDocumentResponse])
async def get_course_files(course_id: str, db: Session = Depends(database.get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    files = db.query(ProcessedDocument).filter(ProcessedDocument.course_id == course_id).all()
    return files


@app.get("/courses/{course_id}", response_model=CourseResponse)
async def get_course_detail(course_id: str, db: Session = Depends(database.get_db)):
    """
    Endpoint para obtener los detalles de un curso específico, incluyendo sus usuarios.
    """
    try:
        course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")

        return CourseResponse(
            id=course.id,
            name=course.name,
            google_drive_folder_id=course.google_drive_folder_id,
            created_at=course.created_at,
            updated_at=course.updated_at,
            users=[{
                'id': user.id,
                'name': user.name,
                'email': user.email
            } for user in course.users]
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error fetching course details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/courses/{course_id}/users/{user_id}")
async def remove_user_from_course(
        course_id: str,
        user_id: str,
        db: Session = Depends(database.get_db)
):
    """
    Endpoint para desasignar un usuario de un curso.
    """
    try:
        result = await course_service.remove_user_from_course(course_id, user_id, db)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error removing user from course: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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

@app.get("/sse/topic/{topic_id}")
async def sse_topic(topic_id: str, db: Session = Depends(database.get_db)):
    logging.info(f"Conectando a la sala de chat para el tema {topic_id}")

    async def event_generator():
        last_title = None
        while True:
            topic = db.query(TopicModel).filter(TopicModel.id == topic_id).first()
            logging.info(f"Comprobando si hay cambios en el tema {topic_id}")
            if topic and topic.name != last_title:
                last_title = topic.name
                yield f"data: {json.dumps({'title': topic.name})}\n\n"
            await asyncio.sleep(1)  # Comprueba cada segundo

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/start")
async def start_chat_session(session_start: ChatSessionStart,
                             db: Session = Depends(database.get_db)):
    result = await question_service.start_chat_session(session_start, db)
    return result


@app.post("/chat/question")
async def process_question(question: QuestionV2, db: Session = Depends(database.get_db)):
    return await question_service.process_question(question, db)


@app.post("/chat/end")
async def end_chat_session(chat_end: ChatSessionEnd, db: Session = Depends(database.get_db)):
    return await question_service.end_chat_session(chat_end.chat_session_id, db)


@app.post("/feedback")
async def submit_feedback(feedback: Feedback, db: Session = Depends(database.get_db)):
    return await question_service.submit_feedback(feedback, db)


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


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    from app.event.tasks import generate_and_update_title
    task = AsyncResult(task_id)
    #task = generate_and_update_title.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is pending...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': task.info.get('status', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'status': 'Task completed successfully',
            'result': task.result
        }
    else:  # FAILURE or other states
        response = {
            'state': task.state,
            'status': 'Task failed',
            'error': str(task.info.get('error', 'Unknown error occurred'))
        }
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
