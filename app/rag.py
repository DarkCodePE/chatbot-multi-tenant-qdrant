import os
import asyncio
from pathlib import Path
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
import logging
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from app.historial import QdrantChatMessageHistory

# Configuration
load_dotenv()
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHAT_HISTORY_COLLECTION =  os.getenv("CHAT_HISTORY_COLLECTION")
DOCS_FOLDER = Path("documents")
def get_session_history(session_id: str) -> QdrantChatMessageHistory:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantChatMessageHistory(session_id, qdrant_client, CHAT_HISTORY_COLLECTION)
class RAG:
    def __init__(self):
        self.qdrant = None
        self.embeddings = None
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.chain = None
        self.chain_with_history = None
    async def initialize(self):
        self.embeddings = OpenAIEmbeddings()
        self.qdrant = await self.initialize_qdrant()

    def get_loaders(self):
        web_loader = WebBaseLoader(["https://www.telefonica.com/es/sala-comunicacion/blog/actualizamos-nuestros-principios-de-inteligencia-artificial/"])
        pdf_loaders = [PyPDFLoader(str(pdf_path)) for pdf_path in DOCS_FOLDER.glob("*.pdf")]
        return [web_loader] + pdf_loaders

    def get_chunks(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)

    def configure_qdrant(self):
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        try:
            client.get_collection(COLLECTION_NAME)
            print(f"La colección {COLLECTION_NAME} ya existe.")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                print(f"La colección {COLLECTION_NAME} no existe. Creándola...")
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=rest.VectorParams(size=1536, distance=rest.Distance.COSINE),
                    hnsw_config=rest.HnswConfigDiff(payload_m=16, m=0),
                    optimizers_config=rest.OptimizersConfigDiff(indexing_threshold=0),
                    on_disk_payload=True
                )
            else:
                raise

        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="group_id",
                field_schema=rest.PayloadSchemaType.KEYWORD
            )
            print("Índice de payload 'group_id' creado con éxito.")
        except UnexpectedResponse as e:
            if e.status_code == 400:
                print("El índice de payload 'group_id' ya existe o no se pudo crear.")
                print(f"Detalles del error: {e.content}")
            else:
                raise

        try:
            client.create_shard_key(COLLECTION_NAME, "user_group_1")
            client.create_shard_key(COLLECTION_NAME, "user_group_2")
            print("Shards creados con éxito.")
        except UnexpectedResponse as e:
            print(f"No se pudieron crear los shards: {e.content}")

        return client

    async def initialize_qdrant(self):
        merged_loader = MergedDataLoader(loaders=self.get_loaders())
        all_docs = merged_loader.load()
        chunks = self.get_chunks(all_docs)

        self.configure_qdrant()

        prepared_docs = []
        for doc in chunks:
            doc_id = str(uuid4())
            group_id = "user_group_1"  # You might want to adjust this logic
            doc.metadata.update({"id": doc_id, "group_id": group_id})
            prepared_docs.append(doc)

        qdrant = QdrantVectorStore.from_documents(
            prepared_docs,
            self.embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            force_recreate=False
        )

        points_to_upsert = []
        for doc in prepared_docs:
            vector = self.embeddings.embed_query(doc.page_content)
            points_to_upsert.append(
                models.PointStruct(
                    id=doc.metadata["id"],
                    payload={
                        "group_id": doc.metadata["group_id"],
                        "metadata": doc.metadata,
                        "page_content": doc.page_content
                    },
                    vector=vector
                )
            )

        batch_size = 100
        for i in range(0, len(points_to_upsert), batch_size):
            batch = points_to_upsert[i:i + batch_size]
            qdrant.client.upsert(collection_name=COLLECTION_NAME, points=batch)

        return qdrant

    @traceable(run_type="retriever")
    async def retriever(self, query: str, group_id: str):
        logging.info(f"Recuperando documentos relevantes para la consulta: {query}")
        results = await asyncio.to_thread(
            self.qdrant.similarity_search,
            query,
            k=5,
            filter=models.Filter(
                must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=group_id))]
            )
        )
        return results
    @traceable(metadata={"model": "gpt-4o-mini"})
    async def rag(self, question: str, session_id: str, group_id: str):
        logging.info(f"Iniciando proceso RAG para la pregunta: {question}")
        try:
            # Recuperar documentos relacionados con la consulta
            docs = await self.retriever(question, group_id)
            logging.info(f"Documentos recuperadosxxx: {docs}")
            context = "\n".join(doc.page_content for doc in docs)

            # Crear el prompt con contexto de documentos y utilizar RunnableWithMessageHistory
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Responde a la pregunta del usuario utilizando solo la información proporcionada a continuación:\n\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])

            # Crear el modelo de lenguaje
            llm = ChatOpenAI(model_name="gpt-4o-mini")
            document_chain = create_stuff_documents_chain(llm, prompt)

            # Envolver con RunnableWithMessageHistory
            chain_with_history = RunnableWithMessageHistory(
                document_chain,
                get_session_history=get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )

            # Asegúrate de usar `await` al invocar el `chain_with_history`
            response = chain_with_history.invoke(
                {
                    "input": question,
                    "context": docs,
                },
                config={"configurable": {"session_id": session_id}}  # Pasar el session_id aquí
            )
            return response
        except Exception as e:
            logging.error(f"Error durante la invocación de chain_with_history: {e}", exc_info=True)
            raise

    async def chatbot(self, question: str, session_id: str, group_id: str):
        response = await self.rag(question, session_id, group_id)
        return response

async def main():
    rag_instance = RAG()
    await rag_instance.initialize()
    print(
        "Sistema RAG inicializado. Puede usar rag_instance.chatbot(question, session_id, group_id) para interactuar con el sistema.")

    question = "¿Cuáles son los principios de IA de Telefónica?"
    session_id = str(uuid4())  # Genera un ID de sesión único
    group_id = "user_group_1"
    response = await rag_instance.chatbot(question, session_id, group_id)
    print(f"Pregunta: {question}")
    print(f"Respuesta: {response}")

if __name__ == "__main__":
    asyncio.run(main())