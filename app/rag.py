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

# Configuration
load_dotenv()
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

DOCS_FOLDER = Path("documents")

class RAG:
    def __init__(self):
        self.qdrant = None
        self.embeddings = None
        self.openai_client = AsyncOpenAI()

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
        results = self.qdrant.similarity_search(
            query,
            k=5,
            filter=models.Filter(
                must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=group_id))]
            )
        )

        print(f"Resultados para group_id: {group_id}")
        for doc in results:
            print(f"Content: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
            print("---")

        return results

    @traceable(metadata={"model": "gpt-4o-mini"})
    async def rag(self, question: str, group_id: str):
        docs = await self.retriever(question, group_id)
        context = "\n".join(doc.page_content for doc in docs)
        system_message = f"""Responde a la pregunta del usuario utilizando solo la información proporcionada a continuación:

        {context}"""

        response = await self.openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            model="gpt-4o-mini",
        )
        return response

    async def chatbot(self, question: str, group_id: str):
        run_id = str(uuid4())
        response = await self.rag(question, group_id, langsmith_extra={"run_id": run_id})
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content, run_id
        else:
            raise ValueError("No se recibió una respuesta válida del modelo de lenguaje")

async def verify_documents(qdrant_client, collection_name):
    response = await qdrant_client.scroll(
        collection_name=collection_name,
        limit=10,
        with_payload=True,
        with_vectors=False
    )

    for point in response[0]:
        print(f"ID: {point.id}")
        print(f"Payload: {point.payload}")
        print("---")

async def main():
    rag_instance = RAG()
    await rag_instance.initialize()
    print("RAG system initialized. You can now use rag_instance.chatbot(question, group_id) to interact with the system.")

    question = "What are Telefónica's AI principles?"
    group_id = "user_group_1"
    response, run_id = await rag_instance.chatbot(question, group_id)
    print(f"Question: {question}")
    print(f"Response: {response}")
    print(f"Run ID: {run_id}")

    await verify_documents(rag_instance.qdrant.client, COLLECTION_NAME)

if __name__ == "__main__":
    asyncio.run(main())