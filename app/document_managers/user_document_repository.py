from uuid import uuid4
from qdrant_client import QdrantClient, models


class UserDocumentManager:
    def process_user_document(self, file, session_id):
        # Procesar el archivo subido
        content = self.extract_text_from_pdf(file)

        # Vectorizar y almacenar en Qdrant
        vector = self.embeddings.embed_query(content)
        self.qdrant_client.upsert(
            collection_name="user_documents",
            points=[models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "session_id": session_id,
                    "content": content
                }
            )]
        )

    def get_user_documents(self, session_id):
        # Recuperar documentos del usuario de Qdrant
        results = self.qdrant_client.search(
            collection_name="user_documents",
            query_vector=[0] * 1536,  # Vector dummy para búsqueda por filtro
            query_filter=models.Filter(
                must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
            )
        )
        return results