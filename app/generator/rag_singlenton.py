import asyncio

from app.generator.rag import RAG


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


async def get_rag_instance():
    return await RAGSingleton.get_instance()
