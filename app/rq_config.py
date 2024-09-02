import os
from redis import Redis
from rq import Queue
import logging
# Usamos el nombre del contenedor de Redis como hostname
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD')
logging.info(f"Conectando a Redis en {redis_host}:{redis_port}")
redis_conn = Redis(host=redis_host, port=redis_port, password=redis_password)
queue = Queue(connection=redis_conn)
