from celery import Celery
import os
import logging

from celery.schedules import crontab
from celery.signals import after_setup_logger

#redis_host = os.getenv('REDIS_HOST', 'localhost')
#logging.info("REDIS_HOST %s", redis_host)
redis_host = os.getenv('REDIS_HOST', 'redis')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD')
redis_url = f'redis://{redis_host}:{redis_port}/0'
logging.info(redis_url)
app = Celery('app', broker=redis_url, backend=redis_url)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=30,
    task_max_retries=3,
)

app.conf.broker_transport_options = {
    'socket_timeout': 30,
    'socket_connect_timeout': 30,
}
app.conf.result_backend_transport_options = {
    'socket_timeout': 30,
    'socket_connect_timeout': 30,
}
app.conf.beat_schedule = {
    'sync-all-courses-every-day': {
        'task': 'tasks.sync_all_courses',
        'schedule': crontab(hour=2, minute=0),  # Ejecutar todos los días a las 2:00 AM
    },
}

app.autodiscover_tasks(lambda: ['app.event.tasks'])

logging.basicConfig(level=logging.DEBUG)
