from . import celeryconfig
from celery import Celery
from redis import Redis
from redis_lock import Lock

cache = Redis()

lock = Lock(cache, 'the-redis-lock')

app = Celery('rdc', )
app.config_from_object(celeryconfig)


# app.autodiscover_tasks(['coordinator'], force=True)
