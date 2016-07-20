from . import celeryconfig
from celery import Celery
from redis import Redis
from redis_lock import Lock

cache = Redis()

weights_name = ['weights', 'weights0', 'weights1', 'weights10']
locks = [Lock(cache, 'lock' + wn) for wn in weights_name]
one_lock = Lock(cache, 'the-lock')

app = Celery('rdc', )
app.config_from_object(celeryconfig)


# app.autodiscover_tasks(['coordinator'], force=True)
