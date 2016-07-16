from . import celeryconfig
from celery import Celery

app = Celery('rdc', )
app.config_from_object(celeryconfig)


# app.autodiscover_tasks(['coordinator'], force=True)
