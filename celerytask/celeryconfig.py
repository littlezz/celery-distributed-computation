CELERY_IMPORTS = ['coordinator.tasks', 'coordinator.neuralnetwork']
BROKER_URL = 'amqp://test:test@192.168.2.104//'
# BROKER_URL = 'amqp://'
# CELERY_RESULT_BACKEND = 'amqp://test:test@192.168.2.104//'
CELERY_RESULT_BACKEND = 'redis://192.168.2.104'


CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
