CELERY_IMPORTS = ['coordinator.tasks']
BROKER_URL = 'amqp://'
CELERY_RESULT_BACKEND = 'amqp://'