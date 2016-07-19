CELERY_IMPORTS = ['coordinator.tasks', 'coordinator.neuralnetwork']
BROKER_URL = 'amqp://'
CELERY_RESULT_BACKEND = 'amqp://'