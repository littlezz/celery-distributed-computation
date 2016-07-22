server:
	gunicorn coordinator.gunicorn_run:app --bind 0.0.0.0:8080 --worker-class aiohttp.worker.GunicornWebWorker

nodes:
	celery multi restart 4 -A celerytask
	python3 manage.py node
