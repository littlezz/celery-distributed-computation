server:
	gunicorn coordinator.gunicorn_run:app --bind 0.0.0.0:8080 --worker-class aiohttp.worker.GunicornWebWorker

node:
	celery multi restart 4 -A celelrytask
	python3 manager.py node
