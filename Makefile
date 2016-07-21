server:
	gunicorn coordinator.gunicorn_run:app --bind 0.0.0.0:8080 --worker-class aiohttp.worker.GunicornWebWorker
