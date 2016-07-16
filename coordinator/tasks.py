from celerytask.celery import app


@app.task
def test_func():
    pass