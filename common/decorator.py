import logging
from functools import wraps, partial

def set_debug(func):
    @wraps(func)
    def wrapper(*args, debug=False,**kwargs):
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        return func(*args, **kwargs)
    return wrapper