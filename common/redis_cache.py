from celerytask import cache
import pickle
import zlib

compress_level = 1

def _pickle_set_pipe(value):
    s = pickle.dumps(value)
    s = zlib.compress(s, compress_level)
    return s


def _pickle_get_pipe(value):
    return pickle.loads(zlib.decompress(value))

def pickle_redis_cache(name):
    """
    auto pickle the value then save to redis, then auto put back to python type
    :param name: unique name in redis
    :return:
    """

    store_name = 'pickle_redis_' + name

    @property
    def prop(self):
        value = cache.get(store_name)
        return _pickle_get_pipe(value)

    @prop.setter
    def prop(self, value):
        cache.set(store_name, _pickle_set_pipe(value))

    return prop