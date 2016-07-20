import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit as sigmoid
from itertools import chain
from celerytask.celery import app, weights_name
from celery.contrib.methods import task_method, task
from celery import chord, group
from common.redis_cache import pickle_redis_cache, _pickle_get_pipe, _pickle_set_pipe
from celerytask import cache, locks
from collections import deque
from common import get_random_string
import random

class CacheAlias:
    train_x = pickle_redis_cache('train_x')
    train_y = pickle_redis_cache('train_y')
    weights = pickle_redis_cache('weights')
    weights0 = pickle_redis_cache('weights0')
    weights1 = pickle_redis_cache('weights1')
    weights10 = pickle_redis_cache('weights10')
    N = pickle_redis_cache('N')
    alpha = pickle_redis_cache('alpha')
    iter_time = pickle_redis_cache('iter_time')
    
    
alias = CacheAlias()

class WeightCache:
    key = get_random_string()
    weights = pickle_redis_cache('weights')
    weights0 = pickle_redis_cache('weights0')
    weights1 = pickle_redis_cache('weights1')
    weights10 = pickle_redis_cache('weights10')





class IntuitiveMethodRssMixin:
    """
    this class for fix the Intuitive Network class rss method.
    """
    @property
    def rss(self):
        eps = 1e-50
        y = self._y_hat
        y[y < eps] = eps
        return - np.sum(np.log(y) * self.train_y)


class Result:
    def __init__(self, y_hat: np.ndarray, y: np.ndarray):
        self.y_hat = y_hat
        self.y = y
        self.prediction_error = np.power(self.y_hat - self.y, 2)
        self.N = y.shape[0]

    @property
    def mse(self):
        """
        mean of prediction error
        :return:
        """
        return np.sum(self.prediction_error) / self.N

    @property
    def std_error(self):
        """
        Standard Error of prediction error
        :return:
        """
        return (np.var(self.prediction_error, ddof=1) / self.N)**0.5

    @property
    def error_rate(self):
        return 1 - (np.sum((self.y_hat == self.y)) / self.N)


class BaseStatModel:
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, features_name=None, do_standardization=True):
        # ensure that train_y is (N x 1)
        train_y = train_y.reshape((train_y.shape[0], 1))
        self.train_x = train_x
        self._raw_train_x = train_x.copy()
        self._raw_train_y = train_y.copy()
        self.train_y = train_y
        self.features_name = features_name

        self.do_standardization = do_standardization
        self._x_std_ = None
        self._x_mean_ = None

    def standardize(self, x, axis=0, with_mean=True, with_std=True):
        if not self.do_standardization:
            return x

        if getattr(self, '_x_std_', None) is None or getattr(self, '_x_mean_', None) is None:
            self._x_mean_ = x.mean(axis=axis)
            self._x_std_ = x.std(axis=axis, ddof=1)
        if with_mean:
            x = x - self._x_mean_
        if with_std:
            x = x / self._x_std_
        return x

    # @property
    # def N(self):
    #     """number of N sample"""
    #     return self._raw_train_x.shape[0]

    @property
    def p(self):
        """
        number of features exclude intercept one
        :return:
        """
        return self._raw_train_x.shape[1]

    def _pre_processing_x(self, X: np.ndarray):
        return X

    def _pre_processing_y(self, y):
        return y

    def pre_processing(self):
        self.train_x = self._pre_processing_x(self.train_x)
        self.train_y = self._pre_processing_y(self.train_y)

    def train(self):
        raise NotImplementedError

    def predict(self, X: np.ndarray):
        raise NotImplementedError

    def test(self, X, y):
        y_hat = self.predict(X)
        y = y.reshape((y.shape[0], 1))
        return Result(y_hat, y)


class ClassificationMixin(BaseStatModel):
    def __init__(self, *args, n_class=None, **kwargs):
        self.n_class = n_class
        self._label_map = dict()
        super().__init__(*args, **kwargs)

    def _get_unique_sorted_label(self):
        y = self._raw_train_y
        unique_label = np.unique(y)
        sorted_label = np.sort(unique_label)
        return sorted_label

    def _pre_processing_y(self, y):
        y = super()._pre_processing_y(y)

        # reference sklearn.preprocessing.label.py
        sorted_label = self._get_unique_sorted_label()
        if self.n_class is None:
            self.n_class = len(sorted_label)

        cols = np.searchsorted(sorted_label, y.flatten())
        rows = np.arange(0, y.shape[0])
        data = np.ones_like(rows)
        matrix = csr_matrix((data, (rows, cols)), shape=(y.shape[0], self.n_class)).toarray()
        return matrix

    def _inverse_matrix_to_class(self, matrix):
        """
        inverse indicator matrix to multi class
        :param matrix:
        :return:
        """
        index = matrix.argmax(axis=1)
        sorted_label = self._get_unique_sorted_label()
        return sorted_label[index].reshape((-1, 1))


class BaseNeuralNetwork(ClassificationMixin, BaseStatModel):
    def __init__(self, *args, n_iter=10, **kwargs):
        self.alpha = kwargs.pop('alpha')
        self.n_iter = n_iter
        super().__init__(*args, **kwargs)

    def _pre_processing_x(self, X: np.ndarray):
        # Manual add bias "1" in `train`
        X = self.standardize(X)
        return X

    @property
    def y_hat(self):
        return self.predict(self._raw_train_x)

    @property
    def rss(self):
        raise NotImplementedError


class IntuitiveNeuralNetwork2(BaseNeuralNetwork):
    train_x = pickle_redis_cache('train_x')
    train_y = pickle_redis_cache('train_y')
    weights = pickle_redis_cache('weights')
    weights0 = pickle_redis_cache('weights0')
    weights1 = pickle_redis_cache('weights1')
    weights10 = pickle_redis_cache('weights10')
    N = pickle_redis_cache('N')
    alpha = pickle_redis_cache('alpha')
    iter_time = pickle_redis_cache('iter_time')
    
    def __init__(self, *args, hidden=12, iter_time=3,**kwargs):
        self.hidden = hidden
        self.iter_time = iter_time
        super().__init__(*args, **kwargs)
        self.N = self._raw_train_x.shape[0]

    @property
    def rss(self):
        eps = 1e-50
        a2 = sigmoid(self._raw_train_x @ self.weights + self.weights0)
        y = sigmoid(a2 @ self.weights1 + self.weights10)
        y[y < eps] = eps
        return - np.sum(np.log(y) * self.train_y)

    def pre_processing(self):
        super().pre_processing()
        self._init_weights()
        
    def _init_weights(self):
        _weights = np.random.uniform(-0.7, 0.7, (self.p + 1, self.hidden))
        weights = _weights[1:]
        weights0 = _weights[0]
        _weights = np.random.uniform(-0.7, 0.7, (self.hidden + 1, self.n_class))
        weights1 = _weights[1:]
        weights10 = _weights[0]
        self.weights = weights
        self.weights0 = weights0
        self.weights1 = weights1
        self.weights10 = weights10
        


    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        a2 = sigmoid(X @ self.weights + self.weights0)
        y = sigmoid(a2 @ self.weights1 + self.weights10)
        return self._inverse_matrix_to_class(y)

    @property
    def y_hat(self):
        return self.predict(self._raw_train_x)

@app.task
def train():
    N  = alias.N
    step = N // 8
    group(_train.s(i, i+step) for i in range(0, step*8, step))().get()
    train.delay()

@app.task
def _train(start, stop):
    X = alias.train_x
    y = alias.train_y
    N = stop-start
    weights = alias.weights
    weights0 = alias.weights0
    weights1 = alias.weights1
    weights10 = alias.weights10
    alpha = alias.alpha

    

    w1 = np.zeros_like(weights1)
    w10 = np.zeros_like(weights10)
    w = np.zeros_like(weights)
    w0 = np.zeros_like(weights0)

    for i in range(start, stop):
        x = X[i]
        a1 = x
        z2 = x @ weights + weights0
        a2 = sigmoid(z2)
        z3 = a2 @ weights1 + weights10
        a3 = sigmoid(z3)

        delta3 = -(y[i] - a3)
        delta2 = weights1 @ delta3 * a2 * (1 - a2)

        w1 += alpha * (a2[:, None] @ delta3[None, :])
        w10 += alpha * delta3

        w += alpha * (a1[:, None] @ delta2[None, :])
        w0 += alpha * delta2

    w0 /= N
    w /= N
    w10 /= N
    w1 /= N
    wc = WeightCache()
    wc.weights = w
    wc.weights0 = w0
    wc.weights1 = w1
    wc.weights10= w10
    update_cache_weight.delay(wc.key)



@app.task
def update_cache_weight(key):
    wc = WeightCache()
    wc.key = key
    l = list(zip(weights_name, locks))
    random.shuffle(l)
    for weight_name, lock in l:
        with lock:
            w = getattr(alias, weight_name)
            wn = getattr(wc, weight_name)
            setattr(alias, weight_name, w - wn)


# _start = None
#
from esl_model.datasets import ZipCodeDataSet
d = ZipCodeDataSet()
nn = IntuitiveNeuralNetwork2(train_x=d.train_x, train_y=d.train_y, n_class=10,alpha=0.38)

nn.pre_processing()

# def start_nn(wait=0):
#     global _start
#     if _start is not None:
#         return _start
#
#     t = nn.train.delay()
#     if wait:
#         t.get()
#     _start = nn
#     return nn
