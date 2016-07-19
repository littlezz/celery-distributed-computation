import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit as sigmoid
from itertools import chain
from celerytask.celery import app
from celery.contrib.methods import task_method
from celery.task import chord


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

    @property
    def N(self):
        """number of N sample"""
        return self._raw_train_x.shape[0]

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


class MiniBatchNeuralNetwork(BaseNeuralNetwork):
    """
    Depend on many book.
    use mini batch update instead af batch update.

    reference
    ---------
    http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
    https://www.coursera.org/learn/machine-learning/lecture/1z9WW/backpropagation-algorithm
    """
    def __init__(self, *args, mini_batch=10, hidden_layer_shape=None, **kwargs):
        self.mini_batch = mini_batch
        self.hidden_layer_shape = hidden_layer_shape or list()
        super().__init__(*args, **kwargs)

    @staticmethod
    def random_weight_matrix(shape):
        return np.random.uniform(-0.7, 0.7, shape)


    def _one_iter_train(self):
        X = self.train_x
        y = self.train_y
        mini_batch = self.mini_batch
        for j in range(0, self.N, mini_batch):
            # x = X[j: j + mini_batch]
            # target = y[j: j + mini_batch]
            # layer_output = self._forward_propagation(x)
            # self._back_propagation(target=target, layer_output=layer_output)

            task = chord(self._one_forward_and_back.s(X[k], y[k]) for k in range(j, j+mini_batch))(self._sum_update.s())
            print(task.get())


    @app.task(filter=task_method)
    def _one_forward_and_back(self, x, y):
        """
        forward and back, return (grad, intercept)
        :param x:
        :param y:
        :return: (grad, intercept)
        """
        layer_output = self._forward_propagation(x)
        return self._back_propagation(layer_output=layer_output, target=y)


    @app.task(filter=task_method)
    def _sum_update(self, values):
        for (theta, intercept), *i in zip(reversed(self.thetas), *values):
            tt = ti = 0
            for (theta_i, intercept_i) in i:
                tt+=theta_i
                ti+=intercept_i
            tt/=len(i)
            ti/=len(i)
            theta -= tt
            intercept -= ti



    def train(self):
        self._init_theta()
        for r in range(self.n_iter):
            self._one_iter_train()

    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        y = self._forward_propagation(X)[-1]
        return self._inverse_matrix_to_class(y)

    @property
    def rss(self):
        eps = 1e-50
        X = self._pre_processing_x(self._raw_train_x)
        y = self._forward_propagation(X)[-1]
        y[y < eps] = eps
        return - np.sum(np.log(y) * self.train_y)

    def _init_theta(self):
        """
        theta is weights
        init all theta, depend on hidden layer
        :return: No return, store the result in self.thetas which is a list
        """
        thetas = []
        input_dimension = self.train_x.shape[1]
        for target_dimension in chain(self.hidden_layer_shape, [self.n_class]):
            _theta = np.random.uniform(-0.7, 0.7, (input_dimension + 1, target_dimension))
            theta = _theta[1:]
            intercept = _theta[0]
            thetas.append((theta, intercept))
            input_dimension = target_dimension
        self.thetas = thetas

    def _forward_propagation(self, x):
        a = x.copy()
        layer_output = list()
        layer_output.append(a)
        for theta, intercept in self.thetas:
            a = sigmoid(a @ theta + intercept)
            layer_output.append(a)
        return layer_output

    def _back_propagation(self, target, layer_output):
        delta = -(target - layer_output[-1])

        ret = []

        for (theta, intercept), a in zip(reversed(self.thetas), reversed(layer_output[:-1])):
            grad = a.T @ delta
            intercept_grad = np.sum(delta, axis=0)
            delta = ((1 - a) * a) * (delta @ theta.T)
            # theta -= grad * self.alpha / self.mini_batch
            # intercept -= intercept_grad * self.alpha / self.mini_batch
            ret.append((grad * self.alpha, intercept_grad * self.alpha))

        return ret

