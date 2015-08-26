import dask
from dask.async import get_sync
dask.set_options(get=get_sync)
from ipdb import set_trace
from copy import copy
from .base import is_classifier
from .externals import six
from .pipeline import _name_estimators
from .grid_search import ParameterGrid
from .cross_validation import check_cv

from dask.imperative import do, value


def fit_transform(e, x, y, params={}):
    e.set_params(**params)
    return e.fit_transform(x, y)

def fit(e, x, y, params={}):
    e.set_params(**params)
    return e.fit(x, y)

def simple_pipe_fit(estimators, parameters, x, y):
    lazy_fit_transform = do(fit_transform)
    lazy_fit = do(fit)
    estimators = copy(estimators)

    estimators = _name_estimators(estimators)
    params_steps = dict((name, {}) for name, _ in estimators)
    for pname, pval in six.iteritems(parameters):
        step, param = pname.split('__', 1)
        params_steps[step][param] = pval

    xt = x
    for name, est in estimators[:-1]:
        xt = lazy_fit_transform(est, xt, y, params_steps[name])

    name, transform = estimators[-1]
    xt = lazy_fit(transform, xt, y, params_steps[name])

    return xt


def dask_grid_search(estimators, parameters, x, y):
    res = [simple_pipe_fit(estimators, p, x, y) for p in ParameterGrid(parameters)]
    all_estimators = value(res).compute()


def sort(estimator, X, y):
    cv = None
    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    # get test
    # get train
