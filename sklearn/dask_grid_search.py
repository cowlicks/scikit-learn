from .externals import six
from .pipeline import _name_estimators

from dask.imperative import do, value


def fit_transform(e, x, y, fit_params={}):
    return e.fit_transform(x, y, **fit_params)


def simple_pipe_fit(estimators, parameters, x, y):
    lazy_fit_transform = do(fit_transform)

    estimators = _name_estimators(estimators)
    fit_params_steps = dict((name, {}) for name, _ in estimators)
    for pname, pval in six.iteritems(parameters):
        step, param = pname.split('__', 1)
        fit_params_steps[step][param] = pval

    xt = x
    for name, est in estimators:
        xt = lazy_fit_transform(est, xt, y, **fit_params_steps[name])
    return xt


def dask_grid_search(estimators, paramters):
    pass
