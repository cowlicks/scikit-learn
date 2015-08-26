import numpy as np
from numpy.testing import assert_array_almost_equal
from dask.imperative import value
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import make_pipeline, make_dask_pipeline
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from ipdb import set_trace

from ..dask_grid_search import fit_transform, simple_pipe_fit, dask_grid_search


iris = load_iris()

def eq(a, b):
    # assert that estimators are almost equal
    assert_array_almost_equal(a.transform(iris.data),
                              b.transform(iris.data))


def test_dask_gridsearch():
    #LinearSVC().fit(iris.data, iris.target, C=666)
    ests = (StandardScaler(), LinearSVC())
    params = {'linearsvc__C': np.logspace(-3, 3)}

    exp = GridSearchCV(
            make_pipeline(StandardScaler(), LinearSVC()),
            param_grid=params
            ).fit(iris.data, iris.target)
    res = dask_grid_search(ests, params, iris.data, iris.target)

    assert_array_almost_equal(res, exp)


def test_simple_pipe_fit():
    ests = (StandardScaler(), LinearSVC())
    res = simple_pipe_fit(ests, {}, iris.data, iris.target)
    exp = LinearSVC().fit(
            StandardScaler().fit_transform(iris.data, iris.target),
            iris.target)
    eq(res.compute(), exp)


def test_fit_transform():
    a = fit_transform(StandardScaler(), iris.data, iris.target)
    b = StandardScaler().fit_transform(iris.data, iris.target)
    assert_array_almost_equal(a, b)

    a = fit_transform(LinearSVC(), iris.data, iris.target)
    b = LinearSVC().fit_transform(iris.data, iris.target)
    assert_array_almost_equal(a, b)


def test_amueller():
    param_grid = {'linearsvc__C': np.logspace(-3, 3)}
    #pipe = make_pipeline(StandardScaler(), LinearSVC())
    grid = GridSearchCV(make_pipeline(StandardScaler(), LinearSVC()), param_grid=param_grid)
    grid.fit(iris.data, iris.target)

