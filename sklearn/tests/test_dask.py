import numpy as np
from numpy.testing import assert_array_almost_equal
from dask.imperative import value
from sklearn.svm import LinearSVC
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
    pass


def test_simple_pipe_fit():
    ests = (StandardScaler(), LinearSVC())
    res = simple_pipe_fit(ests, {}, iris.data, iris.target)
    exp = LinearSVC().fit_transform(
            StandardScaler().fit_transform(iris.data, iris.target),
            iris.target)
    assert_array_almost_equal(res.compute(), exp)


def test_fit_transform():
    a = fit_transform(StandardScaler(), iris.data, iris.target)
    b = StandardScaler().fit_transform(iris.data, iris.target)
    assert_array_almost_equal(a, b)

    a = fit_transform(LinearSVC(), iris.data, iris.target)
    b = LinearSVC().fit_transform(iris.data, iris.target)
    assert_array_almost_equal(a, b)
