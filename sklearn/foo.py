import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import make_pipeline

iris = load_iris()
param_grid = {'linearsvc__C': np.logspace(-3, 3)}
#pipe = make_pipeline(StandardScaler(), LinearSVC())
grid = GridSearchCV(make_pipeline(StandardScaler(), LinearSVC()), param_grid=param_grid)
grid.fit(iris.data, iris.target)

