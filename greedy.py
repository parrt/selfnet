import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

class PolyModel:
    def __init__(self, order=2, model_type=LinearRegression(normalize=True)):
        self.order = order
        self.model_type = model_type

    def predict(self, X):
        return self.model.predict(X[:,self.vars])

    def fit(self, X, y, vars):
        """
        Get nth order polynomial with interaction terms; X is full X,
        vars is tuple, list, or set like (1,5) to indicate columns of X.
        """
        self.vars = list(sorted(vars))
        X = X[:,self.vars]
        poly = PolynomialFeatures(self.order)
        model = make_pipeline(poly, sklearn.base.clone(self.model_type))
        model.fit(X, y)
        lr = model.named_steps[self.model_type.__class__.__name__.lower()]
        terms = poly.get_feature_names()
        terms = [t.replace('x1','x'+str(self.vars[1])) for t in terms]
        terms = [t.replace('x0','x'+str(self.vars[0])) for t in terms]
        terms[0] = 'c' # replace the '1' with 'c'
        terms = [f'{c:.2f}{t}' for c, t in zip(lr.coef_, terms)]
        self.eqn = ' + '.join(terms)
        self.model = model

    def __str__(self):
        return self.eqn
    def __repr__(self):
        return self.__str__()


class GreedyLayer:
    def __init__(self, k=None, order=2, metric=r2_score, model_type=LinearRegression(normalize=True)):
        self.k = k # None implies all n choose k combos of input parameters
        self.order = order
        self.metric = metric
        self.model_type = model_type
        self.models = []

    def predict(self, X): # X is full matrix, each model knows its var pair
        X_output = np.empty_like(X, dtype=float)
        for j,m in enumerate(self.models):
            X_output[:,j] = m.predict(X)
        return X_output

    def fit(self, X_train, y_train):
        allvars = range(X_train.shape[1])
        if self.k is None:
            pairs = list(itertools.combinations(allvars, 2))
        else:
            pairs = np.random.choice(allvars, self.k)
        for j, pair in enumerate(pairs):
            m = PolyModel(order=self.order, model_type=self.model_type)
            m.fit(X_train, y_train, vars=pair)
            self.models.append(m)

    def __str__(self):
        return '\n'.join([str(m) for m in self.models])


class GreedyNet:
    def __init__(self, n_layers=3, k=None, order=2, metric=r2_score, model_type=LinearRegression(normalize=True)):
        self.n_layers = n_layers
        self.k = k
        self.order = order
        self.metric = metric
        self.model_type=model_type
        self.layers = [] # list of GreedyLayer

    def predict(self, X):
        X_input = X
        for layer in self.layers:
            X_output = layer.predict(X_input)
            X_input = X_output
        return X_output

    def fit(self, X_train, X_test, y_train, y_test):
        X_input = X_train
        for layer in range(self.n_layers):
            gl = GreedyLayer(k=self.k, order=self.order, model_type=self.model_type)
            gl.fit(X_input, y_train)
            self.layers.append(gl)
            Y_pred = gl.predict(X_input)
            Y_test_pred = gl.predict(X_test) # move test cases along through the layer
            print(f"LAYER {layer}\n",gl,"\nY_pred\n",Y_pred,"\nY_test_pred\n",X_test)
            for k,m in enumerate(gl.models):
                train_score = self.metric(y_train, Y_pred[:,k])
                test_score = self.metric(y_test, Y_test_pred[:,k])
                print(f"pair {m.vars}", train_score, test_score)
            X_input = Y_pred
            X_test = Y_test_pred

np.random.seed(1)
n = 20
p = 3

X = np.empty(shape=(n,p), dtype=float)
for j in range(p):
    X[:,j] = (np.random.random(size=n)*10).astype(int)

y = X[:,0] + X[:,1] + X[:,2]
print("X\n",X)
print("y\n", y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

net = GreedyNet(n_layers=5, model_type=Lasso(alpha=.01, normalize=True))
net.fit(X_train, X_test, y_train, y_test)
X_output = net.layers[0].predict(X_train)

