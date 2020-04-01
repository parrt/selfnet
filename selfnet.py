import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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

def fit(X,y,order=2):
    "Pass X=[x1,x2] to get nth order polynomial with interaction terms"
    poly = PolynomialFeatures(order, interaction_only=False)
    # model = make_pipeline(poly, Ridge(normalize=True))
    # model = make_pipeline(poly, LinearRegression(normalize=False))
    model = make_pipeline(poly, Lasso(normalize=True))
    model.fit(X, y)
    # lr = model.named_steps['ridge']
    # lr = model.named_steps['linearregression']
    lr = model.named_steps['lasso']
    # y_pred = model.predict(x)
    # ax.plot(x, y_pred, ':', c='k', lw=.7)
    terms = poly.get_feature_names()
    terms[0] = 'c'
#     terms = reversed(terms)
    terms = [f'{c:.4f}{t}' for c, t in zip(lr.coef_, terms)]
#     print(list(zip(ridge.coef_, terms)))
    eqn = ' + '.join( terms )
    return model, eqn


def generation(X_train, X_test, y_train, y_test, k = 5):
#     features = X_train.columns.values
    allpairs = list(itertools.combinations(range(X_train.shape[1]), 2))

    models = []
    eqns = []
    pairs = []
    r2_trains, mae_trains = [], []
    r2_tests, mae_tests = [], []
    for j,pair in enumerate(allpairs):
#         feats = features[[pair[0],pair[1]]]
        feats = (pair[0],pair[1])
        pairs.append(feats)
    #     print(pair)
        model, eqn = fit(X_train[:,feats], y_train)
        models.append(model)
        eqns.append(eqn)
        y_pred = model.predict(X_train[:,feats])
        r2_train, mae_train = r2_score(y_train, y_pred), mean_absolute_error(y_train, y_pred)
        r2_trains.append(r2_train)
        mae_trains.append(mae_train)
        y_pred = model.predict(X_test[:,feats])
        r2_test, mae_test = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
        r2_tests.append(r2_test)
        mae_tests.append(mae_test)

    pairs = np.array(pairs)
    eqns = np.array(eqns)
    r2_tests = np.array(r2_tests)
    mae_tests = np.array(mae_tests)
    r2_trains = np.array(r2_trains)
    mae_trains = np.array(mae_trains)

    # pick best k per generation according to R^2 of tests
    # best_idx = np.array(list(reversed(np.argsort(r2_tests))))
    best_idx = np.array(list(reversed(np.argsort(r2_trains))))
    best_idx = best_idx[:k]
    pairs = pairs[best_idx]
    models = [models[m] for m in best_idx]

    info = list(zip(eqns[best_idx], pairs, r2_trains[best_idx],
                    mae_trains[best_idx], r2_tests[best_idx], mae_tests[best_idx]))
    for eqn, pair, r2_train, mae_train, r2, mae in info:
        print(pair, f'{r2_train:.3f}, {mae_train:.3f}, {r2:.3f}, {mae:.3f}', eqn)

    output = np.empty(shape=(X_train.shape[0], k))
    for j,model in enumerate(models):
#         pairs[]
        feats = pairs[j]
        # print(model)
        output[:,j] = model.predict(X_train[:,feats])

    return pairs, models, output


class SelfNet:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, X_test, y_train, y_test, metric=mean_absolute_error):
        self.generations_models = []
        self.generation_pairs = []
        pairs, models, output = generation(X_train.values, X_test.values, y_train.values,
                                           y_test.values, k=self.k)
        self.generations_models.append(models)
        self.generation_pairs.append(pairs)
        print(output)

        pairs, models, output = generation(output, X_test.values, y_train.values,
                                           y_test.values, k=self.k)
        self.generations_models.append(models)
        self.generation_pairs.append(pairs)
        print(output)

        pairs, models, output = generation(output, X_test.values, y_train.values,
                                           y_test.values, k=self.k)
        self.generations_models.append(models)
        self.generation_pairs.append(pairs)
        print(output)

        pairs, models, output = generation(output, X_test.values, y_train.values,
                                           y_test.values, k=self.k)
        self.generations_models.append(models)
        self.generation_pairs.append(pairs)
        print(output)
        #
        # pairs, models, output = generation(output, X_test.values, y_train.values,
        #                                    y_test.values, k=self.k)
        # self.generations_models.append(models)
        # self.generation_pairs.append(pairs)
        # print(output)
        #
        # pairs, models, output = generation(output, X_test.values, y_train.values,
        #                                    y_test.values, k=self.k)
        # self.generations_models.append(models)
        # self.generation_pairs.append(pairs)
        # print(output)
        #
        # pairs, models, output = generation(output, X_test.values, y_train.values,
        #                                    y_test.values, k=self.k)
        # self.generations_models.append(models)
        # self.generation_pairs.append(pairs)
        # print(output)

        # choose best output column, which has been ordered to first is best
        # metrics = [metric(y_train, output[:,j]) for j in range(output.shape[1])]
        # best_model = np.argmin(metrics)
        best_model = 0
        print("best index", best_model)

    def predict(self, X):
        input = X
        for pairs,models in zip(self.generation_pairs, self.generations_models):
            output = np.empty(shape=(input.shape[0], self.k))
            for j,(feats,model) in enumerate(zip(pairs, models)):
                y_pred = model.predict(input[:, feats])
                output[:, j] = y_pred
            input = output

        # last gen has best model in first position
        return output[:,0]


df = pd.read_csv("rent10k.csv")

X = df.drop('price',axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

net = SelfNet(k=10)
net.fit(X_train, X_test, y_train, y_test)

y_pred = net.predict(X_train.values)
print(r2_score(y_train, y_pred), mean_absolute_error(y_train, y_pred))

y_pred = net.predict(X_test.values)
print(r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred))