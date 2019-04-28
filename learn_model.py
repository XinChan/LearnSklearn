from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target


model = LinearRegression()
model.fit(data_X, data_y)

# print(model.predict(data_X[:4,:]))
# print(data_y[:4])

# print(model.coef_) # y = 0.1x + 0.3
# print(model.intercept_)
# [-1.08011358e-01  4.64204584e-02  2.05586264e-02  2.68673382e+00
#  -1.77666112e+01  3.80986521e+00  6.92224640e-04 -1.47556685e+00
#   3.06049479e-01 -1.23345939e-02 -9.52747232e-01  9.31168327e-03
#  -5.24758378e-01]
# 36.459488385090125

# print(model.get_params())
# {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}

# print(model.score(data_X,data_y)) # R^2 coefficient of determination
# 0.7406426641094095