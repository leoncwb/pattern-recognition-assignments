import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Carregar o dataset
column_names = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]
wine_df = pd.read_csv("wine.data", header=None, names=column_names)

# Função para plotar a fronteira de decisão
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k')
    plt.xlabel(title.split(":")[1].split(" e ")[0])
    plt.ylabel(title.split(" e ")[1])
    plt.title(title)
    plt.grid(True)
    plt.show()

# Lista de pares de features a serem testadas
feature_pairs = [
    ['Alcohol', 'Hue'],
    ['Flavanoids', 'OD280/OD315 of diluted wines'],
    ['Color intensity', 'Hue'],
    ['Malic acid', 'Ash']
]

for pair in feature_pairs:
    X = wine_df[pair].values
    y = wine_df['Class'].values

    # Selecionar apenas as classes 1 e 2
    mask = y != 3
    X = X[mask]
    y = y[mask]
    y = np.where(y == 1, -1, 1)

    # Padronizar
    X_std = StandardScaler().fit_transform(X)

    # Modelo ADALINE com SGDClassifier
    model = SGDClassifier(loss='squared_error', learning_rate='constant', eta0=0.01,
                          max_iter=1000, tol=1e-3, random_state=1)
    model.fit(X_std, y)

    # Plot
    plot_decision_boundary(X_std, y, model, f"ADALINE com features: {pair[0]} e {pair[1]}")