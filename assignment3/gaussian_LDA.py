import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ================================
# QUESTÃO 1 – Gaussian Dataset + LDA (9 classes)
# ================================

np.random.seed(42)  # reprodutibilidade

# Classe 1: 100 vetores
mean1 = np.zeros(3)
cov1 = np.array([[0.5, 0, 0],
                 [0, 0.5, 0],
                 [0, 0, 0.01]])
class1 = np.random.multivariate_normal(mean1, cov1, 100)

# Classe 2: 8 grupos de 100 vetores cada (cada grupo tratado como uma classe separada)
a = 20  # valor corrigido no enunciado
means_class2 = [
    [a, 0, 0], [-a, 0, 0],
    [0, a, 0], [0, -a, 0],
    [a, a, 0], [a, -a, 0],
    [-a, a, 0], [-a, -a, 0]
]
cov2 = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0.01]])

# Gera os 8 grupos
groups = [np.random.multivariate_normal(mean, cov2, 100) for mean in means_class2]
class2 = np.vstack(groups)

# Dataset final
X = np.vstack((class1, class2))

# Labels: classe 0 para a primeira, 1..8 para os grupos
y = np.array([0]*100 + [i+1 for i in range(8) for _ in range(100)])

# ---- Plot 3D do dataset (cores diferentes para cada classe)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10(np.linspace(0,1,9))  # 9 cores diferentes
for i in range(9):
    ax.scatter(X[y==i,0], X[y==i,1], X[y==i,2], 
               color=colors[i], label=f'Classe {i}')

ax.set_title("Questão 1 - Dataset em 3D (9 classes)")
ax.legend()
plt.show()

# ---- LDA com 2 componentes
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.figure(figsize=(8,6))
for i in range(9):
    plt.scatter(X_lda[y==i,0], X_lda[y==i,1], 
                color=colors[i], label=f'Classe {i}', alpha=0.7)

plt.title("Questão 1 - Projeção via LDA (2D, 9 classes)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()
