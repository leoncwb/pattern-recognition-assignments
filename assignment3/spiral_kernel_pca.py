import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA

# ================================
# QUESTÃO 2 – Espiral de Arquimedes
# ================================

a = 0.1
theta = np.arange(0.5, 2.05*np.pi, 0.2)
r = a * theta
x = r * np.cos(theta)
y = r * np.sin(theta)

# 11 camadas em z
z_values = np.linspace(-1, 1, 11)
spiral_points = []
spiral_labels = []

for z in z_values:
    for i in range(len(theta)):
        spiral_points.append([x[i], y[i], z])
        spiral_labels.append(i)  # identifica pontos iguais em diferentes z

spiral_points = np.array(spiral_points)
spiral_labels = np.array(spiral_labels)

# ---- Plot 3D da espiral
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(spiral_points[:,0], spiral_points[:,1], spiral_points[:,2],
                     c=spiral_labels, cmap='hsv', s=20)
ax.set_title("Questão 2 - Espiral 3D de Arquimedes")
plt.show()

# ---- PCA linear
pca = PCA(n_components=2)
spiral_pca = pca.fit_transform(spiral_points)

plt.figure(figsize=(8,6))
plt.scatter(spiral_pca[:,0], spiral_pca[:,1], c=spiral_labels, cmap='hsv', s=20)
plt.title("Questão 2 - Projeção via PCA Linear (2D)")
plt.show()

# ---- Kernel PCA (RBF)
for gamma in [0.01, 0.1, 1, 5]:
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
    spiral_kpca = kpca.fit_transform(spiral_points)

    plt.figure(figsize=(8,6))
    plt.scatter(spiral_kpca[:,0], spiral_kpca[:,1], c=spiral_labels, cmap='hsv', s=20)
    plt.title(f"Questão 2 - Kernel PCA (RBF, gamma={gamma})")
    plt.show()
