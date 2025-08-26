# ============================================================
# Assignment 5 - Clustering Analysis
# Pattern Recognition - André E. Lazzaretti
# Autor: Leonardo Rodrigues Pereira
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------------------------------------
# Função para reduzir a dimensionalidade e visualizar clusters
# ------------------------------------------------------------
def plot_clusters(X, labels, title="Clusters"):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=X_reduced[:,0], y=X_reduced[:,1], hue=labels, palette="Set1", s=50)
    plt.title(title)
    plt.show()

# ------------------------------------------------------------
# K-Means com Elbow Method e Silhouette
# ------------------------------------------------------------
def kmeans_analysis(X, max_k=10):
    inertia = []
    silhouette = []
    Ks = range(2, max_k+1)

    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X, labels))

    # Elbow Method
    plt.plot(Ks, inertia, 'o-')
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inércia (WCSS)")
    plt.title("Método do Cotovelo (Elbow)")
    plt.show()

    # Silhouette
    plt.plot(Ks, silhouette, 'o-')
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Coeficiente de Silhouette")
    plt.show()

    # Melhor k (com base no Silhouette)
    best_k = Ks[np.argmax(silhouette)]
    print(f"Melhor k pelo Silhouette: {best_k}")

    final_model = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_model.fit_predict(X)
    plot_clusters(X, final_labels, f"K-Means com k={best_k}")
    print("Centros encontrados pelo K-Means:")
    print(final_model.cluster_centers_)
    return final_labels

# ------------------------------------------------------------
# Clustering Hierárquico com dendrograma
# ------------------------------------------------------------
def hierarchical_analysis(X, n_clusters=3):
    # Dendrograma
    linked = linkage(X, method='ward')
    plt.figure(figsize=(8,4))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title("Dendrograma - Método Ward")
    plt.show()

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    plot_clusters(X, labels, f"Hierárquico ({n_clusters} clusters)")
    return labels

# ------------------------------------------------------------
# DBSCAN
# ------------------------------------------------------------
def dbscan_analysis(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    plot_clusters(X, labels, f"DBSCAN eps={eps}")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Número de clusters encontrados pelo DBSCAN: {n_clusters}")
    return labels

# ------------------------------------------------------------
# Execução principal - ler todos os datasets TXT
# ------------------------------------------------------------
def main():
    data_path = "/home/leonardo/Downloads/Dados"   # <--- sua pasta com os TXT

    print("Arquivos encontrados:", os.listdir(data_path))

    # Processar cada dataset TXT
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            print(f"\n=== Analisando {file} ===")
            df = pd.read_csv(
                os.path.join(data_path, file),
                sep=r"\s+",       # separador por espaço/tabulação
                engine="python",
                header=None       # assume que não tem cabeçalho
            )
            df = df.dropna()      # remove linhas com valores faltantes
            X = df.values

            print("\n--- K-Means ---")
            kmeans_labels = kmeans_analysis(X)

            print("\n--- Hierárquico ---")
            hier_labels = hierarchical_analysis(X, n_clusters=3)  # ajuste se necessário

            print("\n--- DBSCAN ---")
            db_labels = dbscan_analysis(X, eps=0.5, min_samples=5)

# ------------------------------------------------------------
# Rodar
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
