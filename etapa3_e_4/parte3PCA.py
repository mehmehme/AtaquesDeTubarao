import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("sharkattack_numerico.csv", sep=',', on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.strip()
df = df.dropna()


#  Padronização
scaler = StandardScaler()
scaled_all = scaler.fit_transform(df)


# PCA com todos

pca = PCA(n_components=0.7)
pca_all = pca.fit_transform(scaled_all)

plt.figure(figsize=(7,5))
plt.scatter(pca_all[:, 0], pca_all[:, 1], alpha=0.6)
plt.title("PCA - Todas as Variáveis")
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.show()

print("Variância explicada:", pca.explained_variance_ratio_)


# PCA colorido por tipo de ataque

plt.figure(figsize=(7,5))
plt.scatter(pca_all[:, 0], pca_all[:, 1],
        c=df['Type'],
        cmap='viridis',
        alpha=0.7)
plt.title("PCA Colorido por 'Type'")
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.colorbar(label="Type")
plt.show()


# Método do Cotovelo (K-Means)

silhuetas = []
ks = range(2, 11)

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_all)
    score = silhouette_score(scaled_all, labels)
    silhuetas.append(score)

plt.figure(figsize=(7,5))
plt.plot(ks, silhuetas, marker='o')
plt.title("Silhouette Score por Número de Clusters")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()


# Escolhe o melhor K 
best_k = 2  
print("Melhor K encontrado:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(scaled_all)

df['cluster'] = clusters

plt.figure(figsize=(7,5))
plt.scatter(
    pca_all[:, 0],
    pca_all[:, 1],
    c=df['cluster'],
    cmap='Set1',
    alpha=0.7
)

plt.title(f"PCA com K-Means (K={best_k}) – Usando Subgrupos")
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.colorbar(label="Cluster")
plt.show()

print("Clusters criados e visualizados com sucesso!")