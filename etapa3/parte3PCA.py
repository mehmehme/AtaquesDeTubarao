import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches

df = pd.read_csv("sharkattack_tratado.csv", sep=',', on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.strip()

cols_to_drop = [
    'Date', 'Time', 'Investigator or Source', 'pdf','Name','Location',
    'href', 'Case Number','Case Number.1',
    'original order','href formula'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

df = df.dropna()

# 1. LabelEncoder para todas

for col in df.columns:
    if df[col].dtype == 'object':
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str))


# 2. Padronização

scaler = StandardScaler()
scaled_all = scaler.fit_transform(df)


# 3. PCA com todos os grupos

pca = PCA(n_components=2)
pca_all = pca.fit_transform(scaled_all)

plt.figure(figsize=(7,5))
plt.scatter(pca_all[:, 0], pca_all[:, 1], alpha=0.6)
plt.title("PCA - Todas as Variáveis (Numéricas + Categóricas + Textuais)")
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.show()

print("Variância explicada:", pca.explained_variance_ratio_)


# 5. PCA colorido por tipo de ataque

enc_type = LabelEncoder()
df["Type_encoded"] = enc_type.fit_transform(df["Type"].astype(str))

print(dict(zip(enc_type.classes_, enc_type.transform(enc_type.classes_))))

if 'Type' in df.columns:
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


# 6. Método do Cotovelo (K-Means)

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_all)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(K_range, inertia, marker='o')
plt.title("Método do Cotovelo (Elbow Method)")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Inércia")
plt.grid(True)
plt.show()


# 7. Escolhe o melhor K, nesse caso ficou entre 4 e 6 

best_k = 5  
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(scaled_all)

df['cluster'] = clusters

plt.figure(figsize=(7,5))
plt.scatter(pca_all[:, 0], pca_all[:, 1],
            c=df['cluster'],
            cmap='Set1',
            alpha=0.7)

plt.title(f"PCA com K-Means (K={best_k})")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label="Cluster")
plt.show()

print("Clusters criados e visualizados com sucesso!")