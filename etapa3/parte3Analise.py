import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

df = pd.read_csv("globalsharkattack_tratado.csv",sep=',', on_bad_lines="skip",encoding= "utf-8")

corr = df[['Year', 'Age']].corr(method='pearson')
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title("Correlação entre variáveis numéricas")
plt.show()

scaler = StandardScaler()
scaled = scaler.fit_transform(df[['Year', 'Age']].dropna())

# Aplica PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled)

# Visualiza
plt.scatter(pca_data[:,0], pca_data[:,1], alpha=0.6)
plt.title("Visualização PCA - Year x Age")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()

# Variância explicada
print("Variância explicada por cada componente:", pca.explained_variance_ratio_)

plt.scatter(pca_data[:,0], pca_data[:,1], c=df['Type'], cmap='viridis', alpha=0.6)
plt.title("PCA colorido por tipo de ataque")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()

inercia = []
X = df[['Year', 'Age']].dropna()

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inercia.append(kmeans.inertia_)

plt.plot(range(1, 11), inercia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.show()

best_k = 6  
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

plt.scatter(X['Year'], X['Age'], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.title("Grupos encontrados pelo K-Means")
plt.xlabel("Year")
plt.ylabel("Age")
plt.show()