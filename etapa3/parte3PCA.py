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
    'Date', 'Sex', 'Investigator or Source', 'pdf','Name','Location',
    'href', 'Case Number','Case Number.1',
    'original order','href formula','Species','outlier','Area'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

df = df.dropna()

#paises microgrupo
america = ["USA", "Mexico", "Brazil", "Bahamas", "Cuba", "Canada", "Argentina", "Colombia"]
europa = ["Spain", "UK", "France", "Italy", "Portugal", "Greece"]
asia = ["Japan", "China", "India", "Indonesia", "Thailand"]
oceania = ["Australia", "New Zealand"]
africa = ["South Africa", "Egypt", "Mozambique"]

def categorize_country(country):
    if country in america:
        return "America"
    elif country in europa:
        return "Europe"
    elif country in asia:
        return "Asia"
    elif country in oceania:
        return "Oceania"
    elif country in africa:
        return "Africa"
    else:
        return "Other"

df["Country_group"] = df["Country"].apply(categorize_country)

#atividades
def categorize_activity(a):
    a = str(a).lower()
    if any(x in a for x in ["surf", "board", "bodyboard", "windsurf"]):
        return "Athlete"
    if any(x in a for x in ["swim", "bath", "wading","standing"]):
        return "Beach"
    if any(x in a for x in ["fish", "fishing", "spearfish"]):
        return "Fishing"
    return "Other"

df["Activity_group"] = df["Activity"].apply(categorize_activity)

#idades
def categorize_age(age):
    age = float(age)
    if age < 18:
        return "Young"
    if age < 60:
        return "Adult"
    return "Elderly"

df["Age_group"] = df["Age"].apply(categorize_age)

#anos
def categorize_year(y):
    y = int(y)
    if 1800 <= y <= 1900:
        return "1800-1900"
    if 1901 <= y <= 2000:
        return "1901-2000"
    if 2001 <= y <= 2023:
        return "2001-2023"
    return "Other"

df["Year_group"] = df["Year"].apply(categorize_year)

#hora
def categorize_time_numeric(h):
    h = float(h)

    if 5 <= h < 12:
        return "Morning"     # manhã
    if 12 <= h < 18:
        return "Afternoon"   # tarde
    if 18 <= h <= 23:
        return "Night"       # noite
    if 0 <= h < 5:
        return "Dawn"        # madrugada

    return "Unknown"

df["Time_group"] = df["Time"].apply(categorize_time_numeric)

# 1. LabelEncoder para todas

for col in df.columns:
    if df[col].dtype == object:
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
plt.title("PCA - Todas as Variáveis + SubGrupos")
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.show()

print("Variância explicada:", pca.explained_variance_ratio_)


# 5. PCA colorido por tipo de ataque

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


# 7. Escolhe o melhor K

best_k = 10 
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(scaled_all)

df['cluster'] = clusters

plt.figure(figsize=(7,5))
plt.scatter(pca_all[:, 0], pca_all[:, 1],
            c=df['cluster'],
            cmap='Set1',
            alpha=0.7)

plt.title(f"PCA com K-Means (K={best_k})")
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.colorbar(label="Cluster")
plt.show()

print("Clusters criados e visualizados com sucesso!")