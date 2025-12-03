import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("shark_attack.csv", sep=',', on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.strip()

cols_to_drop = [
    'Injury','Date', 'Species', 'Investigator or Source', 'pdf', 'Name', 'Location', 'Sex',
    'href', 'Case Number','Case Number.1', 'original order','href formula'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

num_cols = ['Year', 'Age','Time']
cat_cols = ['Fatal (Y/N)', 'Type ', 'Activity', 'Country', 'Area']

print("\n--- NULOS INICIAIS ---")
null_before = df.isnull().sum()
pct_before = (df.isnull().mean() * 100)

print(pd.DataFrame({
    "Nulos": null_before,
    "% Nulos": pct_before.round(2)
}))

#removendo 'nan'
df['Activity'] = df['Activity'].replace("nan", np.nan)
df.dropna(subset=['Activity'], inplace=True)

#tratando nulos no categórico
cat_impute_cols = ['Activity', 'Country', 'Area']

encoders = {}
for col in cat_impute_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str) 
    df[col] = df[col].dropna()  
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

imputer = KNNImputer(n_neighbors=5)
df[cat_impute_cols] = imputer.fit_transform(df[cat_impute_cols])

for col in cat_impute_cols:
    le = encoders[col]
    df[col] = df[col].round().astype(int)   
    df[col] = le.inverse_transform(df[col])

#tratando hora
def clean_time(value):
    """
    Converte formatos para apenas a HORA inteira (ex: 13, 7, 15).
    """
    if pd.isna(value):
        return np.nan
    
    v = str(value).lower().strip()


    if 'h' in v:
        partes = v.split('h')
        if partes[0].isdigit():
            return int(partes[0])

 
    if ':' in v:
        partes = v.split(':')
        if partes[0].isdigit():
            return int(partes[0])

    return np.nan

if "Time" in df.columns:
    df["Time"] = df["Time"].apply(clean_time)
else:
    df["Time"] = np.nan

# Transformar Year e Age
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[(df['Year'] >= 1800) & (df['Year'] <= 2025)]

# Preencher numéricos
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Year'].fillna(df['Year'].median(), inplace=True)
df['Time'].fillna(df['Time'].median(), inplace=True)

# Ajustes Fatal

df['Fatal (Y/N)'] = df['Fatal (Y/N)'].astype(str).str.strip().str.upper()
df['Fatal (Y/N)'] = df['Fatal (Y/N)'].map({'Y': 1, 'N': 0})

#para os nulos
df.dropna(subset=['Fatal (Y/N)'], inplace=True)

# TYPE → provoked, unprovoked, questionable
def classify_type(value):
    v = str(value).strip().lower()
    if 'provok' in v:
        return 0
    elif 'unprovok' in v:
        return 1
    elif 'question' in v:
        return 2
    else:
        return np.nan

if 'Type' in df.columns:
    df['Type'] = df['Type'].apply(classify_type)
else:
    df['Type'] = np.nan

df['Type'].fillna(1, inplace=True) 

#tratando atividade
def pegar_primeira_palavra(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    
    # divide por espaço
    primeira = x.split()[0]
    return primeira

df['Activity'] = df['Activity'].apply(pegar_primeira_palavra)

#nulos no final
print("\n--- NULOS FINAIS ---")
null_after = df.isnull().sum()
pct_after = (df.isnull().mean() * 100)

print(pd.DataFrame({
    "Nulos": null_after,
    "% Nulos": pct_after.round(2)
}))


# Normalidade + Plots
for col in num_cols:
    plt.figure(figsize=(14,4))

    plt.subplot(1,3,1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histograma - {col}')

    plt.subplot(1,3,2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot - {col}')

    plt.subplot(1,3,3)
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot - {col}')
    plt.show()

    stat, p = stats.shapiro(df[col].dropna())
    print(f"{col}: p-valor={p:.4f} -> {'Normal' if p>0.05 else 'Não normal'}")

# Outliers
iso = IsolationForest(contamination=0.05, random_state=42)
df['outlier'] = iso.fit_predict(df[num_cols])

outliers = df[df['outlier'] == -1]
print("Qtd. de outliers detectados:", outliers.shape[0])

# Plots de outliers
plt.figure(figsize=(6,4))
sns.boxplot(x=df[col])
plt.title(f"Outliers - {col}")
plt.show()

# Missing
missing = df.isnull().sum()
print("Valores faltosos por coluna:\n", missing[missing>0])


print("Colunas finais:", df.columns.tolist())
print("Dimensão final:", df.shape)
print(df.head())

# Salvar
df.to_csv("sharkattack_tratado.csv", index=False, encoding='utf-8')
print("\nArquivo 'sharkattack_tratado.csv' criado com sucesso!")

# Estatísticas
desc = df[num_cols].describe(include='all').T
desc['moda'] = [df[c].mode()[0] for c in num_cols]
desc['valores_distintos'] = [df[c].nunique() for c in num_cols]
desc['mediana'] = [df[c].median() for c in num_cols]
desc['variancia'] = [df[c].var() for c in num_cols]
print(desc)
