import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import IsolationForest

df = pd.read_csv("globalsharkattack.csv",sep=';', on_bad_lines="skip",encoding= "utf-8")

print(df.columns)
df.columns = df.columns.str.strip()#espaços extras ('Sex ')

cols_to_drop = [
    'Date', 'Time', 'Investigator or Source', 'pdf',
    'href', 'Case Number','Case Number.1', 'original order','href formula'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

num_cols = ['Year', 'Age']
cat_cols = ['Sex', 'Fatal (Y/N)','Type ']
log_cols = ['Activity', 'Country', 'Species','Injury', 'Area']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Year'].fillna(df['Year'].median(), inplace=True)

# masculino ou feminino = 1 e 0
df['Sex'] = df['Sex'].str.strip().str.upper()
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

#sim ou não = 1 e 0
df['Fatal (Y/N)'] = df['Fatal (Y/N)'].str.strip().str.upper()
df['Fatal (Y/N)'] = df['Fatal (Y/N)'].map({'Y': 1, 'N': 0})

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

# substitui valores nulos de Type por 1 (mais comum: unprovoked)
df['Type'].fillna(1, inplace=True)

#em caso de estar null
df['Sex'].fillna(0, inplace=True)
df['Fatal (Y/N)'].fillna(0, inplace=True)

for col in log_cols:
    df[col] = df[col].fillna("Unknown").str.strip().str.lower()

print("Colunas finais:", df.columns.tolist())
print("Dimensão final:", df.shape)
print(df.head())

#salva dataset tratado
#df.to_csv("globalsharkattack_tratado.csv", index=False, encoding='utf-8')
#print("\n Arquivo 'globalsharkattack_tratado.csv' criado com sucesso!")

desc = df[num_cols].describe(include='all').T
desc['moda'] = [df[c].mode()[0] for c in num_cols]
desc['valores_distintos'] = [df[c].nunique() for c in num_cols]
print(desc)

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

    # Teste de normalidade (Shapiro)
    stat, p = stats.shapiro(df[col].dropna())
    print(f"{col}: p-valor={p:.4f} -> {'Normal' if p>0.05 else 'Não normal'}")
    
iso = IsolationForest(contamination=0.05, random_state=42)
df['outlier'] = iso.fit_predict(df[num_cols])

# -1 = outlier
outliers = df[df['outlier'] == -1]
print("Qtd. de outliers detectados:", outliers.shape[0])
sns.boxplot(data=df[num_cols])
plt.title("Detecção de Outliers (Isolation Forest)")
plt.show()

missing = df.isnull().sum()
print("Valores faltosos por coluna:\n", missing[missing>0])

# exemplo de análise
for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"\nAssociação de valores nulos em {col} com Fatal(Y/N):")
        print(df.groupby(df['Fatal (Y/N)'])[col].apply(lambda x: x.isnull().sum()))