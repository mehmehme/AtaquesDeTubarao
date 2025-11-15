import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import IsolationForest

df = pd.read_csv("sharkattack_tratado.csv",sep=',', on_bad_lines="skip",encoding= "utf-8")

print(df.columns)
df.columns = df.columns.str.strip()#espaços extras ('Sex ')

cols_to_drop = [
    'Date', 'Time', 'Investigator or Source', 'pdf', 'Name', 'Location',
    'href', 'Case Number','Case Number.1', 'original order','href formula'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

num_cols = ['Year', 'Age']
cat_cols = ['Sex', 'Fatal (Y/N)','Type ']
log_cols = ['Activity', 'Country', 'Species','Injury', 'Area']

print("Dimensão do dataset:", df.shape)
print("\nVisualização inicial:")
print(df.head())

#Ainda há Valores nulos?
print("\nValores nulos por coluna:")
print(df.isnull().sum())


# Ainda há Linhas duplicadas?
duplicadas = df.duplicated().sum()
print(f"\nLinhas duplicadas: {duplicadas}")


#remover linhas duplicadas para garantir a qualidade dos dados
df = df.drop_duplicates()

#verificando ruidos em colunas numéricas
if 'Sex' in df.columns:
    print("\nValores únicos em Sex:", df['Sex'].unique())

if 'Fatal (Y/N)' in df.columns:
    print("Valores únicos em Fatal (Y/N):", df['Fatal (Y/N)'].unique())
    
#garantindo que não há valores fora do esperado
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 1 else 0)
df['Fatal (Y/N)'] = df['Fatal (Y/N)'].apply(lambda x: 1 if x == 1 else 0)

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

df = df[(df['Year'].notna()) & (df['Year'] > 0) & (df['Year'] <= 2025)]

# limpar idades negativas ou loucas
df = df[(df['Age'].notna()) & (df['Age'] >= 0) & (df['Age'] <= 120)]

#detectando outliers em ano e idade
def detectar_outliers(df, coluna):
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
    return outliers

for col in ['Year', 'Age']:
    if col in df.columns:
        outliers = detectar_outliers(df, col)
        print(f"\nOutliers em {col}: {len(outliers)}")
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
        plt.show()
        