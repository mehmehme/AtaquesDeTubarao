import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

df = pd.read_csv("shark_attack.csv", sep=',', on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.strip()

cols_to_drop = [
    'Date', 'Time', 'Investigator or Source', 'pdf','Name','Location',
    'href', 'Case Number','Case Number.1',
    'original order','href formula'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

numericas = ['Year', 'Age']
categoricas = ['Sex', 'Fatal (Y/N)', 'Type']
textuais = ['Country', 'Area', 'Injury', 'Species']

# Funções de imputação


def imputar_regressao_linear(df, target):
    df_temp = df.select_dtypes(include=[np.number]).copy()
    if target not in df_temp.columns:
        return df, None

    known = df_temp[df_temp[target].notnull()]
    unknown = df_temp[df_temp[target].isnull()]

    if unknown.empty or len(known) < 5:
        return df, None

    X_train = known.drop(columns=[target])
    y_train = known[target]
    X_pred = unknown.drop(columns=[target])

    X_train = X_train.fillna(X_train.mean())
    X_pred = X_pred.fillna(X_train.mean())

    model = LinearRegression()
    model.fit(X_train, y_train)
    df.loc[df[target].isnull(), target] = model.predict(X_pred)

    score = cross_val_score(model, X_train, y_train, scoring='r2', cv=5).mean()
    return df, round(score, 3)

def imputar_regressao_logistica(df, target):
    df_temp = df.copy()
    label_enc = LabelEncoder()

    for col in df_temp.select_dtypes(include=['object']).columns:
        df_temp[col] = df_temp[col].astype(str).fillna('missing')
        df_temp[col] = label_enc.fit_transform(df_temp[col])

    if target not in df_temp.columns:
        return df, None

    known = df_temp[df_temp[target].notnull()]
    unknown = df_temp[df_temp[target].isnull()]

    if unknown.empty or len(known) < 5:
        return df, None

    X_train = known.drop(columns=[target])
    y_train = known[target]
    X_pred = unknown.drop(columns=[target])

    if y_train.nunique() > len(y_train) * 0.5:
        moda = df[target].mode()[0]
        df.loc[df[target].isnull(), target] = moda
        return df, 1.0

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    df.loc[df[target].isnull(), target] = model.predict(X_pred)

    score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()
    return df, round(score, 3)


# 3. Aplicar imputações

relatorio = []

df_num = df[numericas].copy()
for col in numericas:
    df_num, score = imputar_regressao_linear(df_num, col)
    relatorio.append({'Variável': col, 'Tipo': 'Numérica', 'Métrica': 'R² médio', 'Valor': score})

df_cat = df[categoricas].copy()
for col in categoricas:
    df_cat, score = imputar_regressao_logistica(df_cat, col)
    relatorio.append({'Variável': col, 'Tipo': 'Categórica', 'Métrica': 'Acurácia média', 'Valor': score})

df_txt = df[textuais].copy()
for col in textuais:
    df_txt, score = imputar_regressao_logistica(df_txt, col)
    relatorio.append({'Variável': col, 'Tipo': 'Textual', 'Métrica': 'Acurácia média', 'Valor': score})

relatorio_df = pd.DataFrame(relatorio)
relatorio_df['Valor'] = relatorio_df['Valor'].fillna('Sem valores faltosos ou insuficientes')
relatorio_df.to_csv('relatorio_tratamento.csv', index=False)

# 4. Análises gráficas


df_final = pd.concat([df_num, df_cat, df_txt], axis=1)

num_cols = numericas
corr = df_final[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title("Correlação entre variáveis numéricas")
plt.show()

print(" Tratamento + Análises concluídos!")
