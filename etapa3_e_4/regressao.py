from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

df = pd.read_csv("sharkattack_numerico.csv", sep=',', on_bad_lines="skip", encoding="utf-8")
df.columns = df.columns.str.strip()

def avaliar_regressao_completa(df, target):
    df_model = df.copy()

    # remove linhas com target nulo
    df_model = df_model[df_model[target].notna()]

    # separa X e y
    y = df_model[target]
    X = df_model.drop(columns=[target])

    # transforma texto para número
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # separa treino/validação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n Regressão para prever: {target}")
    print(f"R²:  {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE:{rmse:.3f}")

    if r2 > 0.8:
        print("Modelo excelente ")
    elif r2 > 0.6:
        print("Modelo bom ")
    elif r2 > 0.4:
        print("Modelo ok ")
    else:
        print("Modelo fraco ")

def avaliar_classificacao_completa(df, target):
    df_model = df.copy()

    df_model = df_model[df_model[target].notna()]

    y = df_model[target]
    X = df_model.drop(columns=[target])

    # transforma tudo em número
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n Classificação para prever: {target}")
    print(f"Acurácia: {acc:.3f}")

    if acc > 0.9:
        print("Classificação perfeita")
    elif acc > 0.75:
        print("Muito boa")
    elif acc > 0.6:
        print("Aceitável")
    else:
        print("Ruim")
        
    
    
avaliar_regressao_completa(df, "Age")
avaliar_regressao_completa(df, "Year")
avaliar_regressao_completa(df, "Time")

avaliar_classificacao_completa(df, "Type")
avaliar_classificacao_completa(df, "Fatal (Y/N)")
avaliar_classificacao_completa(df, "Activity")
avaliar_classificacao_completa(df, "Country")
