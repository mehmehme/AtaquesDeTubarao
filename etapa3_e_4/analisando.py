import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.metrics import classification_report


df = pd.read_csv("sharkattack_numerico.csv")
target = "Fatal (Y/N)"   


X = df.drop(columns=[target])  
y = df[target]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


acc = accuracy_score(y_test, y_pred)

print("Acurácia:", round(acc, 3))

X = df.drop(columns=["Fatal (Y/N)"])


X = sm.add_constant(X)


model = sm.Logit(y, X)


result = model.fit()

print(result.summary())

if acc >= 0.90:
    print("Modelo excelente ")
elif acc >= 0.75:
    print("Modelo muito bom ")
elif acc >= 0.60:
    print("Modelo aceitável ")
else:
    print("Modelo fraco ")


report = classification_report(y_test, y_pred, output_dict=True)


df_report = pd.DataFrame(report).transpose()

print(df_report.index)

df_report = df_report.rename(index={
    "0.0": "fatal",
    "1.0": "acidental",
     0.0: "fatal",
     1.0: "acidental",
    "accuracy": "acurácia"
})


linhas_desejadas = ["fatal", "acidental", "acurácia", "macro avg", "weighted avg"]
colunas_desejadas = ["precision", "recall", "f1-score", "support"]

linhas = [l for l in linhas_desejadas if l in df_report.index]
colunas = [c for c in colunas_desejadas if c in df_report.columns]

df_final = df_report.loc[linhas, colunas]


df_final = df_final.rename(columns={
    "precision": "precisão",
    "recall": "recall",
    "f1-score": "F1",
    "support": "suporte"
})

print("\n=== Tabela de Métricas ===\n")
print(df_final.round(5))