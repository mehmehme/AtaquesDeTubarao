import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("sharkattack_numerico.csv", sep=',', on_bad_lines="skip", encoding="utf-8")
df.columns = df.columns.str.strip()

y = df["Fatal (Y/N)"]
X = df.drop(columns=["Fatal (Y/N)"])

#pradonizando
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=200)

#cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="accuracy")
auc_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="roc_auc")

print("\n==== Cross Validation ====")
print("Accuracy por fold:", acc_scores)
print("Accuracy média:", acc_scores.mean())

print("\nAUC por fold:", auc_scores)
print("AUC média:", auc_scores.mean())
print("==============================\n")

y_pred = cross_val_predict(model, X_scaled, y, cv=kf, method="predict")
y_proba = cross_val_predict(model, X_scaled, y, cv=kf, method="predict_proba")[:, 1]


#matriz confusa
cm = confusion_matrix(y, y_pred)
print("\nMatriz de Confusão:")
print(cm)

#métricas
print("\nRelatório de Classificação:")
print(classification_report(y, y_pred))

# auc-roc mede o quão bem o modelo separa classes indepedente do corte
roc_auc = roc_auc_score(y, y_proba)
print(f"AUC-ROC: {roc_auc:.4f}")

#Curva ROC
fpr, tpr, _ = roc_curve(y, y_proba)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Shark Fatality Classificação")
plt.grid(True)
plt.show()