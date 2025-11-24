import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("sharkattack_numerico.csv", sep=',', on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.strip()

df_num = df[[
    "Year",
    "Age",
    "Time",
    "outlier",
    "Type",
    "Country_group",
    "Activity_group",
    "Fatal (Y/N)"
]]

y = df_num["Fatal (Y/N)"]
X = df_num.drop(columns=["Fatal (Y/N)"])

X_sm = sm.add_constant(X)

modelo = sm.Logit(y, X_sm)
resultado = modelo.fit()

print(resultado.summary())

df_num.corr()["Fatal (Y/N)"].sort_values(ascending=False)

df_num.groupby("Country_group")["Fatal (Y/N)"].mean()
df_num.groupby("Activity_group")["Fatal (Y/N)"].mean()