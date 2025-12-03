import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("sharkattack_tratado.csv", sep=',', on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.strip()

df["Country"] = df["Country"].str.upper().str.strip()
le_country = LabelEncoder()
df["Country"] = le_country.fit_transform(df["Country"])

le = LabelEncoder()
df['Activity'] = le.fit_transform(df['Activity'].astype(str))

df_num = df[[
    "Year",
    "Age",
    "Time",
    "Type",
    "Country",
    "Activity",
    "Fatal (Y/N)"
]]

df_num.to_csv("sharkattack_numerico.csv", index=False, encoding="utf-8")