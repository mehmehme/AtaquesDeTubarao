import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("sharkattack_tratado.csv", sep=',', on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.strip()

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

america = ["USA", "Mexico", "Brazil", "Bahamas", "Cuba", "Canada", "Argentina", "Colombia"]
europa = ["Spain", "UK", "France", "Italy", "Portugal", "Greece"]
asia = ["Japan", "China", "India", "Indonesia", "Thailand"]
oceania = ["Australia", "New Zealand"]
africa = ["South Africa", "Egypt", "Mozambique"]

def categorize_country(country):
    if country in america:
        return 0
    elif country in europa:
        return 1
    elif country in asia:
        return 2
    elif country in oceania:
        return 3
    elif country in africa:
        return 4
    else:
        return 5

df["Country_group"] = df["Country"].apply(categorize_country)

#atividades
def categorize_activity(a):
    a = str(a).lower()
    if any(x in a for x in ["surf", "board", "bodyboard", "windsurf"]):
        return 0
    if any(x in a for x in ["swim", "bath", "wading","standing"]):
        return 1
    if any(x in a for x in ["fish", "fishing", "spearfish"]):
        return 2
    return 3

df["Activity_group"] = df["Activity"].apply(categorize_activity)
df["Country_group"] = df["Country"].apply(categorize_country)

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

df_num.to_csv("sharkattack_numerico.csv", index=False, encoding="utf-8")