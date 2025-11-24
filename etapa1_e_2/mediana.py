import pandas as pd
import numpy as np


df = pd.read_csv("sharkattack_tratado.csv", sep=',', on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.strip()
mediana_age = df['Age'].median()
mediana_year = df['Year'].median()
mediana_time = df['Time'].median()

print(mediana_age, mediana_year, mediana_time)

variancia_age = df['Age'].var()
variancia_year = df['Year'].var()
variancia_time = df['Time'].var()

print(variancia_age, variancia_year, variancia_time)