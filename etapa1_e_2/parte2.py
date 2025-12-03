import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency, norm

df = pd.read_csv("sharkattack_tratado.csv")

col_year = 'Year'
col_fatal = 'Fatal (Y/N)'
col_country = 'Country'
col_activity = 'Activity'
col_type = 'Type'
col_area = 'Area'
col_time = 'Time'
col_age = 'Age'

if col_year in df.columns:
    df[col_year] = pd.to_numeric(df[col_year], errors='coerce').astype('Int64')
if col_age in df.columns:
    df[col_age] = pd.to_numeric(df[col_age], errors='coerce')
if col_time in df.columns:
    df[col_time] = pd.to_numeric(df[col_time], errors='coerce')

# Hipótese A: existe um ano que foi o mais fatal?
if col_year in df.columns and col_fatal in df.columns:
    by_year = df.groupby(col_year)[col_fatal].agg(['sum', 'count']).rename(columns={'sum': 'fatal_count', 'count': 'total'})
    by_year['fatal_rate'] = by_year['fatal_count'] / by_year['total']
    top_year = by_year['fatal_count'].idxmax()
    print(f"Hipótese A -> Ano com mais ataques fatais: {top_year}")
    print(by_year.loc[top_year])
    
    x1 = int(by_year.loc[top_year, 'fatal_count'])
    n1 = int(by_year.loc[top_year, 'total'])
    x2 = int(by_year['fatal_count'].sum()) - x1
    n2 = int(by_year['total'].sum()) - n1

    # evitar divisão por zero
    if n1 > 0 and n2 > 0:
        p1 = x1 / n1
        p2 = x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        if se > 0:
            z = (p1 - p2) / se
            pval = norm.sf(abs(z)) * 2  # two-sided
            print(f"Comparação (z-score manual): z={z:.4f}, p-value={pval:.6f}")
            print(f"Taxa {top_year}: {p1:.4f}  | Taxa resto: {p2:.4f}")
        else:
            print("Erro: erro padrão zero ao calcular z (p_pool pode ser 0 ou 1).")
    else:
        print("Amostras muito pequenas para teste z.")
else:
    print("Hipótese A: colunas 'Year' ou 'Fatal (Y/N)' ausentes.")

if 'by_year' in locals():
    by_year.to_csv("shark_by_year_summary.csv")
    print("Resumo by_year salvo em 'shark_by_year_summary.csv'.")

# Hipótese B: existe um país/área com maior concentração de ataques?
if col_country in df.columns:
    country_counts = df[col_country].value_counts().rename_axis('country').reset_index(name='counts')
    top_country = country_counts.iloc[0]
    print(f"\nHipótese B -> País/Área com maior concentração: {top_country['country']} ({top_country['counts']} ataques)")
    print("\nTop 10 países/áreas com mais ataques:")
    print(country_counts.head(10).to_string(index=False))
else:
    print(" Hipótese B: coluna 'Country' ausente.")


# Hipótese C: há mais ataques provocados do que não provocados?
if col_type in df.columns:
    total_provoked = int((df[col_type] == 0).sum())
    total_unprovoked = int((df[col_type] == 1).sum())
    total_questionable = int((df[col_type] == 2).sum())
    print(f"\nHipótese C -> Provoked={total_provoked}, Unprovoked={total_unprovoked}, Questionable={total_questionable}")
    chi2, p, dof, ex = chi2_contingency([[total_provoked, total_unprovoked, total_questionable]])
    print(f"Chi² test: chi2={chi2:.3f}, p={p:.4f} (H0: proporções iguais)")
else:
    print(" Hipótese C: coluna 'Type' ausente.")


# Hipótese D & E: áreas com mais ataques provocados e não provocados
if col_area in df.columns and col_type in df.columns:
    type_by_area = df.groupby(col_area)[col_type].value_counts().unstack(fill_value=0)

    # lugar com mais ataques NÃO provocados
    if 1 in type_by_area.columns:
        most_unprov_area = type_by_area[1].idxmax()
        most_unprov_count = type_by_area[1].max()
        print(f"\nHipótese D -> Área com mais ataques NÃO provocados: {most_unprov_area} ({most_unprov_count})")

    # lugar com mais ataques PROVOCADOS
    if 0 in type_by_area.columns:
        most_prov_area = type_by_area[0].idxmax()
        most_prov_count = type_by_area[0].max()
        print(f"Hipótese E -> Área com mais ataques PROVOCADOS: {most_prov_area} ({most_prov_count})")

    # top 10 de cada tipo
    print("\nTop 10 áreas com mais ataques NÃO provocados:")
    print(type_by_area.sort_values(1, ascending=False).head(10)[[1]])

    print("\nTop 10 áreas com mais ataques PROVOCADOS:")
    print(type_by_area.sort_values(0, ascending=False).head(10)[[0]])
else:
    print(" Hipótese D/E: colunas 'Area' ou 'Type' ausentes.")


# Hipótese F: qual atividade tem mais chances de ataque e onde é mais seguro?
if col_activity in df.columns:
    act_counts = df[col_activity].value_counts().rename_axis('activity').reset_index(name='counts')
    top_activity = act_counts.iloc[0]
    print(f"\nHipótese F -> Atividade com mais ataques: {top_activity['activity']} ({top_activity['counts']})")

    # estatísticas por atividade
    min_cases = 20
    act_stats = df.groupby(col_activity).agg(
        total_attacks=(col_activity, 'count'),
        fatal_count=(col_fatal, 'sum') if col_fatal in df.columns else (col_activity, lambda s: 0),
        provoked_count=(col_type, lambda s: (s == 0).sum()),
        unprovoked_count=(col_type, lambda s: (s == 1).sum())
    )
    act_stats['fatal_rate'] = act_stats['fatal_count'] / act_stats['total_attacks']
    act_stats['prov_rate'] = act_stats['provoked_count'] / act_stats['total_attacks']
    act_stats_filtered = act_stats[act_stats['total_attacks'] >= min_cases].sort_values('total_attacks', ascending=False)

    print("\nAtividades com >= 20 casos (resumo):")
    print(act_stats_filtered[['total_attacks', 'fatal_rate', 'prov_rate']].head(10).to_string())

    # atividade mais segura (menor taxa de fatalidade)
    safest = act_stats_filtered['fatal_rate'].idxmin()
    print(f"\nAtividade mais segura: {safest} (fatal_rate={act_stats_filtered.loc[safest, 'fatal_rate']:.3f})")
else:
    print(" Hipótese F: coluna 'Activity' ausente.")

# hipotese G: idade mais ferrada
print("\nHipótese G -> Média de idade atacada por ano:")

age_by_year = None
if col_year in df.columns and col_age in df.columns:
    # média da idade por ano
    age_by_year = df.groupby(col_year)[col_age].mean().reset_index()
    age_by_year = age_by_year.rename(columns={col_age: "mean_age"})

    print(age_by_year.head(20).to_string(index=False))

    age_by_year.to_csv("shark_age_mean_by_year.csv", index=False)
    print("Salvo: 'shark_age_mean_by_year.csv'")
else:
    print("Hipótese G: colunas 'Year' ou 'Age' ausentes.")


#hipotese H: hora mais letal
print("\nHipótese H -> Hora com mais ataques provocados e acidentais (questionable = acidental):")
if col_time in df.columns and col_type in df.columns:
    # preparar df com Time inteiro (0-23). já assumimos que Time é numérico
    df_time = df.dropna(subset=[col_time, col_type]).copy()
    df_time[col_time] = df_time[col_time].astype(int)

    # definimos acidentais = unprovoked (1) + questionable (2)
    provoked = df_time[df_time[col_type] == 0]
    accidental = df_time[df_time[col_type].isin([1,2])]

    prov_by_hour = provoked.groupby(col_time).size()
    acc_by_hour = accidental.groupby(col_time).size()

    if not prov_by_hour.empty:
        hour_most_prov = int(prov_by_hour.idxmax())
        cnt_prov = int(prov_by_hour.max())
        print(f"Hora com mais ataques PROVOCADOS: {hour_most_prov}h ({cnt_prov} ataques)")
    else:
        print("Nenhum ataque provocado registrado com Time válido.")

    if not acc_by_hour.empty:
        hour_most_acc = int(acc_by_hour.idxmax())
        cnt_acc = int(acc_by_hour.max())
        print(f"Hora com mais ataques ACIDENTAIS (incluindo questionable): {hour_most_acc}h ({cnt_acc} ataques)")
    else:
        print("Nenhum ataque acidental/questionable registrado com Time válido.")

    # salvar contagens por hora
    prov_by_hour = prov_by_hour.rename("provoked_count").reset_index()
    acc_by_hour = acc_by_hour.rename("accidental_count").reset_index()
    by_hour = pd.merge(prov_by_hour, acc_by_hour, on=col_time, how='outer').fillna(0).sort_values(col_time)
    by_hour.to_csv("shark_attacks_by_hour_type.csv", index=False)
    print("Salvo: 'shark_attacks_by_hour_type.csv'")
else:
    print("Hipótese H: colunas 'Time' ou 'Type' ausentes.")

# Salvar resumos
if 'country_counts' in locals():
    country_counts.to_csv("shark_by_country_counts.csv")
if 'act_stats' in locals():
    act_stats.to_csv("shark_activity_stats.csv")

print("\nArquivos gerados (se disponíveis):")
for fname in [
    "shark_by_year_summary.csv",
    "shark_by_country_counts.csv",
    "shark_activity_stats.csv",
    "shark_age_mode_by_year.csv",
    "shark_attacks_by_hour_type.csv"
]:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            print(" -", fname)
    except FileNotFoundError:
        pass

print("\nFim da análise!")
