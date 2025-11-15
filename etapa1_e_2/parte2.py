import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

df = pd.read_csv("sharkattack_tratado.csv")

col_year = 'Year'
col_fatal = 'Fatal (Y/N)'
col_country = 'Country'
col_activity = 'Activity'
col_type = 'Type'
col_area = 'Area'

# Hipótese A: existe um ano que foi o mais fatal?
if col_year in df.columns and col_fatal in df.columns:
    by_year = df.groupby(col_year)[col_fatal].agg(['sum', 'count']).rename(columns={'sum': 'fatal_count', 'count': 'total'})
    by_year['fatal_rate'] = by_year['fatal_count'] / by_year['total']
    top_year = by_year['fatal_count'].idxmax()
    print(f"Hipótese A -> Ano com mais ataques fatais: {top_year}")
    print(by_year.loc[top_year])
    
    # teste estatístico de proporção (z-test)
    year_fatals = int(by_year.loc[top_year, 'fatal_count'])
    year_total = int(by_year.loc[top_year, 'total'])
    global_fatals = int(by_year['fatal_count'].sum()) - year_fatals
    global_total = int(by_year['total'].sum()) - year_total
    count = np.array([year_fatals, global_fatals])
    nobs = np.array([year_total, global_total])
    stat, pval = proportions_ztest(count, nobs)
    print(f"Teste proporção (ano vs resto): z={stat:.3f}, p={pval:.4f}")
else:
    print("Hipótese A: colunas 'Year' ou 'Fatal (Y/N)' ausentes.")


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


# Salvar resumos
if 'by_year' in locals():
    by_year.to_csv("shark_by_year_summary.csv")
if 'country_counts' in locals():
    country_counts.to_csv("shark_by_country_counts.csv")
if 'act_stats' in locals():
    act_stats.to_csv("shark_activity_stats.csv")

print("\n Arquivos de resumo gerados (se disponíveis):")
print(" - shark_by_year_summary.csv")
print(" - shark_by_country_counts.csv")
print(" - shark_activity_stats.csv")
print("\n Fim da análise!")
