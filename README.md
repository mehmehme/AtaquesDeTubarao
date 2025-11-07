<h1 align="center">
  🐻 Análise de Dados sobre Ataques de Tubarões 🐾  
  <br>
  <img src="https://media.tenor.com/JQFOQPsvJG4AAAAM/shark-cute.gif" alt="Cute Bears" width="250">
</h1>

---

## 📘 **Descrição do Projeto**

Este projeto tem como objetivo consolidar as técnicas e métodos estudados ao longo da disciplina de **Ciência de Dados**, por meio da análise de um conjunto de dados real e do compartilhamento de análises em diferentes contextos.

A base de dados escolhida foi a do Kaggle:  
🔗 [Global Shark Attack Dataset - Kaggle]([https://www.kaggle.com/datasets/mexwell/global-shark-attack])

O trabalho foi desenvolvido seguindo as etapas semanais de análise, tratamento e modelagem dos dados.

---

## 🎯 **Objetivo Geral**

Aplicar as etapas do processo de **Ciência de Dados** — desde a escolha da base até a apresentação dos resultados — para extrair informações relevantes sobre os incidentes de ataques de tubarões, compreender padrões e testar hipóteses explicativas e preditivas.

---

##  **Especificação**

O projeto foi dividido em **cinco etapas principais**, conforme as orientações da disciplina:

---

###  **Etapa 1 – Escolha da Base de Dados e Análise Inicial (07/11/2025 à 14/11/2025)**

Nesta etapa, selecionamos a base **Global Shark Attack**, que contém registros de ataques de tubarões reportados em diferentes regiões.  
A base atende aos requisitos de variabilidade de dados e quantidade mínima de instâncias.

**Análises realizadas:**
- Descrição completa da base (autor, fonte, estrutura e objetivo);
- Identificação dos tipos de variáveis (quantitativas discretas e contínuas, qualitativas nominais e ordinais);
- Definição de **cinco hipóteses** para investigação, abordando análises:
  - **Exploratórias:** distribuição de espécies e localização dos incidentes;
  - **Explicativas:** influência do tipo de animal na gravidade do incidente;
  - **Preditivas:** previsão do risco de mordidas de acordo com variáveis demográficas e ambientais.

---

###  **Etapa 2 – Qualidade dos Dados (14/11/2025 à 21/11/2025)**

Foram avaliadas e tratadas as questões de **qualidade dos dados**, incluindo:
1. Estatísticas descritivas (média, mediana, moda, variância, amplitude, valores distintos);
2. Verificação da normalidade das distribuições com histogramas, boxplots e Q-Q plots;
3. Identificação e tratamento de **outliers** via **Isolation Forest**;
4. Determinação das variáveis dependentes e independentes conforme as hipóteses;
5. Análise de **valores faltantes**, verificando sua relação com as classes do problema;
6. Aplicação de **limpeza adicional** conforme as características observadas.

---

###  **Etapa 3 – Transformações e Análise Exploratória (21/11/2025 à 28/11/2025)**

Nesta etapa, tratamos os valores faltosos por meio de **modelos indutores** (regressão linear e logística).  
Realizamos as seguintes análises:

- **Correlação entre variáveis quantitativas**;
- **Redução de dimensionalidade (PCA)** para visualização e interpretação dos dados;
- **Análise de agrupamentos (k-Means)** com determinação da quantidade ótima de clusters via **método do cotovelo**;
- Interpretação dos grupos identificados no contexto do problema, revelando padrões de incidentes por tipo de animal e localização.

---

###  **Etapa 4 – Análise Preditiva e Explicativa (28/11/2025 à 05/12/2025)**

Definimos uma variável alvo para a predição: **país com maior numero de acidentes**.  
As análises realizadas incluíram:

- Modelagem estatística de **regressão** para verificar relações entre as variáveis preditoras e a variável alvo;
- Avaliação da **generalização** dos modelos com **cross-validation**;
- Exibição de **matriz de confusão** e **gráficos de dispersão** para validar a performance;
- Interpretação dos resultados no contexto real dos incidentes de mordidas.

---

###  **Etapa 5 – Apresentação dos Resultados (05/12/2025)**

Foi realizada uma **apresentação** com os seguintes destaques:
- Principais hipóteses testadas e resultados obtidos;
- Dificuldades encontradas e estratégias de solução;
- Visualizações interativas e conclusões extraídas das análises;
- Discussão sobre possíveis aplicações práticas e limitações da base.

---

##  **Considerações Finais**

- As análises foram desenvolvidas e entregues em formato **Jupyter Notebook (.ipynb)**;
- Todas as etapas geraram **insights e conhecimentos**, mesmo que inconclusivos;
- A base de dados foi submetida à **validação do professor** conforme o cronograma da disciplina.

---

## 👥 **Equipe**
- **Elisa Nascimento dos Santos**

---

##  **Referências**
- MexWell. *Global Shark Attack.* Kaggle, 2018.  
  [https://www.kaggle.com/datasets/mexwell/global-shark-attack)](https://www.kaggle.com/datasets/mexwell/global-shark-attack)
- Documentação e materiais da disciplina de **Ciência de Dados (2025)**.

---

<h3 align="center"> “Cada dado conta uma história — basta sabermos ouvi-lo.” </h3>

