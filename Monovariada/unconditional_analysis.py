import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

caminho_arquivo = 'winequality-red.csv'

coluna_resposta = 'quality'

print(f"Iniciando a análise do dataset: '{caminho_arquivo}'")

try:

    df = pd.read_csv(caminho_arquivo, sep=';')

    print("Dataset carregado com sucesso.\n")

except FileNotFoundError:

    print(f"ERRO: O arquivo '{caminho_arquivo}' não foi encontrado.")

    print("Por favor, verifique o caminho e tente novamente.")

    exit() 

print("--- DESCRIÇÃO TÉCNICA DO DATASET ---")

N = df.shape[0]

print(f"\n[N] Número de Observações (amostras): {N}")

preditores = [col for col in df.columns if col != coluna_resposta]

D = len(preditores)

print(f"\n[D] Número de Preditores (características): {D}")

print("   Lista de Preditores:")

for i, pred in enumerate(preditores):

    print(f"   {i+1:2d}. {pred} (Tipo: {df[pred].dtype})")

print(f"\nVariável de Resposta (Classe): '{coluna_resposta}' (Tipo: {df[coluna_resposta].dtype})")

classes = df[coluna_resposta].unique()

classes.sort()

L = len(classes)

print(f"\n[L] Número de Classes (valores únicos de qualidade): {L}")

print(f"   Classes identificadas: {classes}")

print("\nDistribuição das Classes (Contagem por amostra):")


distribuicao_classes = df[coluna_resposta].value_counts().sort_index()

print(distribuicao_classes)

print("\n   -> Fato Notável: O dataset é altamente desbalanceado.")

print(f"      As classes {distribuicao_classes.idxmax()} e {distribuicao_classes.nlargest(2).index[1]} dominam o conjunto.")

print("\nVerificação de Dados Faltantes (por coluna):")

dados_faltantes = df.isnull().sum()

print(dados_faltantes)

if dados_faltantes.sum() == 0:

    print("\n   -> Fato Notável: Não há dados faltantes no dataset.")

else:

    print(f"\n   -> Foram encontrados {dados_faltantes.sum()} dados faltantes.")

predictors_df = df.drop(coluna_resposta, axis=1)

response_series = df[coluna_resposta]

print("\n--- CÁLCULOS DA ANÁLISE INCONDICIONAL ---")

print("Calculando Média, Desvio Padrão e Assimetria para todos os N=1599 vinhos.")

estatisticas_descritivas = predictors_df.describe()

estatisticas_assimetria = predictors_df.skew()

tabela_item2 = estatisticas_descritivas.T

tabela_item2['skewness'] = estatisticas_assimetria

colunas_pedidas = ['mean', 'std', 'skewness']

print("\n--- Tabela Resumo para o Relatório (Item 2) ---")

print(tabela_item2[colunas_pedidas])

print("\n--- GERANDO GRÁFICOS DO ITEM 2 (Análise Incondicional) ---")

script_dir = os.path.dirname(os.path.abspath(__file__)) 

nome_subpasta_graficos = "graficos_incondicional"

output_dir_incondicional = os.path.join(script_dir, nome_subpasta_graficos)

if not os.path.exists(output_dir_incondicional):

    os.makedirs(output_dir_incondicional)

    print(f"Pasta criada em: '{output_dir_incondicional}'")  

print(f"Salvando {len(predictors_df.columns) * 2} gráficos em '{output_dir_incondicional}'...")

for column in predictors_df.columns:

    plt.figure(figsize=(10, 6)) 

    sns.histplot(predictors_df[column], kde=True, bins=30) 

    plt.title(f'Histograma de {column} (Incondicional)')

    plt.xlabel(column)

    plt.ylabel('Frequência')
   
    plt.savefig(os.path.join(output_dir_incondicional, f'{column}_histograma.png')) 

    plt.close()

    plt.figure(figsize=(10, 6))

    sns.boxplot(x=predictors_df[column])

    plt.title(f'Box-plot de {column} (Incondicional)')

    plt.xlabel(column)

    plt.savefig(os.path.join(output_dir_incondicional, f'{column}_boxplot.png')) 
    plt.close()

print("\n-----------------------------------------------------")

print(f"Gráficos Incondicionais concluídos e salvos em '{output_dir_incondicional}'.")

print("-----------------------------------------------------")