import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

caminho_arquivo = 'winequality-red.csv'

coluna_resposta = 'quality'

print(f"Iniciando a Análise Item 3 (Condicional)")

print(f"Carregando dataset: '{caminho_arquivo}'...")

try:
    df = pd.read_csv(caminho_arquivo, sep=';')

    print("Dataset carregado com sucesso.\n")

except FileNotFoundError:

    print(f"ERRO: O arquivo '{caminho_arquivo}' não foi encontrado.")

    print("Por favor, verifique o caminho e tente novamente.")

    exit() 

predictors_df = df.drop(coluna_resposta, axis=1)

print("\n--- Item 3: ANÁLISE CONDICIONAL ---")

print(f"Calculando Média, Desvio Padrão e Assimetria para cada classe.")

agrupado_por_classe = df.groupby(coluna_resposta)

media_condicional = agrupado_por_classe.mean()

print("\n--- Médias Condicionais (μ_d|l) ---")

print(media_condicional)

std_condicional = agrupado_por_classe.std()

print("\n--- Desvios Padrão Condicionais (σ_d|l) ---")

print(std_condicional)

skew_condicional = agrupado_por_classe.apply(lambda x: x.drop(coluna_resposta, axis=1).skew())

print("\n--- Assimetrias Condicionais (γ_d|l) ---")

print(skew_condicional)

script_dir = os.path.dirname(os.path.abspath(__file__)) 

nome_subpasta_graficos = "tabelas_condicional"

output_dir_item3_tabelas = os.path.join(script_dir, nome_subpasta_graficos)

if not os.path.exists(output_dir_item3_tabelas):

    os.makedirs(output_dir_item3_tabelas)

media_condicional.to_csv(os.path.join(output_dir_item3_tabelas, "media_condicional.csv"))

std_condicional.to_csv(os.path.join(output_dir_item3_tabelas, "std_condicional.csv"))

skew_condicional.to_csv(os.path.join(output_dir_item3_tabelas, "skew_condicional.csv"))

print("\n-----------------------------------------------------")

print(f"Tabelas de cálculos do Item 3 salvas em '{output_dir_item3_tabelas}'.")

print("-----------------------------------------------------")

print("\n--- Gerando Gráficos (Item 3) ---")

script_dir = os.path.dirname(os.path.abspath(__file__)) 

nome_subpasta_graficos = "graficos_condicional"

output_dir_item3_graficos = os.path.join(script_dir, nome_subpasta_graficos)

if not os.path.exists(output_dir_item3_graficos):

    os.makedirs(output_dir_item3_graficos)

print(f"Salvando {len(predictors_df.columns) * 2} gráficos comparativos em '{output_dir_item3_graficos}'...")

for column in predictors_df.columns:

    plt.figure(figsize=(12, 7))

    sns.boxplot(x=coluna_resposta, y=column, data=df, palette="viridis")

    plt.title(f'Box-plot de {column} (Condicional por Qualidade)')

    plt.xlabel('Qualidade (Classe)')

    plt.ylabel(f'Valor de {column}')

    plt.savefig(os.path.join(output_dir_item3_graficos, f'{column}_boxplot_condicional.png'))

    plt.close()

    plt.figure(figsize=(12, 7))

    sns.kdeplot(data=df, x=column, hue=coluna_resposta, 
                
                fill=True, common_norm=False, palette="viridis", alpha=0.1)
    
    plt.title(f'Distribuição de {column} (Condicional por Qualidade)')

    plt.xlabel(f'Valor de {column}')

    plt.ylabel('Densidade')

    plt.savefig(os.path.join(output_dir_item3_graficos, f'{column}_densidade_condicional.png'))

    plt.close()

print("\n-----------------------------------------------------")

print(f"Análises concluídas.")

print(f"Gráficos salvos em '{output_dir_item3_graficos}'.")

print("-----------------------------------------------------")

