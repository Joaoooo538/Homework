# ===============================================================
# Universidade Federal do Ceará (UFC)
# Departamento de Engenharia de Computação
# Curso: Engenharia de Computação
# Disciplina: Inteligencia Computacional Aplicada
# Autor: Leonardo de Freitas V.
# ===============================================================

import pandas as pd
import seaborn as sns  #graficos mais estilizados junto com o plt
import matplotlib.pyplot as plt
import os

#========== Grafico de tabela de correlacao ==========
#analise bivariada incondicional.
#carregando os dados e separando---------------------------------
bd = pd.read_csv('winequality-red.csv', sep=';', quotechar='"')
bd.columns = bd.columns.str.strip()
preditores = bd.drop(columns='quality')
script_dir = os.path.dirname(os.path.abspath(__file__))
#----------------------------------------------------------------


corr = preditores.corr()                                #criacao da matriz de correlacao com os preditores
fig, ax = plt.subplots(figsize=(10,10))                 #tamanho da imagem a ser criada
cax = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm') #plot da matriz
fig.colorbar(cax, fraction=0.046, pad=0.04)             #barra colorida de heatmap lateral escalada para o tamanho do grafico


#ajuste dos ticks
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, ha='right', fontsize=10)
ax.set_yticklabels(corr.columns, fontsize=10)

#coloca os valores numericos da correlacao dentro de cada celula-
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=9)
#----------------------------------------------------------------

plt.tight_layout() #tight layout pra ajustar melhor o grafico no espaco

image_filename = "Correlation_matrix_with_heatmap.png"

full_save_path = os.path.join(script_dir, image_filename)

plt.savefig(full_save_path, dpi=300)

print(f"Matriz de Correlação salva em: '{full_save_path}'")

plt.close()