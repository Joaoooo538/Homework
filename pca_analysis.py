# ARQUIVO: analise_item_5_pca.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler 

print("Iniciando a Análise Item 5 (PCA)")

caminho_arquivo = 'winequality-red.csv'

coluna_resposta = 'quality'
try:
    df = pd.read_csv(caminho_arquivo, sep=';')
    print("Dataset carregado com sucesso.\n")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{caminho_arquivo}' não foi encontrado.")
    exit()


X = df.drop(coluna_resposta, axis=1) 
y = df[coluna_resposta]            

print("Garantindo Média=0 e Desvio Padrão=1 para todos os 11 preditores.")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nVerificando a padronização (média e desvio padrão após escalar):")

print("Médias (devem ser próximas de 0):")

print(X_scaled_df.mean()) 

print("\nDesvios Padrão (devem ser próximos de 1):")

print(X_scaled_df.std())

print("\n-----------------------------------------------------")

print("Padronização concluída. Próximo passo: Matriz de Covariância.")

print("-----------------------------------------------------")

print("\n--- Calculando a Matriz de Covariância ---")

print("Como os dados estão padronizados, isso é equivalente à Matriz de Correlação.")

cov_matrix = np.cov(X_scaled.T) 

cov_matrix_df = pd.DataFrame(cov_matrix, columns=X.columns, index=X.columns)

print("\nMatriz de Covariância (Correlação) dos dados padronizados (11x11):")

print(cov_matrix_df)

output_dir_item5 = "resultados_pca"

if not os.path.exists(output_dir_item5):

    os.makedirs(output_dir_item5)

cov_matrix_df.to_csv(os.path.join(output_dir_item5, "matriz_covariancia_pca.csv"))

print(f"\nMatriz de Covariância salva em '{output_dir_item5}/matriz_covariancia_pca.csv'")

print("\n-----------------------------------------------------")

print("Matriz de Covariância calculada. Próximo passo: Autovalores e Autovetores.")

print("-----------------------------------------------------")

print("\n--- Calculando Autovalores e Autovetores ---")

print("Estes definem as direções (componentes principais) e a variância capturada.")

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = eigenvalues.argsort()[::-1] 

eigenvalues = eigenvalues[idx]    

eigenvectors = eigenvectors[:, idx] 

print("\nAutovalores (ordenados do maior para o menor):")

print(eigenvalues)

print("\nAutovetores (colunas ordenadas correspondentes aos autovalores):")

eigenvectors_df = pd.DataFrame(eigenvectors, 
                               index=X.columns,
                               columns=[f'PC{i+1}' for i in range(len(eigenvalues))]) # Colunas são os PCs
print(eigenvectors_df)

print("\n-----------------------------------------------------")

print("Autovalores e Autovetores calculados e ordenados.")

print("Próximo passo: Calcular Variância Explicada e Selecionar Componentes.")

print("-----------------------------------------------------")

print("\n--- Calculando Variância Explicada e Selecionando PCs ---")

total_variance = np.sum(eigenvalues) 

explained_variance_ratio = eigenvalues / total_variance 

cumulative_explained_variance = np.cumsum(explained_variance_ratio) 

print(f"\nVariância Total (Soma dos Autovalores): {total_variance:.4f}")

print("\nVariância Explicada por Componente:")

for i, ratio in enumerate(explained_variance_ratio):

    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print("\nVariância Explicada Acumulada:")

for i, cum_ratio in enumerate(cumulative_explained_variance):

    print(f"  Até PC{i+1}: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")

num_components_to_keep = 2
print(f"\nSelecionando os primeiros {num_components_to_keep} componentes principais para visualização.")

projection_matrix_W = eigenvectors[:, :num_components_to_keep]

print(f"\nMatriz de Projeção W (Autovetores de PC1 e PC2 como colunas, shape {projection_matrix_W.shape}):")

projection_matrix_df = pd.DataFrame(projection_matrix_W, 
                                    
                                     index=X.columns, 

                                     columns=[f'PC{i+1}' for i in range(num_components_to_keep)])

print(projection_matrix_df)

print("\n-----------------------------------------------------")

print("Variância explicada calculada e matriz de projeção criada.")

print("Próximo passo: Transformar (projetar) os dados originais.")

print("-----------------------------------------------------")

print("\n--- Projetando os Dados nos Componentes Principais ---")

print("Reduzindo de 11 dimensões para 2 dimensões (PC1 e PC2).")

X_pca = X_scaled @ projection_matrix_W 

print(f"\nDados transformados para o espaço PCA (shape {X_pca.shape}).")

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

pca_df['quality'] = y.values 

print("\nDataFrame final com PC1, PC2 e Quality pronto para visualização:")

print(pca_df.head()) # Mostra as primeiras linhas

print("\n-----------------------------------------------------")

print("Projeção dos dados concluída.")

print("Próximo passo: Visualizar PC1 vs PC2.")

print("-----------------------------------------------------")

print("\n--- Visualizando os Dados Projetados (PC1 vs PC2) ---")

plt.figure(figsize=(12, 8))

sns.scatterplot(
    x='PC1',              
    y='PC2',             
    hue='quality',       
    data=pca_df,         
    palette='viridis',    
    s=50,                 
    alpha=0.7            
)

plt.title('Dados Projetados nos 2 Primeiros Componentes Principais (PCA)')

plt.xlabel('Componente Principal 1 (PC1)')

plt.ylabel('Componente Principal 2 (PC2)')

plt.legend(title='Quality')

plt.grid(True, linestyle='--', alpha=0.6)

pca_plot_path = os.path.join(output_dir_item5, "pca_pc1_vs_pc2.png")

plt.savefig(pca_plot_path, dpi=300)

plt.close()

print(f"\nGráfico PCA (PC1 vs PC2) salvo em '{pca_plot_path}'")

print("\n-----------------------------------------------------")
