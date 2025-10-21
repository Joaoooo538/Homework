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

#========== Grafico de tabela de scatterplot (pairplot) ==========
#análise bivariada incondicional.
bd = pd.read_csv('winequality-red.csv', sep=';', quotechar='"')
bd.columns = bd.columns.str.strip()

preditores = bd.drop(columns='quality')
rotulo     = bd['quality']

sns.set(style="ticks", font_scale=1.0)
paleta = sns.color_palette("tab10", n_colors=rotulo.nunique())

g = sns.pairplot(bd,                    #dados para o plot
                 hue='quality',         #variavel que da as cores (rotulos)
                 diag_kind='hist',      #diagonal de histogramas
                 palette='plasma',      #paleta de cores [coolwarm, viridis, plasma, icefire] plasma ficou melhor na minha opiniao
                 height=2.5,
                 aspect=1.2,
                 corner=False)

for ax in g.axes.flatten():            #rotacao pra caber no grafico
    if ax is not None:                 #retirar os valores do rotulo que nao tem nenhuma amostra
        ax.set_xlabel(ax.get_xlabel(), rotation=45, ha='right')
        ax.set_ylabel(ax.get_ylabel(), rotation=45, ha='right')

#ajuste pra legenda dos rotulos sair de cima da figura
g.fig.tight_layout()                   # apertado pra ver se cabe
g.fig.subplots_adjust(right=0.9)       # espaço direita
g._legend.set_bbox_to_anchor((0.95, 0.5)) 
g._legend.set_title('Quality')

g.fig.savefig('pairplot_winequality.png', dpi=300, bbox_inches='tight') #save da figura no caminho do folder
plt.close(g.fig)
