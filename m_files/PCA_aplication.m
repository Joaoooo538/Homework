%{%****************************************************************************
% * @autor:     Leonardo de F. V. dos Santos
% * @email:     leonardofvs@alu.ufc.br
% *
% * @curso:     Engenharia da Computação
% * @inst:      Universidade Federal do Ceará (UFC)
% * @depto:     Departamento de Engenharia de Teleinformática (DETI)
% *
% * @data:      [16/10/2025]
% * @arquivo:   [Pca_analisys.m]
% * @versao:    [1.0]
% *
% * @disciplina: Inteligência Computacional Aplicada
% *
% * @descrição: [Código fonte de geração de componentes de análise gráfica e,
% * aplicação manual da técnica de PCA]
% ***************************************************************************%}

%leitura dos dados e separacao
ds = readtable('winequality-red.csv');

% Pegar só os preditores (sem a coluna quality)
data = ds{:, 1:end-1};
quality = ds(:,end);
rotulos = ds.Properties.VariableNames;
[linhas,colunas] = size(data);


%%=========== CALCULOS DE ASSIMETRIA E PLOT DOS HISTOGRAMAS ================
%analise grafica e de valores importantes de preditores de forma isolada para
%investigação de técnicas a serem adotadas durante o tratamento dos dados.

assimetria = zeros(1,colunas);
figure(1);
for i = 1:colunas
    assimetria(i) = skewness(data(:, i));	%calculo das assimetrias por preditor via funcao nativa
  
    subplot(4,3,i); 			%organizacao dos subplots na mesma figura
    h1 = histfit(data(:,i)); title(rotulos(i));	%histograma dos dados por preditor
    
    %funcoes de ajuste estetico----
    h1(1).FaceColor = [0.6 .6 .9];
    h1(2).Color = [1 .4 .4];
    grid on;
    grid minor;
    %------------------------------
end


%%====== REALIZACA DO BOXCOX PARA MINIMIZAR EFEITO DE OUTLIERS =============
%graças a observacao dos dados fez-se necessário o uso de metodos de 
%transformacao para minimizar os efeitos dos outliers presentes nos dados.

data = data + 0.1;              		%para nao dar problema no boxcox <=0
transdat = zeros(size(data));   		%Matriz de dados que vai receber o boxcox

figure(2);
for i = 1:colunas
    [transdat(:,i),lambda(i)] = boxcox(data(:,i));	%aplicacao do metodo boxcox com labda automatico
   
    subplot(4,3,i);
    h2 = histfit(transdat(:,i)), title(rotulos(i));	%plot do histograma já ajustado pela transfomacao
    
    %funcoes de ajuste estetico----
    h2(1).FaceColor = [0.6 .6 .9];
    h2(2).Color = [1 .4 .4];
    grid on;
    grid minor;
    %------------------------------
end

%Valores do coeficiente de assimetria depois do boxcox. (melhores)

for i = 1:colunas
    assimetria_boxcox(i) = skewness(transdat(:, i));
end


%%========== NORMALIZACAO DOS DADOS E APLICACAO DO METODO PCA ==============
%para a aplicacao do metodo primeiro os dados foram normalizados para possu-
%-irem media 0 e desvio padrao 1, assim sendo possivel a analise por meio da
%matriz de covariancia

%normalizar o dataset pra poder fazer a matriz de covariancia certo (se fosse correlação nao precisava)
for i = 1:colunas
    transdat_norm(:,i) = ( transdat(:,i) - mean(transdat(:,i)) ) ./ (std(transdat(:,i)));
end

%Teste das métricas estatísticas
std(transdat_norm(:,1))
mean(transdat_norm(:,1))

%Calcular a matriz de covariancia dos dados normalizados
m_cov = cov(transdat_norm);

%Calcular os autovetores (PC's) e autovalores (variancia explicada)
[autovetores, autovalores] = eig(m_cov);

%preparacao para o screeplot
autovalores = diag(autovalores);			%pegar a diagonal importante dos autovalores
explained = 100 * autovalores / sum(autovalores); 	%fazer a variancia explicada percentual

%ordenacao dos autovalores e dos pc's com a mesma ordem de indice.
[autovalores_ord, indices_ord] = sort(autovalores, 'descend');
autovetores_ord = autovetores(:, indices_ord);
explained_ord = explained(indices_ord);

%plot do gráfico
figure(4);
screeplot = bar(explained_ord); 	 %grafico de barras barras
hold on;
%linha que conecta as barras
x = 1:length(explained_ord);
plot(x, explained_ord, '-', 'LineWidth', 2, 'Color', 'r');
xlabel('Componentes Principais');
ylabel('Variância Explicada (%)');
title('Variância explicada por PC');
grid on;
grid minor;
ylim([0,100]);
yticks(0:10:100);
hold off;


%%======== TRANSFORMACAO DE DIMENSIONALIDADE VIA PCA NOS DADOS =============
%Para analisar a eficacia dos dois primeiros preditores devemos observar a
%formacao de clusteres do mesmo tipo se formando em relacao a qualidade no
%scatterplot representado apenas pelas dimensoes resultantes do PCA (2).

PCs = autovetores_ord(:,1:2);		%selecao dos dois primeiros pcs ordenados (1 e 2)

data_reduzido = transdat_norm * PCs;%transformacao dos dados para a dimensao dos 2 pcs

%plot do gráfico
figure(5)
qualityplot = table2array(quality);
unique_values = unique(qualityplot);
unique_labels = string(unique_values);
quality_cat = categorical(qualityplot, unique_values, unique_labels);
g = gscatter(data_reduzido(:,1) , data_reduzido(:,2) , quality_cat, parula(length(unique_values)) , [] , 15); %dispersao colorido

xlabel('PC1');
ylabel('PC2');
title('PCA score plot');
grid on
grid minor
legend();

