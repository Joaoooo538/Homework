%%====================== Scatterplot com gscatter =========================
data = readtable('winequality-red.csv');
%variaveis pro plot
varX = 'totalSulfurDioxide';
varY = 'freeSulfurDioxide';
%selecao
x = data{:,varX};
y = data{:,varY};
quality = data.quality;

%scatterplot por qualidade
figure; hold on;
h = gscatter(x, y, quality, lines(max(quality)-min(quality)+1), 'o', 3, 'filled'); %gscatter pra separacao automatica dos rotulos

%plot da linha de tendência
p = polyfit(x, y, 1);
yfit = polyval(p, x);
plot(x, yfit, 'k-', 'LineWidth', 2);

%título e eixos estetico ----------------
xlabel(varX, 'Interpreter','none');
ylabel(varY, 'Interpreter','none');
title([varX, ' vs ', varY, ' por Qualidade']);
grid on;
%----------------------------------------

%cria legenda manualmente
legend(h, arrayfun(@(q) sprintf('Qualidade %d', q), unique(quality), 'UniformOutput', false));

hold off;
