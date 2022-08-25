# SEL0362 - Exercício 1
## Henrique Sander Lourenço - 10802705

```matlab
clear all
clc

% Entrada
X = [
    0 0 0 1; 
    0 0 1 1; 
    0 1 1 1; 
    1 1 1 1; 
    1 0 0 1; 
    1 1 0 1
    ];
% Saída desejada
D = [0; 0; 1; 1; 0; 1];

% Pesos
W = 2*rand(1, 4) - 1;

% Treinamento da RN através do ajuste dos pesos (w)
% Época = 1000 - número de vezes que a IA é treinada
for epoch = 1:10000
    W = DeltaSGD(W, X, D);
end

% Inferência
N = 6;
y = zeros(N, 1);
for k = 1:N
    x = X(k,:)';
    v = W*x;
    y(k) = Sigmoid(v);
end

disp('Resultados:');
disp('');
disp('      D         Y');
disp([D y]);
disp('');
disp('     w1         w2        w3        w4');
disp([W(1) W(2) W(3) W(4)])

% Método Gradiente Descendente com função de ativação sigmoidal
function W = DeltaSGD(W, X, D)
    alpha = 0.9; % taxa de aprendizagem
    N = 6;
    for k = 1:N
        x = X(k,:)';
        d = D(k);
        v = W * x;
        y = Sigmoid(v);
        e = d - y;
        delta = y * (1 - y) * e;
        dW = alpha * delta * x;
        W(1) =  W(1) + dW(1);
        W(2) =  W(2) + dW(2);
        W(3) =  W(3) + dW(3);
    end
end

% Função sigmoidal
function y = Sigmoid(x)
    y = 1 / (1 + exp(-x));
end
```
