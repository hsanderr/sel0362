% This function performs backpropagation SGD training of a neural network with one hidden layer.
% W1: weight matrix between the input layer and hidden layer.
% W2: weight matrix between the hidden layer and output layer.
% X: inputs for supervise

function [W1, W2] = BackpropXOR(W1, W2, X, D)
    alpha = 0.9;
    N = length(X);
    for k = 1:N
        x = X(k, :); % <-- tirei o transposto
        d = D(k);
        v1 = W1*x'; % <-- coloquei o transposto
        y1 = Sigmoid(v1)
        v = W2*y1; % <-- transposto
        y = Sigmoid(v);
        e = d - y;
        delta = y.*(1-y).*e;
        e1 = W2'*delta;
        delta1 = y1.*(1-y1).*e1;
        delta1
        x
        dW1 = alpha*delta1.*x;
        W1 = W1 + dW1;
        dW2 = alpha*delta*y1';
        W2 = W2 + dW2;
    end
end
