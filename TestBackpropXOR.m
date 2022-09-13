% This program calls in the BackpropXOR.m 
% function and trains the neural network max epoch times.

clear all
clc

% Inputs:
X = [0 0 1; 0 1 1; 1 0 1; 1 1 1];

% Desired outputs:
D = [0; 1; 1; 0];

% Initialization of weights:
W1 = 2*rand(4, 3) - 1;
W2 = 2*rand(4, 1) - 1;

% Training process (adjusting weights):
max_epoch = 10000;
for epoch = 1:max_epoch %train
    [W1, W2] = BackpropXOR(W1, W2, X, D);
end

%Inference:
N = size(X,1);
y = zeros(N,1);
for k = 1:N
    x = X(k, :)';
    v1 = W1*x;
    y1 = Sigmoid(v1);
    v = W2*y1;
    y(k) = Sigmoid(v); %obtained output.
end

disp('Results:');
disp('[desired neuron output]');
disp([D y]);